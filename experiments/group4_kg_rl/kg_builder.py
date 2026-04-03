"""
Knowledge Graph construction and commonsense verification.

Extracts (subject, relation, object) triples from news text and
verifies them against external knowledge graphs.
"""

import re
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class Triple:
    subject: str
    relation: str
    obj: str
    confidence: float = 1.0


@dataclass
class KGVerification:
    triple: Triple
    verified: bool
    kg_score: float  # 0=no match, 1=strong match
    source: str = ""
    details: str = ""


class TripleExtractor:
    """Extract (subject, relation, object) triples from text."""

    LLM_PROMPT = (
        "Extract factual triples (subject, relation, object) from the text below. "
        "Return a JSON list of objects with keys 'subject', 'relation', 'object'. "
        "Focus on verifiable factual claims. Return at most 8 triples.\n\n"
        "Text:\n{text}\n\nRespond ONLY with valid JSON array."
    )

    def __init__(self, method: str = "heuristic", llm_client=None, model: str = "gpt-4o-mini"):
        self.method = method
        self.llm_client = llm_client
        self.model = model

    def extract(self, text: str) -> List[Triple]:
        if self.method == "llm" and self.llm_client is not None:
            return self._extract_llm(text)
        return self._extract_heuristic(text)

    def _extract_llm(self, text: str) -> List[Triple]:
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.LLM_PROMPT.format(text=text[:2000])}],
                temperature=0.0,
                max_tokens=1000,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())
                return [
                    Triple(
                        subject=item["subject"],
                        relation=item["relation"],
                        obj=item["object"],
                    )
                    for item in items[:8]
                ]
        except Exception as e:
            print(f"  [TripleExtractor] LLM error: {e}")
        return self._extract_heuristic(text)

    def _extract_heuristic(self, text: str) -> List[Triple]:
        """Simple pattern-based triple extraction."""
        triples = []
        sentences = re.split(r"[.!?]+", text)

        patterns = [
            # "X is Y"
            (r"([A-Z][a-zA-Z\s]+?)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*([a-zA-Z\s]+)",
             "is_a"),
            # "X [verb] Y"
            (r"([A-Z][a-zA-Z\s]+?)\s+(announced|said|reported|claimed|confirmed|denied|launched|acquired|bought|sold)\s+(.+)",
             "action"),
            # "X located in Y"
            (r"([A-Z][a-zA-Z\s]+?)\s+(?:is\s+)?(?:located|based|headquartered)\s+in\s+([A-Z][a-zA-Z\s]+)",
             "located_in"),
        ]

        for sent in sentences[:20]:
            sent = sent.strip()
            if len(sent) < 15:
                continue

            for pattern, rel_type in patterns:
                match = re.search(pattern, sent)
                if match:
                    groups = match.groups()
                    if rel_type == "is_a" and len(groups) >= 2:
                        triples.append(Triple(
                            subject=groups[0].strip()[:50],
                            relation="is_a",
                            obj=groups[1].strip()[:50],
                        ))
                    elif rel_type == "action" and len(groups) >= 3:
                        triples.append(Triple(
                            subject=groups[0].strip()[:50],
                            relation=groups[1].strip(),
                            obj=groups[2].strip()[:80],
                        ))
                    elif rel_type == "located_in" and len(groups) >= 2:
                        triples.append(Triple(
                            subject=groups[0].strip()[:50],
                            relation="located_in",
                            obj=groups[1].strip()[:50],
                        ))

                    if len(triples) >= 8:
                        return triples

        return triples


class ConceptNetChecker:
    """Verify triples against ConceptNet API."""

    BASE_URL = "http://api.conceptnet.io"

    def verify(self, triple: Triple) -> KGVerification:
        """Check if a triple is supported by ConceptNet."""
        try:
            query = f"/query?node=/c/en/{self._normalize(triple.subject)}&other=/c/en/{self._normalize(triple.obj)}"
            url = self.BASE_URL + query
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            edges = data.get("edges", [])
            if edges:
                max_weight = max(e.get("weight", 0) for e in edges)
                return KGVerification(
                    triple=triple,
                    verified=max_weight > 1.0,
                    kg_score=min(max_weight / 5.0, 1.0),
                    source="conceptnet",
                    details=f"Found {len(edges)} edges, max_weight={max_weight:.2f}",
                )

            return KGVerification(
                triple=triple, verified=False, kg_score=0.0,
                source="conceptnet", details="No matching edges found",
            )

        except Exception as e:
            return KGVerification(
                triple=triple, verified=False, kg_score=0.0,
                source="conceptnet", details=f"API error: {e}",
            )

    def _normalize(self, text: str) -> str:
        return urllib.parse.quote(
            re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).strip().replace(" ", "_")
        )

    def verify_all(self, triples: List[Triple]) -> List[KGVerification]:
        return [self.verify(t) for t in triples]


class WikidataChecker:
    """Verify entity existence and relations via Wikidata."""

    SEARCH_URL = "https://www.wikidata.org/w/api.php"

    def verify_entity(self, entity_name: str) -> Tuple[bool, str]:
        """Check if an entity exists on Wikidata."""
        try:
            params = urllib.parse.urlencode({
                "action": "wbsearchentities",
                "search": entity_name,
                "language": "en",
                "format": "json",
                "limit": 1,
            })
            url = f"{self.SEARCH_URL}?{params}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            results = data.get("search", [])
            if results:
                return True, results[0].get("description", "")
            return False, ""

        except Exception:
            return False, ""

    def verify(self, triple: Triple) -> KGVerification:
        subj_exists, subj_desc = self.verify_entity(triple.subject)
        obj_exists, obj_desc = self.verify_entity(triple.obj)

        score = 0.0
        if subj_exists:
            score += 0.5
        if obj_exists:
            score += 0.5

        return KGVerification(
            triple=triple,
            verified=subj_exists and obj_exists,
            kg_score=score,
            source="wikidata",
            details=f"subject={'found' if subj_exists else 'not found'} "
                    f"({subj_desc[:50]}), "
                    f"object={'found' if obj_exists else 'not found'} "
                    f"({obj_desc[:50]})",
        )


class CommonsenseChecker:
    """High-level commonsense verification combining multiple KG sources."""

    def __init__(self, sources: Optional[List[str]] = None):
        self.sources = sources or ["conceptnet"]
        self.checkers = {}
        if "conceptnet" in self.sources:
            self.checkers["conceptnet"] = ConceptNetChecker()
        if "wikidata" in self.sources:
            self.checkers["wikidata"] = WikidataChecker()

    def check(self, triples: List[Triple]) -> Dict:
        """
        Check all triples against KG sources.

        Returns:
            Dictionary with 'verifications', 'violation_score',
            'verified_ratio', 'details'.
        """
        all_verifications = []

        for triple in triples:
            for name, checker in self.checkers.items():
                v = checker.verify(triple)
                all_verifications.append(v)

        if not all_verifications:
            return {
                "verifications": [],
                "violation_score": 0.5,
                "verified_ratio": 0.0,
                "details": "No triples to verify",
            }

        verified_count = sum(1 for v in all_verifications if v.verified)
        total = len(all_verifications)
        avg_score = sum(v.kg_score for v in all_verifications) / total

        # violation_score: 1 = many violations, 0 = all verified
        violation_score = 1.0 - avg_score

        return {
            "verifications": all_verifications,
            "violation_score": violation_score,
            "verified_ratio": verified_count / total,
            "details": f"{verified_count}/{total} triples verified, "
                       f"avg_score={avg_score:.2f}",
        }
