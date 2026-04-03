"""
Multi-agent fake news verification pipeline.

Four agents work together:
1. ClaimExtractor - Extract factual claims from news articles
2. WebSearcher   - Search the web for evidence
3. EvidenceScorer - Score (claim, evidence) pairs with NLI
4. JudgeAgent    - Aggregate evidence and make final prediction
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

import torch


class _Message:
    def __init__(self, content: str):
        self.content = content

class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)

class _Response:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]

class _Completions:
    def __init__(self, client):
        self._client = client

    def create(self, model=None, messages=None, temperature=0.0,
               max_tokens=512, **kwargs):
        prompt = messages[-1]["content"] if messages else ""
        text = self._client.generate(prompt, max_new_tokens=min(max_tokens, 512))
        return _Response(text)

class _Chat:
    def __init__(self, client):
        self.completions = _Completions(client)

class LocalLLMClient:
    """Drop-in replacement for openai.OpenAI() using a local HF model."""

    def __init__(self, model_name="google/flan-t5-base", device=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(f"cuda:{device}" if isinstance(device, int) and device >= 0 else "cpu")
        label = "GPU" if self._device.type == "cuda" else "CPU"
        print(f"  [LocalLLM] Loading '{model_name}' on {label}...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self._device)
        self._model.eval()
        print(f"  [LocalLLM] Ready")
        self.chat = _Chat(self)

    def generate(self, prompt, max_new_tokens=512):
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=512).to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens,
                                           do_sample=False)
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)


@dataclass
class Claim:
    text: str
    importance: float = 1.0


@dataclass
class Evidence:
    snippet: str
    source_url: str = ""
    relevance: float = 0.0


@dataclass
class NLIScore:
    claim: str
    evidence: str
    entailment: float = 0.0
    contradiction: float = 0.0
    neutral: float = 0.0


@dataclass
class VerificationResult:
    article_text: str
    claims: List[Claim]
    evidence_map: Dict[str, List[Evidence]]
    nli_scores: List[NLIScore]
    prediction: int  # 0=fake, 1=real
    confidence: float = 0.0
    reasoning: str = ""


class ClaimExtractor:
    """Extract key factual claims from a news article using an LLM."""

    PROMPT_TEMPLATE = (
        "Extract the key factual claims from the following news article. "
        "Return a JSON list of objects with 'text' (the claim) and "
        "'importance' (0-1 score). Return at most 5 claims.\n\n"
        "Article:\n{article}\n\n"
        "Respond ONLY with valid JSON array."
    )

    def __init__(self, llm_client=None, model: str = "gpt-4o-mini"):
        self.llm_client = llm_client
        self.model = model
        self._llm_failed = False

    def extract(self, article: str) -> List[Claim]:
        """Extract claims using LLM or fall back to heuristic."""
        if self.llm_client is not None and not self._llm_failed:
            return self._extract_with_llm(article)
        return self._extract_heuristic(article)

    def _extract_with_llm(self, article: str) -> List[Claim]:
        prompt = self.PROMPT_TEMPLATE.format(article=article[:3000])
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())
                return [
                    Claim(text=item["text"], importance=item.get("importance", 1.0))
                    for item in items[:5]
                ]
        except Exception as e:
            if not self._llm_failed:
                print(f"  [ClaimExtractor] LLM unavailable: {e}")
                print(f"  [ClaimExtractor] Switching to heuristic mode for all remaining samples")
                self._llm_failed = True
        return self._extract_heuristic(article)

    def _extract_heuristic(self, article: str) -> List[Claim]:
        """Simple sentence-based claim extraction as fallback."""
        sentences = re.split(r"[.!?]+", article)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

        claims = []
        for sent in sentences[:5]:
            importance = 1.0 if any(
                kw in sent.lower()
                for kw in ["said", "reported", "according", "announced", "claimed",
                           "confirmed", "revealed", "stated", "million", "billion",
                           "percent", "killed", "arrested", "elected"]
            ) else 0.5
            claims.append(Claim(text=sent[:200], importance=importance))

        return claims


class WebSearcher:
    """Search the web for evidence related to claims."""

    def __init__(self, search_fn=None, top_k: int = 5):
        self.search_fn = search_fn
        self.top_k = top_k
        self._search_disabled = False
        self._fallback_fn = None

    def search(self, claim: Claim) -> List[Evidence]:
        """Search for evidence. Uses provided search_fn or returns empty."""
        if self.search_fn is not None and not self._search_disabled:
            return self._search_api(claim)
        if self._fallback_fn is not None:
            return self._search_with_fallback(claim)
        return self._search_stub(claim)

    def _search_api(self, claim: Claim) -> List[Evidence]:
        try:
            results = self.search_fn(claim.text, max_results=self.top_k)
            if results:
                return [
                    Evidence(
                        snippet=r.get("content", r.get("snippet", "")),
                        source_url=r.get("url", ""),
                        relevance=r.get("score", 0.5),
                    )
                    for r in results[:self.top_k]
                ]
            return self._search_stub(claim)
        except Exception as e:
            if not self._search_disabled:
                print(f"  [WebSearcher] Primary search failed: {e}")
                self._try_fallback_search()
            if self._fallback_fn is not None:
                return self._search_with_fallback(claim)
            return self._search_stub(claim)

    def _try_fallback_search(self):
        """Try to set up DuckDuckGo as fallback when primary search fails."""
        try:
            import warnings
            warnings.filterwarnings("ignore", message=".*duckduckgo_search.*renamed.*")
            from duckduckgo_search import DDGS
            _ddgs_instance = DDGS()
            def ddg_search(query, max_results=5):
                return [{"content": r.get("body", ""), "url": r.get("href", ""), "score": 0.5}
                        for r in _ddgs_instance.text(query, max_results=max_results)]
            self._fallback_fn = ddg_search
            print(f"  [WebSearcher] Switched to DuckDuckGo (free)")
        except ImportError:
            print(f"  [WebSearcher] DuckDuckGo not installed, using stub mode")
            self._fallback_fn = None
        self._search_disabled = True

    def _search_with_fallback(self, claim: Claim) -> List[Evidence]:
        try:
            results = self._fallback_fn(claim.text, max_results=self.top_k)
            return [
                Evidence(snippet=r.get("content", ""), source_url=r.get("url", ""),
                         relevance=r.get("score", 0.5))
                for r in results[:self.top_k] if r.get("content")
            ]
        except Exception:
            return self._search_stub(claim)

    def _search_stub(self, claim: Claim) -> List[Evidence]:
        """Stub for when no search API is configured."""
        return [Evidence(snippet="[No search API configured]", relevance=0.0)]

    def search_all(self, claims: List[Claim]) -> Dict[str, List[Evidence]]:
        return {claim.text: self.search(claim) for claim in claims}


def _mnli_label_indices(id2label: Dict[int, str]) -> Tuple[int, int, int]:
    """Map HuggingFace MNLI id2label to (contradiction, neutral, entailment) indices."""
    lower = {i: (v or "").lower() for i, v in id2label.items()}
    rev = {v: i for i, v in lower.items()}
    c = rev.get("contradiction", 0)
    n = rev.get("neutral", 1)
    e = rev.get("entailment", 2)
    return c, n, e


class EvidenceScorer:
    """Score (claim, evidence) pairs using a 3-way NLI model (premise=evidence, hypothesis=claim)."""

    def __init__(
        self,
        nli_pipeline=None,
        nli_model=None,
        nli_tokenizer=None,
        nli_device=None,
    ):
        self.nli_pipeline = nli_pipeline
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.nli_device = nli_device
        self._nli_cne: Optional[Tuple[int, int, int]] = None
        if nli_model is not None and hasattr(nli_model, "config"):
            cfg = nli_model.config
            if getattr(cfg, "id2label", None):
                self._nli_cne = _mnli_label_indices(dict(cfg.id2label))

    def score(self, claim: str, evidence: str) -> NLIScore:
        if self.nli_model is not None and self.nli_tokenizer is not None:
            return self._score_nli_mnli(claim, evidence)
        if self.nli_pipeline is not None:
            return self._score_nli(claim, evidence)
        return self._score_heuristic(claim, evidence)

    def _score_nli_mnli(self, claim: str, evidence: str) -> NLIScore:
        """Standard MNLI: premise=evidence snippet, hypothesis=claim."""
        try:
            claim_trunc = claim[:512]
            evidence_trunc = evidence[:512]
            tok = self.nli_tokenizer(
                evidence_trunc,
                claim_trunc,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            dev = self.nli_device or torch.device("cpu")
            tok = {k: v.to(dev) for k, v in tok.items()}
            with torch.no_grad():
                logits = self.nli_model(**tok).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
            if self._nli_cne is None or len(probs) < 3:
                return NLIScore(claim=claim, evidence=evidence, neutral=1.0)
            ci, ni, ei = self._nli_cne
            return NLIScore(
                claim=claim,
                evidence=evidence,
                contradiction=float(probs[ci]),
                neutral=float(probs[ni]),
                entailment=float(probs[ei]),
            )
        except Exception as e:
            if not hasattr(self, "_nli_error_shown"):
                print(f"  [EvidenceScorer] MNLI error: {e}")
                self._nli_error_shown = True
            return NLIScore(claim=claim, evidence=evidence, neutral=1.0)

    def _score_nli(self, claim: str, evidence: str) -> NLIScore:
        """Legacy zero-shot path (misaligned with MNLI); prefer nli_model + tokenizer."""
        try:
            claim_trunc = claim[:512]
            evidence_trunc = evidence[:512]
            result = self.nli_pipeline(
                evidence_trunc,
                candidate_labels=[claim_trunc],
                hypothesis_template="{}",
                multi_label=False,
            )
            score = result["scores"][0] if result["scores"] else 0.5
            if score > 0.6:
                return NLIScore(claim=claim, evidence=evidence,
                                entailment=score, contradiction=0,
                                neutral=1 - score)
            elif score < 0.3:
                return NLIScore(claim=claim, evidence=evidence,
                                entailment=0, contradiction=1 - score,
                                neutral=score)
            else:
                return NLIScore(claim=claim, evidence=evidence,
                                entailment=score, contradiction=0,
                                neutral=1 - score)
        except Exception as e:
            if not hasattr(self, '_nli_error_shown'):
                print(f"  [EvidenceScorer] NLI error: {e}")
                self._nli_error_shown = True
            return NLIScore(claim=claim, evidence=evidence, neutral=1.0)

    def _score_heuristic(self, claim: str, evidence: str) -> NLIScore:
        """Simple word-overlap heuristic as fallback."""
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())
        if not claim_words:
            return NLIScore(claim=claim, evidence=evidence, neutral=1.0)

        overlap = len(claim_words & evidence_words) / len(claim_words)
        if overlap > 0.5:
            return NLIScore(claim=claim, evidence=evidence,
                            entailment=overlap, neutral=1 - overlap)
        return NLIScore(claim=claim, evidence=evidence, neutral=1.0)

    def score_all(self, claims: List[Claim], evidence_map: Dict[str, List[Evidence]]) -> List[NLIScore]:
        all_scores = []
        for claim in claims:
            for ev in evidence_map.get(claim.text, []):
                if ev.snippet and ev.snippet != "[No search API configured]":
                    score = self.score(claim.text, ev.snippet)
                    all_scores.append(score)
        return all_scores


@dataclass
class ToolObservation:
    """One tool's analysis result (FactAgent-style)."""
    tool_name: str
    fake_score: float     # 0 = looks real, 1 = looks fake
    reasoning: str = ""


# ---------------------------------------------------------------------------
# FactAgent-style analysis tools (no LLM needed — rule / pattern based)
# Paper ref: Phrase_tool, Language_tool, Commonsense_tool, Search_tool
# ---------------------------------------------------------------------------

SENSATIONAL_PHRASES = [
    "breaking", "shocking", "you won't believe", "exposed", "bombshell",
    "urgent", "alert", "insider", "secret", "conspiracy", "cover-up",
    "outrage", "scandal", "exclusive", "leaked", "hoax", "busted",
    "must see", "goes viral", "mainstream media won't", "they don't want you",
    "what they're hiding", "wake up", "destroyed", "slammed", "blasts",
]


class PhraseAnalyzer:
    """FactAgent Phrase_tool: detect sensational / clickbait language."""

    def analyze(self, text: str) -> ToolObservation:
        lower = text.lower()
        words = lower.split()
        n_words = max(len(words), 1)

        hits = sum(1 for p in SENSATIONAL_PHRASES if p in lower)
        n_exclaim = text.count("!")
        n_question = text.count("?")
        n_allcaps = sum(1 for w in text.split() if w.isupper() and len(w) > 2)

        score = min(1.0, (hits * 0.15 + n_exclaim * 0.1 +
                          n_allcaps / n_words * 2.0 + n_question * 0.03))
        parts = []
        if hits:
            parts.append(f"{hits} sensational phrases")
        if n_exclaim:
            parts.append(f"{n_exclaim} exclamation marks")
        if n_allcaps:
            parts.append(f"{n_allcaps} ALL-CAPS words")
        reason = ", ".join(parts) if parts else "no sensational language detected"
        return ToolObservation("phrase", score, reason)


class LanguageAnalyzer:
    """FactAgent Language_tool: detect grammar / style anomalies."""

    _TYPO_PATTERNS = [
        r"\b(teh|adn|taht|recieve|definately|seperate|goverment|occured)\b",
        r"[.!?]{3,}",                         # excessive punctuation
        r'""',                                 # doubled quotes
        r"\b[A-Z]{5,}\b",                      # very long all-caps
        r"(?<!\d),(?!\s)",                     # comma not followed by space
    ]

    def __init__(self):
        self._compiled = [re.compile(p) for p in self._TYPO_PATTERNS]

    def analyze(self, text: str) -> ToolObservation:
        issues = 0
        details = []
        for pat in self._compiled:
            found = pat.findall(text)
            if found:
                issues += len(found)
                details.append(f"{pat.pattern}×{len(found)}")

        sents = re.split(r"[.!?]+", text)
        sents = [s.strip() for s in sents if len(s.strip()) > 3]
        if sents:
            lengths = [len(s.split()) for s in sents]
            avg_len = sum(lengths) / len(lengths)
            if avg_len < 6:
                issues += 1
                details.append("very short sentences")

        score = min(1.0, issues * 0.12)
        reason = ", ".join(details) if details else "no language anomalies"
        return ToolObservation("language", score, reason)


class CommonsenseChecker:
    """FactAgent Commonsense_tool: use NLI to check plausibility against commonsense templates."""

    COMMONSENSE_TEMPLATES = [
        "This is a well-known fact.",
        "This is true.",
        "This is common knowledge.",
    ]

    def __init__(self, nli_model=None, nli_tokenizer=None, nli_device=None):
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.nli_device = nli_device
        self._cne: Optional[Tuple[int, int, int]] = None
        if nli_model is not None and hasattr(nli_model, "config"):
            cfg = nli_model.config
            if getattr(cfg, "id2label", None):
                self._cne = _mnli_label_indices(dict(cfg.id2label))

    def analyze(self, text: str) -> ToolObservation:
        if self.nli_model is None or self.nli_tokenizer is None:
            return ToolObservation("commonsense", 0.5, "no NLI model")

        claim = text[:256]
        scores = []
        for template in self.COMMONSENSE_TEMPLATES:
            try:
                tok = self.nli_tokenizer(
                    claim, template, truncation=True, max_length=256,
                    return_tensors="pt",
                )
                dev = self.nli_device or torch.device("cpu")
                tok = {k: v.to(dev) for k, v in tok.items()}
                with torch.no_grad():
                    logits = self.nli_model(**tok).logits
                    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
                if self._cne and len(probs) >= 3:
                    ci, ni, ei = self._cne
                    scores.append(probs[ei])
            except Exception:
                pass

        if not scores:
            return ToolObservation("commonsense", 0.5, "NLI failed")

        avg_entail = sum(scores) / len(scores)
        fake_score = 1.0 - avg_entail
        return ToolObservation(
            "commonsense", fake_score,
            f"avg plausibility={avg_entail:.2f} (low → more suspicious)",
        )


# ---------------------------------------------------------------------------
# URL_tool — domain credibility (FactAgent Section 3)
# ---------------------------------------------------------------------------

_CREDIBLE_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "washingtonpost.com", "theguardian.com", "npr.org", "pbs.org",
    "abcnews.go.com", "cbsnews.com", "nbcnews.com", "cnn.com",
    "usatoday.com", "wsj.com", "economist.com", "nature.com",
    "sciencemag.org", "politifact.com", "snopes.com", "factcheck.org",
    "time.com", "theatlantic.com", "ft.com", "bloomberg.com",
    "aljazeera.com", "dw.com", "france24.com",
}

_SUSPICIOUS_DOMAINS = {
    "infowars.com", "naturalnews.com", "beforeitsnews.com",
    "worldnewsdailyreport.com", "yournewswire.com", "newspunch.com",
    "thegatewaypundit.com", "dailybuzzlive.com", "empirenews.net",
    "huzlers.com", "nationalreport.net", "theracketreport.com",
    "abcnews.com.co", "cnn-trending.com",
}


class URLCredibilityChecker:
    """FactAgent URL_tool: assess domain credibility from article text or metadata."""

    def __init__(self, extra_credible=None, extra_suspicious=None):
        self.credible = _CREDIBLE_DOMAINS | set(extra_credible or [])
        self.suspicious = _SUSPICIOUS_DOMAINS | set(extra_suspicious or [])

    @staticmethod
    def _extract_domains(text: str) -> List[str]:
        return re.findall(
            r"(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})+)", text
        )

    def analyze(self, text: str) -> ToolObservation:
        domains = self._extract_domains(text)
        if not domains:
            return ToolObservation("url", 0.5, "no URLs found in text")

        credible_hits = [d for d in domains if d in self.credible]
        suspicious_hits = [d for d in domains if d in self.suspicious]

        if suspicious_hits:
            score = min(1.0, 0.6 + 0.1 * len(suspicious_hits))
            return ToolObservation(
                "url", score,
                f"suspicious domains: {', '.join(suspicious_hits[:3])}",
            )
        if credible_hits:
            score = max(0.0, 0.2 - 0.05 * len(credible_hits))
            return ToolObservation(
                "url", score,
                f"credible domains: {', '.join(credible_hits[:3])}",
            )
        return ToolObservation("url", 0.45, f"unknown domains: {', '.join(domains[:3])}")


# ---------------------------------------------------------------------------
# Standing_tool — political bias / stance detector (FactAgent Section 3)
# ---------------------------------------------------------------------------

_POLITICAL_KEYWORDS = [
    "democrat", "republican", "liberal", "conservative", "trump", "biden",
    "congress", "senate", "election", "vote", "campaign", "gop", "dnc",
    "left-wing", "right-wing", "socialism", "fascism", "impeach",
    "political", "partisan", "legislation", "governor", "president",
    "white house", "capitol", "primary", "ballot", "amendment",
]

_BIAS_INDICATORS = [
    "radical", "extremist", "corrupt", "crooked", "evil", "traitor",
    "patriot", "real americans", "deep state", "rigged", "stolen",
    "witch hunt", "fake news", "enemy of the people", "shill",
    "puppet", "regime", "tyranny", "freedom-loving", "woke",
    "libtard", "snowflake", "maga",
]


class StandingAnalyzer:
    """FactAgent Standing_tool: detect political bias / one-sided framing."""

    def analyze(self, text: str) -> ToolObservation:
        lower = text.lower()

        pol_hits = sum(1 for kw in _POLITICAL_KEYWORDS if kw in lower)
        is_political = pol_hits >= 2

        if not is_political:
            return ToolObservation("standing", 0.0, "not political — skipped")

        bias_hits = sum(1 for b in _BIAS_INDICATORS if b in lower)
        score = min(1.0, bias_hits * 0.15)
        if bias_hits == 0:
            reason = "political but balanced tone"
        else:
            matches = [b for b in _BIAS_INDICATORS if b in lower]
            reason = f"bias indicators ({', '.join(matches[:4])})"
        return ToolObservation("standing", score, reason)


class JudgeAgent:
    """FactAgent-style multi-signal judge with 6 tools + LLM/rule aggregation."""

    DEFAULT_WEIGHTS = {
        "phrase": 0.15,
        "language": 0.08,
        "commonsense": 0.15,
        "search_nli": 0.40,
        "url": 0.12,
        "standing": 0.10,
    }
    DEFAULT_THRESHOLD = 0.50

    def __init__(self, mode: str = "rule", llm_client=None, model: str = "gpt-4o-mini",
                 phrase_analyzer: Optional["PhraseAnalyzer"] = None,
                 language_analyzer: Optional["LanguageAnalyzer"] = None,
                 commonsense_checker: Optional["CommonsenseChecker"] = None,
                 url_checker: Optional["URLCredibilityChecker"] = None,
                 standing_analyzer: Optional["StandingAnalyzer"] = None,
                 tool_weights: Optional[Dict[str, float]] = None,
                 fake_threshold: Optional[float] = None):
        self.mode = mode
        self.llm_client = llm_client
        self.model = model
        self.phrase_analyzer = phrase_analyzer or PhraseAnalyzer()
        self.language_analyzer = language_analyzer or LanguageAnalyzer()
        self.commonsense_checker = commonsense_checker
        self.url_checker = url_checker or URLCredibilityChecker()
        self.standing_analyzer = standing_analyzer or StandingAnalyzer()
        self.tool_weights = tool_weights or dict(self.DEFAULT_WEIGHTS)
        self.fake_threshold = fake_threshold if fake_threshold is not None else self.DEFAULT_THRESHOLD

    # ------------------------------------------------------------------
    # Core: collect observations from all tools
    # ------------------------------------------------------------------
    def _collect_observations(self, article, claims, evidence_map, nli_scores) -> List[ToolObservation]:
        observations: List[ToolObservation] = []

        observations.append(self.phrase_analyzer.analyze(article))
        observations.append(self.language_analyzer.analyze(article))

        if self.commonsense_checker is not None:
            title = article.split(".")[0] if "." in article else article[:200]
            observations.append(self.commonsense_checker.analyze(title))
        else:
            observations.append(ToolObservation("commonsense", 0.5, "skipped"))

        observations.append(self.url_checker.analyze(article))
        observations.append(self.standing_analyzer.analyze(article))

        if nli_scores:
            total_e = sum(s.entailment for s in nli_scores)
            total_c = sum(s.contradiction for s in nli_scores)
            total_n = sum(s.neutral for s in nli_scores)
            total = total_e + total_c + total_n
            if total > 0:
                e_ratio = total_e / total
                c_ratio = total_c / total
                search_fake = max(0.0, min(1.0, 0.5 + (c_ratio - e_ratio) * 0.5))
                reason = f"entail={e_ratio:.2f} contra={c_ratio:.2f}"
            else:
                search_fake, reason = 0.5, "no scores"
            observations.append(ToolObservation("search_nli", search_fake, reason))
        else:
            observations.append(ToolObservation("search_nli", 0.5, "no evidence"))

        return observations

    def _aggregate_score(self, observations: List[ToolObservation]) -> Tuple[float, str]:
        weighted_fake = 0.0
        total_weight = 0.0
        parts = []
        for obs in observations:
            if obs.tool_name == "standing" and obs.fake_score == 0.0:
                continue
            w = self.tool_weights.get(obs.tool_name, 0.05)
            weighted_fake += obs.fake_score * w
            total_weight += w
            parts.append(f"[{obs.tool_name}] fake={obs.fake_score:.2f} ({obs.reasoning})")

        avg_fake = weighted_fake / total_weight if total_weight > 0 else 0.5
        reasoning = f"FactAgent score={avg_fake:.3f} (threshold={self.fake_threshold}) | " + " | ".join(parts)
        return avg_fake, reasoning

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def judge(self, article, claims, evidence_map, nli_scores) -> VerificationResult:
        if self.mode == "llm" and self.llm_client is not None:
            return self._judge_llm_factagent(article, claims, evidence_map, nli_scores)
        return self._judge_rule_factagent(article, claims, evidence_map, nli_scores)

    def _judge_rule_factagent(self, article, claims, evidence_map, nli_scores) -> VerificationResult:
        observations = self._collect_observations(article, claims, evidence_map, nli_scores)
        avg_fake, reasoning = self._aggregate_score(observations)
        prediction = 0 if avg_fake > self.fake_threshold else 1
        confidence = abs(avg_fake - 0.5) * 2
        return VerificationResult(
            article_text=article, claims=claims,
            evidence_map=evidence_map, nli_scores=nli_scores,
            prediction=prediction, confidence=confidence,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # LLM Judge — FactAgent-style checklist prompt (GPT-3.5/4)
    # ------------------------------------------------------------------
    _LLM_JUDGE_PROMPT = """\
You are a professional fact-checker. Analyze the following news article using the checklist below.

## Article (first 800 chars)
{article}

## Tool Observations
{observations}

## Checklist — answer YES/NO for each:
1. Does the article contain sensational or clickbait language?
2. Does the article have grammar errors or formatting anomalies?
3. Does the article contradict common sense or well-known facts?
4. Does the web evidence contradict the article's claims?
5. Does the article originate from a non-credible domain?
6. Does the article show strong political bias or one-sided framing?

Based on your overall assessment across ALL checklist items, determine if the article is REAL or FAKE.
Respond with JSON only: {{"prediction": 0, "confidence": 0.85, "reasoning": "..."}}
Where 0 = FAKE, 1 = REAL.
"""

    def _judge_llm_factagent(self, article, claims, evidence_map, nli_scores) -> VerificationResult:
        observations = self._collect_observations(article, claims, evidence_map, nli_scores)
        obs_text = "\n".join(
            f"- {obs.tool_name}: score={obs.fake_score:.2f}, {obs.reasoning}"
            for obs in observations
        )

        nli_detail = ""
        if nli_scores:
            for s in nli_scores[:8]:
                nli_detail += (
                    f"\n  Claim: {s.claim[:120]}"
                    f"\n    Evidence: {s.evidence[:120]}"
                    f"\n    entail={s.entailment:.2f} contra={s.contradiction:.2f} neutral={s.neutral:.2f}"
                )
            obs_text += "\n- search_nli detail:" + nli_detail

        prompt = self._LLM_JUDGE_PROMPT.format(
            article=article[:800],
            observations=obs_text,
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return VerificationResult(
                    article_text=article, claims=claims,
                    evidence_map=evidence_map, nli_scores=nli_scores,
                    prediction=int(result.get("prediction", 0)),
                    confidence=float(result.get("confidence", 0.5)),
                    reasoning=result.get("reasoning", content[:300]),
                )
        except Exception as e:
            print(f"  [JudgeAgent] LLM error: {e}, falling back to rule-based")

        return self._judge_rule_factagent(article, claims, evidence_map, nli_scores)

    # ------------------------------------------------------------------
    # Threshold / weight tuning on a validation set
    # ------------------------------------------------------------------
    @staticmethod
    def grid_search(
        articles: List[str],
        labels,
        pipeline: Any,
        thresholds=None,
        weight_sets=None,
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Find best threshold & weights on a small validation set. Returns (best_threshold, best_weights, best_metrics)."""
        import numpy as np
        from experiments.metrics import compute_metrics

        if thresholds is None:
            thresholds = [round(0.40 + i * 0.02, 2) for i in range(11)]
        if weight_sets is None:
            weight_sets = [
                {"phrase": 0.15, "language": 0.08, "commonsense": 0.15,
                 "search_nli": 0.40, "url": 0.12, "standing": 0.10},
                {"phrase": 0.20, "language": 0.05, "commonsense": 0.10,
                 "search_nli": 0.50, "url": 0.10, "standing": 0.05},
                {"phrase": 0.10, "language": 0.10, "commonsense": 0.20,
                 "search_nli": 0.35, "url": 0.15, "standing": 0.10},
                {"phrase": 0.18, "language": 0.07, "commonsense": 0.15,
                 "search_nli": 0.45, "url": 0.10, "standing": 0.05},
            ]

        print(f"  [GridSearch] {len(articles)} val samples × {len(thresholds)} thresholds × {len(weight_sets)} weight sets")

        cache = []
        for i, text in enumerate(articles):
            result = pipeline.verify(text)
            cache.append(result)
            if (i + 1) % 10 == 0:
                print(f"  [GridSearch] cached {i+1}/{len(articles)}")

        best_f1, best_t, best_w = -1.0, 0.5, weight_sets[0]

        judge: JudgeAgent = pipeline.judge
        for ws in weight_sets:
            for t in thresholds:
                preds = []
                for r in cache:
                    obs = judge._collect_observations(
                        r.article_text, r.claims, r.evidence_map, r.nli_scores
                    )
                    old_w, old_t = judge.tool_weights, judge.fake_threshold
                    judge.tool_weights, judge.fake_threshold = ws, t
                    avg_fake, _ = judge._aggregate_score(obs)
                    judge.tool_weights, judge.fake_threshold = old_w, old_t
                    preds.append(0 if avg_fake > t else 1)

                m = compute_metrics(
                    np.array(labels), np.array(preds),
                    np.array([0.5] * len(preds)),
                    experiment_name="gs", group="gs",
                )
                if m.f1 > best_f1:
                    best_f1 = m.f1
                    best_t = t
                    best_w = ws
                    print(f"    new best: t={t} F1={m.f1:.4f} Acc={m.accuracy:.4f} "
                          f"P={m.precision:.4f} R={m.recall:.4f}")

        print(f"  [GridSearch] BEST → threshold={best_t}, F1={best_f1:.4f}, weights={best_w}")
        return best_t, best_w, {"f1": best_f1}
