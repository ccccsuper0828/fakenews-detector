"""
Multi-agent orchestration pipeline.

Chains together ClaimExtractor -> WebSearcher -> EvidenceScorer -> JudgeAgent
to produce a verification result for each article.
"""

import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiments.group3_multiagent.agents import (
    ClaimExtractor, WebSearcher, EvidenceScorer, JudgeAgent,
    CommonsenseChecker, PhraseAnalyzer, LanguageAnalyzer,
    URLCredibilityChecker, StandingAnalyzer,
    VerificationResult,
)
from experiments.group3_multiagent.search_tools import create_search_fn


class VerificationPipeline:
    """
    End-to-end multi-agent verification pipeline.

    Usage:
        pipeline = VerificationPipeline.from_config(cfg.multiagent)
        result = pipeline.verify("Some news article text...")
    """

    def __init__(
        self,
        claim_extractor: ClaimExtractor,
        web_searcher: WebSearcher,
        evidence_scorer: EvidenceScorer,
        judge: JudgeAgent,
    ):
        self.claim_extractor = claim_extractor
        self.web_searcher = web_searcher
        self.evidence_scorer = evidence_scorer
        self.judge = judge

    @classmethod
    def from_config(cls, ma_cfg) -> "VerificationPipeline":
        """Build pipeline from MultiAgentConfig."""
        import torch
        llm_client = None
        llm_model = ma_cfg.llm_model

        # --- LLM setup: try OpenAI → fall back to local model ---
        api_key = os.environ.get(ma_cfg.api_key_env, "")
        if api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                for model_name in [ma_cfg.llm_model, "gpt-4o-mini", "gpt-3.5-turbo"]:
                    try:
                        client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": "test"}],
                            max_tokens=1,
                        )
                        llm_client = client
                        llm_model = model_name
                        print(f"  [Pipeline] OpenAI connected → {model_name}")
                        break
                    except Exception:
                        continue
            except ImportError:
                pass

        if llm_client is None:
            local_model = os.environ.get("LOCAL_LLM_MODEL", "none")
            if local_model.lower() in ("none", "skip", ""):
                print("  [Pipeline] No LLM → using heuristic claim extraction + rule judge (saves ~1GB RAM)")
            else:
                print(f"  [Pipeline] OpenAI unavailable, loading local LLM '{local_model}'...")
                try:
                    from experiments.group3_multiagent.agents import LocalLLMClient
                    llm_client = LocalLLMClient(model_name=local_model)
                    llm_model = local_model
                except Exception as e:
                    print(f"  [Pipeline] Local LLM failed ({e}), using heuristic mode")

        # --- Search setup ---
        search_key = os.environ.get(ma_cfg.search_api_key_env, "")
        search_fn = create_search_fn(ma_cfg.search_provider, search_key)

        # --- NLI: 3-way MNLI (premise=evidence, hypothesis=claim), not zero-shot misuse ---
        nli_model = None
        nli_tokenizer = None
        nli_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            device_label = "GPU" if nli_device.type == "cuda" else "CPU"
            print(f"  [Pipeline] Loading MNLI '{ma_cfg.nli_model}' on {device_label}...")
            nli_tokenizer = AutoTokenizer.from_pretrained(ma_cfg.nli_model)
            nli_model = AutoModelForSequenceClassification.from_pretrained(ma_cfg.nli_model)
            nli_model.eval()
            nli_model.to(nli_device)
            print(f"  [Pipeline] MNLI loaded on {device_label}")
        except Exception as e:
            print(f"  [Pipeline] Could not load MNLI model: {e}")

        cs_checker = None
        if nli_model is not None and nli_tokenizer is not None:
            cs_checker = CommonsenseChecker(
                nli_model=nli_model,
                nli_tokenizer=nli_tokenizer,
                nli_device=nli_device,
            )

        return cls(
            claim_extractor=ClaimExtractor(llm_client, llm_model),
            web_searcher=WebSearcher(search_fn, top_k=ma_cfg.search_top_k),
            evidence_scorer=EvidenceScorer(
                nli_model=nli_model,
                nli_tokenizer=nli_tokenizer,
                nli_device=nli_device,
            ),
            judge=JudgeAgent(
                mode=ma_cfg.judge_mode,
                llm_client=llm_client,
                model=llm_model,
                phrase_analyzer=PhraseAnalyzer(),
                language_analyzer=LanguageAnalyzer(),
                commonsense_checker=cs_checker,
                url_checker=URLCredibilityChecker(),
                standing_analyzer=StandingAnalyzer(),
            ),
        )

    def verify(self, article: str) -> VerificationResult:
        """Run the full verification pipeline on a single article."""
        claims = self.claim_extractor.extract(article)
        evidence_map = self.web_searcher.search_all(claims)
        nli_scores = self.evidence_scorer.score_all(claims, evidence_map)
        result = self.judge.judge(article, claims, evidence_map, nli_scores)
        return result

    def verify_batch(self, articles: List[str]) -> List[VerificationResult]:
        """Verify a batch of articles."""
        return [self.verify(article) for article in articles]
