"""
Commonsense-based fake news scoring.

Provides a feature extraction layer that combines KG verification
scores with model predictions for ensemble classification.
"""

import numpy as np
from typing import List, Dict, Optional
from experiments.group4_kg_rl.kg_builder import (
    TripleExtractor, CommonsenseChecker, Triple,
)


class CommonsenseFeatureExtractor:
    """
    Extract commonsense-based features from news articles
    that can be combined with model predictions.
    """

    def __init__(self, kg_sources: Optional[List[str]] = None, use_llm: bool = False,
                 llm_client=None):
        self.extractor = TripleExtractor(
            method="llm" if use_llm else "heuristic",
            llm_client=llm_client,
        )
        self.checker = CommonsenseChecker(sources=kg_sources or ["conceptnet"])

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract commonsense features from a single article.

        Returns dict with:
            - violation_score: 0-1 (higher = more violations)
            - verified_ratio: fraction of triples verified
            - num_triples: number of extracted triples
            - avg_kg_score: average KG verification score
        """
        triples = self.extractor.extract(text)

        if not triples:
            return {
                "violation_score": 0.5,
                "verified_ratio": 0.0,
                "num_triples": 0,
                "avg_kg_score": 0.0,
            }

        result = self.checker.check(triples)

        return {
            "violation_score": result["violation_score"],
            "verified_ratio": result["verified_ratio"],
            "num_triples": len(triples),
            "avg_kg_score": 1.0 - result["violation_score"],
        }

    def extract_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        features = []
        for i, text in enumerate(texts):
            feat = self.extract_features(text)
            features.append(feat)
            if (i + 1) % 10 == 0:
                print(f"  [KG Features] {i+1}/{len(texts)} processed")
        return features
