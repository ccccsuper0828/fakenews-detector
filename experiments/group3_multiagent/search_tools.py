"""
Web search API wrappers for the multi-agent pipeline.

Supports Tavily, SerpAPI, and a mock search for testing.
"""

import os
import json
from typing import List, Dict, Optional


def create_tavily_search(api_key: Optional[str] = None):
    """Create a Tavily search function."""
    key = api_key or os.environ.get("TAVILY_API_KEY", "")
    if not key:
        print("  [SearchTools] No TAVILY_API_KEY found, using mock search")
        return None

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=key)

        def search_fn(query: str, max_results: int = 5) -> List[Dict]:
            response = client.search(query, max_results=max_results)
            return [
                {
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "score": r.get("score", 0.5),
                }
                for r in response.get("results", [])
            ]

        return search_fn
    except ImportError:
        print("  [SearchTools] tavily package not installed")
        return None


def create_serpapi_search(api_key: Optional[str] = None):
    """Create a SerpAPI search function."""
    key = api_key or os.environ.get("SERPAPI_API_KEY", "")
    if not key:
        return None

    try:
        from serpapi import GoogleSearch

        def search_fn(query: str, max_results: int = 5) -> List[Dict]:
            params = {
                "q": query,
                "api_key": key,
                "num": max_results,
            }
            results = GoogleSearch(params).get_dict()
            organic = results.get("organic_results", [])
            return [
                {
                    "content": r.get("snippet", ""),
                    "url": r.get("link", ""),
                    "score": 0.5,
                }
                for r in organic[:max_results]
            ]

        return search_fn
    except ImportError:
        print("  [SearchTools] serpapi package not installed")
        return None


def create_duckduckgo_search():
    """Create a DuckDuckGo search function (free, no API key needed)."""
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*duckduckgo_search.*renamed.*")
        from duckduckgo_search import DDGS

        def search_fn(query: str, max_results: int = 5) -> List[Dict]:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))
                return [
                    {
                        "content": r.get("body", ""),
                        "url": r.get("href", ""),
                        "score": 0.5,
                    }
                    for r in results
                ]
            except Exception as e:
                return []

        print("  [SearchTools] Using DuckDuckGo (free, no API key)")
        return search_fn
    except ImportError:
        return None


def create_mock_search():
    """Mock search function for testing without API keys."""

    def search_fn(query: str, max_results: int = 5) -> List[Dict]:
        return [
            {
                "content": f"Mock evidence for: {query[:100]}",
                "url": "https://example.com/mock",
                "score": 0.5,
            }
        ]

    return search_fn


def create_search_fn(provider: str = "tavily", api_key: Optional[str] = None):
    """Factory function to create the appropriate search function."""
    if provider == "tavily":
        fn = create_tavily_search(api_key)
        if fn:
            return fn
    elif provider == "serpapi":
        fn = create_serpapi_search(api_key)
        if fn:
            return fn

    ddg = create_duckduckgo_search()
    if ddg:
        return ddg

    print("  [SearchTools] Falling back to mock search")
    return create_mock_search()
