"""
Keyword Filter Adapter
=======================
A naive keyword-matching safety layer.  Useful as a simple baseline to
show how adapters work and to illustrate the gap between keyword matching
and real safety systems.

Usage:
    python benchmark_runner.py --adapter adapters.keyword_filter \
        --gauntlet gauntlets/gauntlet_v3.txt
"""

from typing import Any, Dict
from safety_layer import SafetyLayer, SafetyResult

# Very rough keyword list — intentionally simplistic
DEFAULT_KEYWORDS = [
    "how to make a bomb",
    "how to hack",
    "how to kill",
    "synthesize",
    "manufacture drugs",
    "build a weapon",
    "exploit",
    "malware",
    "ransomware",
    "phishing",
    "steal identity",
    "bypass security",
]


class KeywordFilterLayer(SafetyLayer):
    """Blocks prompts that contain any keyword from a configurable list."""

    def setup(self, config: Dict[str, Any]) -> None:
        self.keywords = config.get("keywords", DEFAULT_KEYWORDS)

    def evaluate(self, prompt: str) -> SafetyResult:
        prompt_lower = prompt.lower()
        matched = [kw for kw in self.keywords if kw in prompt_lower]

        if matched:
            return SafetyResult(
                decision="block",
                confidence=min(1.0, len(matched) * 0.3),
                metadata={"matched_keywords": matched},
            )

        return SafetyResult(decision="allow", confidence=0.8)

    @property
    def name(self) -> str:
        return "Keyword Filter"
