"""
Baseline Adapter — No Safety Layer
====================================
Always allows every prompt.  Useful as a control / lower bound to measure
how much a safety layer actually improves over a naked model.

Usage:
    python benchmark_runner.py --adapter adapters.baseline --gauntlet gauntlets/gauntlet_v3.txt
"""

from safety_layer import SafetyLayer, SafetyResult


class BaselineLayer(SafetyLayer):
    """Passes every prompt through — no filtering whatsoever."""

    def evaluate(self, prompt: str) -> SafetyResult:
        return SafetyResult(decision="allow", confidence=1.0)

    @property
    def name(self) -> str:
        return "Baseline (No Safety)"
