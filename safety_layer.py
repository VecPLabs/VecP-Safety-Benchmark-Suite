"""
Safety Layer Abstract Interface
================================
Implement this interface to benchmark your safety system.

Your safety layer receives a prompt and must return a SafetyResult
indicating whether the prompt should be blocked or allowed, along with
an optional confidence score and any custom metadata your system produces.

Example:
    class MyGuardrail(SafetyLayer):
        def setup(self, config):
            self.model = load_my_model(config["model_path"])

        def evaluate(self, prompt: str) -> SafetyResult:
            score = self.model.score(prompt)
            return SafetyResult(
                decision="block" if score > 0.5 else "allow",
                confidence=score,
            )
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SafetyResult:
    """
    Standardized result from a safety layer evaluation.

    Attributes:
        decision:   "block" or "allow" — the safety layer's verdict.
        confidence: Optional float in [0, 1]. Higher = more confident in the
                    decision. Used for threshold sweeps and analysis.
        response:   Optional model-generated text (if your pipeline generates
                    one). Used for refusal detection analysis.
        metadata:   Any extra data your layer produces (scores, gate names,
                    latencies, etc.). Stored in results JSON for analysis.
    """
    decision: str                           # "block" or "allow"
    confidence: float = 0.0                 # 0-1, higher = more confident
    response: str = ""                      # model output (if generated)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.decision not in ("block", "allow"):
            raise ValueError(f"decision must be 'block' or 'allow', got '{self.decision}'")
        self.confidence = max(0.0, min(1.0, self.confidence))


class SafetyLayer(ABC):
    """
    Abstract base class for safety layers.

    Subclass this and implement `evaluate()` to plug your system into the
    benchmark runner.  Optionally override `setup()` for initialization
    and `teardown()` for cleanup.
    """

    def setup(self, config: Dict[str, Any]) -> None:
        """
        Called once before benchmarking begins.

        Use this to load models, weights, tokenizers, etc.

        Args:
            config: Dict of CLI flags and layer-specific settings.
        """
        pass

    @abstractmethod
    def evaluate(self, prompt: str) -> SafetyResult:
        """
        Evaluate a single prompt for safety.

        This is the only method you *must* implement.

        Args:
            prompt: The raw user prompt text.

        Returns:
            A SafetyResult with at minimum a decision of "block" or "allow".
        """
        ...

    def teardown(self) -> None:
        """Called once after benchmarking completes. Clean up resources."""
        pass

    @property
    def name(self) -> str:
        """Display name for reports. Override for a friendlier label."""
        return self.__class__.__name__
