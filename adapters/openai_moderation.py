"""
OpenAI Moderation API Adapter
===============================
Uses the OpenAI Moderation endpoint for safety classification.

Requirements:
    pip install openai
    export OPENAI_API_KEY=your_key

Usage:
    python benchmark_runner.py --adapter adapters.openai_moderation \
        --gauntlet gauntlets/gauntlet_v3.txt
"""

import os
import time
from typing import Any, Dict
from safety_layer import SafetyLayer, SafetyResult


class OpenAIModerationLayer(SafetyLayer):
    """
    Wraps the OpenAI Moderation API.

    Respects rate limits with configurable delay between calls.
    Expects OPENAI_API_KEY environment variable.
    """

    def setup(self, config: Dict[str, Any]) -> None:
        from openai import OpenAI

        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY env var or pass api_key in --layer-config"
            )
        self.client = OpenAI(api_key=api_key)
        self.model = config.get("model", "omni-moderation-latest")
        self.delay = config.get("delay_seconds", 0.1)

    def evaluate(self, prompt: str) -> SafetyResult:
        response = self.client.moderations.create(
            model=self.model,
            input=prompt,
        )
        result = response.results[0]

        # Find highest category score
        scores = {k: v for k, v in result.category_scores.__dict__.items()}
        max_category = max(scores, key=scores.get)
        max_score = scores[max_category]

        flagged_categories = [
            k for k, v in result.categories.__dict__.items() if v
        ]

        time.sleep(self.delay)  # Rate limit courtesy

        return SafetyResult(
            decision="block" if result.flagged else "allow",
            confidence=max_score if result.flagged else 1.0 - max_score,
            metadata={
                "flagged": result.flagged,
                "top_category": max_category,
                "top_score": max_score,
                "flagged_categories": flagged_categories,
            },
        )

    @property
    def name(self) -> str:
        return "OpenAI Moderation"
