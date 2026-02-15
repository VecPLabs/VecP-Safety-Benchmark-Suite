"""
HuggingFace Safety Classifier Adapter
=======================================
Generic adapter for any HuggingFace text-classification model trained
for safety/toxicity detection.

Works out of the box with models like:
  - unitary/toxic-bert
  - s-nlp/roberta_toxicity_classifier
  - Any model with a binary safe/unsafe or toxic/non-toxic output

Requirements:
    pip install transformers torch

Usage:
    python benchmark_runner.py --adapter adapters.hf_classifier \
        --gauntlet gauntlets/gauntlet_v3.txt \
        --layer-config '{"model": "s-nlp/roberta_toxicity_classifier", "toxic_label": "toxic"}'
"""

from typing import Any, Dict
from safety_layer import SafetyLayer, SafetyResult


class HFClassifierLayer(SafetyLayer):
    """
    Wraps any HuggingFace text-classification pipeline for safety detection.

    Config options (via --layer-config):
        model:       HuggingFace model ID (required)
        toxic_label: The label string that means "unsafe" (default: "toxic")
        threshold:   Confidence threshold for blocking (default: 0.5)
        device:      "cuda" or "cpu" (default: cuda)
    """

    def setup(self, config: Dict[str, Any]) -> None:
        from transformers import pipeline

        model_id = config.get("model", "s-nlp/roberta_toxicity_classifier")
        device = config.get("device", "cuda")
        device_idx = 0 if device == "cuda" else -1

        print(f"  Loading HF classifier: {model_id}")
        self.pipe = pipeline(
            "text-classification",
            model=model_id,
            device=device_idx,
            truncation=True,
            max_length=512,
        )
        self.toxic_label = config.get("toxic_label", "toxic").lower()
        self.threshold = config.get("threshold", 0.5)

    def evaluate(self, prompt: str) -> SafetyResult:
        results = self.pipe(prompt)
        top = results[0]

        label = top["label"].lower()
        score = top["score"]

        is_toxic = (label == self.toxic_label and score >= self.threshold)

        return SafetyResult(
            decision="block" if is_toxic else "allow",
            confidence=score,
            metadata={"label": top["label"], "score": score},
        )

    @property
    def name(self) -> str:
        return "HF Classifier"
