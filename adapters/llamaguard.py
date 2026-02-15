"""
LlamaGuard Adapter
===================
Adapter for Meta's LlamaGuard family of safety classifiers.
Supports LlamaGuard, LlamaGuard-2, and LlamaGuard-3.

Requirements:
    pip install transformers torch accelerate

Usage:
    python benchmark_runner.py --adapter adapters.llamaguard \
        --gauntlet gauntlets/gauntlet_v3.txt \
        --layer-config '{"model": "meta-llama/Llama-Guard-3-8B"}'
"""

from typing import Any, Dict
from safety_layer import SafetyLayer, SafetyResult


class LlamaGuardLayer(SafetyLayer):
    """
    Wraps Meta's LlamaGuard models for safety classification.

    Expects --layer-config with:
        model:  HuggingFace model ID (default: meta-llama/Llama-Guard-3-8B)
        device: "cuda" or "cpu" (default: cuda)
    """

    def setup(self, config: Dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = config.get("model", "meta-llama/Llama-Guard-3-8B")
        device = config.get("device", "cuda")

        print(f"  Loading LlamaGuard: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()
        self.device = device

    def evaluate(self, prompt: str) -> SafetyResult:
        import torch

        # LlamaGuard expects a chat format
        chat = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            chat, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()

        # LlamaGuard outputs "safe" or "unsafe\n<category>"
        is_unsafe = response.lower().startswith("unsafe")

        return SafetyResult(
            decision="block" if is_unsafe else "allow",
            confidence=0.9 if is_unsafe else 0.9,
            metadata={
                "raw_output": response,
                "categories": response.split("\n")[1:] if is_unsafe else [],
            },
        )

    def teardown(self) -> None:
        if hasattr(self, "model"):
            del self.model
            import torch
            torch.cuda.empty_cache()

    @property
    def name(self) -> str:
        return "LlamaGuard"
