"""
multi_runner.py
===============
Orchestrates running multiple adapters sequentially against a gauntlet
and emits structured events into a queue for the SSE stream.

Event types emitted:
  {"type": "progress", "method": NAME, "adapter_index": i, "adapter_total": N,
   "index": prompt_current, "total": prompt_total, "pct": 0-100}
  {"type": "result",   "method": NAME, "metrics": {...}}
  {"type": "error",    "method": NAME, "message": "..."}
  {"type": "done"}
"""

import importlib
import inspect
import pathlib
import queue
from typing import List, Optional

from benchmark_runner import load_adapter, load_gauntlet, run_benchmark
from safety_layer import SafetyLayer

# Maps the short UI names sent from the browser to dotted import paths
ADAPTER_MAP = {
    "BASELINE":      "adapters.baseline",
    "KEYWORD":       "adapters.keyword_filter",
    "LLAMAGUARD":    "adapters.llamaguard",
    "OPENAI-MOD":    "adapters.openai_moderation",
    "HF-CLASSIFIER": "adapters.hf_classifier",
}


def _discover_adapters() -> dict:
    """
    Scan the adapters/ directory for .py files that contain a SafetyLayer
    subclass but are NOT already in ADAPTER_MAP's values.
    Returns {stem: dotpath}, e.g. {"my_system": "adapters.my_system"}.
    """
    found = {}
    known_dotpaths = set(ADAPTER_MAP.values())
    adapter_dir = pathlib.Path(__file__).parent / "adapters"
    for path in sorted(adapter_dir.glob("*.py")):
        stem = path.stem
        if stem.startswith("_"):
            continue
        dotpath = f"adapters.{stem}"
        if dotpath in known_dotpaths:
            continue
        try:
            mod = importlib.import_module(dotpath)
            has_layer = any(
                issubclass(obj, SafetyLayer) and obj is not SafetyLayer
                for _, obj in inspect.getmembers(mod, inspect.isclass)
            )
            if has_layer:
                found[stem] = dotpath
        except Exception:
            pass  # Skip files that fail to import (missing deps, syntax errors, etc.)
    return found


def run_multi(
    adapter_names: List[str],
    gauntlet_path: str,
    max_prompts: Optional[int],
    target_model: str,
    event_queue: queue.Queue,
) -> None:
    """
    Runs in a background thread. Loads the gauntlet once, iterates over
    each requested adapter, and puts event dicts into event_queue.
    Always emits {"type": "done"} as the final event.
    """
    # Load gauntlet once up front
    try:
        prompts = load_gauntlet(gauntlet_path)
    except Exception as e:
        event_queue.put({"type": "error", "method": "GAUNTLET", "message": str(e)})
        event_queue.put({"type": "done"})
        return

    total_adapters = len(adapter_names)
    total_prompts = min(len(prompts), max_prompts) if max_prompts else len(prompts)

    combined_map = {**ADAPTER_MAP, **_discover_adapters()}

    for adapter_idx, name in enumerate(adapter_names):
        dotpath = combined_map.get(name)
        if dotpath is None:
            event_queue.put({
                "type": "error",
                "method": name,
                "message": f"Unknown adapter '{name}'. Valid: {list(combined_map.keys())}",
            })
            continue

        # Load and set up adapter
        try:
            layer = load_adapter(dotpath)
            layer.setup({})
        except Exception as e:
            event_queue.put({"type": "error", "method": name, "message": str(e)})
            continue

        # Build per-prompt progress callback
        def make_callback(method_name, a_idx, a_total, p_total):
            def callback(current, total):
                pct = int((current / total) * 100) if total > 0 else 0
                event_queue.put({
                    "type": "progress",
                    "method": method_name,
                    "adapter_index": a_idx + 1,
                    "adapter_total": a_total,
                    "index": current,
                    "total": p_total,
                    "pct": pct,
                })
            return callback

        callback = make_callback(name, adapter_idx, total_adapters, total_prompts)

        # Run benchmark
        import time
        t0 = time.time()
        try:
            data = run_benchmark(
                layer,
                prompts,
                verbose=False,
                max_prompts=max_prompts,
                progress_callback=callback,
            )
        except Exception as e:
            event_queue.put({"type": "error", "method": name, "message": str(e)})
            try:
                layer.teardown()
            except Exception:
                pass
            continue

        elapsed = time.time() - t0

        try:
            layer.teardown()
        except Exception:
            pass

        s = data["summary"]
        event_queue.put({
            "type": "result",
            "method": name,
            "metrics": {
                "time_sec":     round(elapsed, 2),
                "avg_ms":       round(s["avg_time_ms"], 1),
                "recall":       round(s["recall"], 4),
                "precision":    round(s["precision"], 4),
                "f1":           round(s["f1"], 4),
                "fp_rate":      round(s["fp_rate"], 4),
                "accuracy":     round(s["accuracy"], 4),
                "eff_recall":   round(s["effective_recall"], 4),
                "tp": s["tp"], "tn": s["tn"],
                "fp": s["fp"], "fn": s["fn"],
                "total_prompts": s["total_prompts"],
                "target_model":  target_model,
            },
        })

    event_queue.put({"type": "done"})
