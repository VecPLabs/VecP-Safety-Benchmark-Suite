"""
app.py
======
Flask web server for the VecP Safety Benchmark UI.

Run with:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import json
import queue
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from multi_runner import ADAPTER_MAP, _discover_adapters, run_multi

app = Flask(__name__)

# In-memory run registry: run_id -> {queue, results, status, config}
RUNS: dict = {}

# Valid gauntlet filenames (prevents path traversal)
GAUNTLET_DIR = Path(__file__).parent / "gauntlets"
VALID_GAUNTLETS = ["gauntlet_v3.txt", "gauntlet_v3_enhanced.txt"]

# Prompt volume presets: UI label -> max_prompts (None = all)
VOLUME_MAP = {
    "FULL":     None,
    "500":      500,
    "200":      200,
    "100":      100,
    "50":       50,
    "QUICK-20": 20,
}


@app.route("/")
def index():
    return render_template(
        "index.html",
        adapters=list(ADAPTER_MAP.keys()),
        gauntlets=VALID_GAUNTLETS,
        volume_options=list(VOLUME_MAP.keys()),
    )


@app.route("/adapters")
def list_adapters():
    """Return discovered (non-hardcoded) adapter names for the target model dropdown."""
    return jsonify(sorted(_discover_adapters().keys()))


@app.route("/run", methods=["POST"])
def start_run():
    """
    Start a new multi-adapter benchmark run.

    Expected JSON body:
    {
        "adapters":     ["BASELINE", "KEYWORD"],
        "gauntlet":     "gauntlet_v3.txt",
        "volume":       "QUICK-20",
        "target_model": "claude-sonnet-4-6"
    }

    Returns: {"run_id": "<hex>"}
    """
    body = request.get_json(force=True, silent=True) or {}

    selected = body.get("adapters", [])
    if not isinstance(selected, list):
        selected = []

    # Include target adapter (from dropdown) at the front of the run list
    target_adapter = str(body.get("target_adapter", "")).strip()
    if target_adapter and target_adapter not in selected:
        selected = [target_adapter] + selected

    if not selected:
        return jsonify({"error": "Provide at least one adapter or select a target model."}), 400

    all_adapters = {**ADAPTER_MAP, **_discover_adapters()}
    unknown = [a for a in selected if a not in all_adapters]
    if unknown:
        return jsonify({
            "error": f"Unknown adapters: {unknown}. Valid: {list(all_adapters.keys())}"
        }), 400

    gauntlet_name = body.get("gauntlet", VALID_GAUNTLETS[0])
    if gauntlet_name not in VALID_GAUNTLETS:
        return jsonify({"error": f"Unknown gauntlet '{gauntlet_name}'."}), 400

    volume_key = body.get("volume", "100")
    if volume_key not in VOLUME_MAP:
        return jsonify({"error": f"Unknown volume '{volume_key}'."}), 400

    target_model = target_adapter or str(body.get("target_model", "")).strip() or "unspecified"
    gauntlet_path = str(GAUNTLET_DIR / gauntlet_name)
    max_prompts = VOLUME_MAP[volume_key]

    run_id = uuid.uuid4().hex
    eq: queue.Queue = queue.Queue()
    RUNS[run_id] = {
        "queue":   eq,
        "results": {},
        "status":  "running",
        "config": {
            "adapters":     selected,
            "gauntlet":     gauntlet_name,
            "volume":       volume_key,
            "target_model": target_model,
            "max_prompts":  max_prompts,
        },
    }

    t = threading.Thread(
        target=run_multi,
        args=(selected, gauntlet_path, max_prompts, target_model, eq),
        daemon=True,
    )
    t.start()

    return jsonify({"run_id": run_id})


@app.route("/stream/<run_id>")
def stream(run_id):
    """Server-Sent Events stream for real-time benchmark progress."""
    run = RUNS.get(run_id)
    if run is None:
        err = json.dumps({"type": "error", "method": "", "message": "run not found"})
        return Response(
            f"data: {err}\n\n",
            mimetype="text/event-stream",
            status=404,
        )

    def generate():
        eq = run["queue"]
        while True:
            try:
                event = eq.get(timeout=30)
            except queue.Empty:
                # Keep-alive comment to prevent proxy timeout
                yield ": keep-alive\n\n"
                continue

            if event["type"] == "result":
                run["results"][event["method"]] = event["metrics"]

            yield f"data: {json.dumps(event)}\n\n"

            if event["type"] == "done":
                run["status"] = "complete"
                break

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/results/<run_id>")
def results(run_id):
    """Return all collected results for a run (useful for polling fallback)."""
    run = RUNS.get(run_id)
    if run is None:
        return jsonify({"error": "run not found"}), 404
    return jsonify({
        "status":  run["status"],
        "config":  run["config"],
        "results": run["results"],
    })


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
