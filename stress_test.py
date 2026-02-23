"""
stress_test.py — headless stability check for Visual Neural Network.

Runs 40 diverse inputs through the full pipeline (encoder → memory → network →
projector) with no GUI, then does 5 learn steps.  Prints a pass/fail report.

Usage:
    python3 stress_test.py
"""

import os
import sys
import time

# Suppress HuggingFace noise before any imports
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY",        "error")
os.environ.setdefault("HF_HUB_VERBOSITY",              "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM",        "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import DEVICE, MEMORY_THRESHOLD, MEMORY_CAPACITY, VISUAL_LAYER_SIZES

# ── 40 diverse test inputs ────────────────────────────────────────────────────

INPUTS = [
    # Greeting (0)
    "Hello there!", "Hi, good morning!", "Hey!", "Howdy partner",
    # Farewell (1)
    "Goodbye, see you later!", "Bye!", "Take care, farewell", "See you soon",
    # What-Q (2)
    "What is this?", "What does it do?", "What is AI?", "What time is it?",
    # How-Q (3)
    "How are you?", "How does this work?", "How do I start?", "How can I learn?",
    # Help-Req (4)
    "Help me please", "I need help", "Can you assist me?", "I am stuck",
    # Info-Req (5)
    "Tell me about neural networks", "Give me information on this",
    "What is deep learning?", "Explain how transformers work",
    # Confirm (6)
    "Yes, that is correct", "Sure, go ahead", "Alright, I agree", "OK confirmed",
    # Praise (7)
    "Great job!", "That is amazing", "Well done!", "Excellent work",
    # Deny (8)
    "No, that is wrong", "I disagree", "That is incorrect", "Nope, not right",
    # Complaint (9)
    "This is terrible", "I am not happy with this", "This is very bad", "Awful",
    # Command (10)
    "Do it now", "Start the process", "Run the program", "Execute this",
    # Small-Talk (11)
    "Nice weather today", "How about those sports?", "Interesting day",
    "So what do you think?",
]

assert len(INPUTS) >= 40, f"Expected >= 40 inputs, got {len(INPUTS)}"
N_INPUTS = len(INPUTS)

# 5 learn examples (text, class_idx)
LEARN_EXAMPLES = [
    ("Hello friend!", 0),
    ("Goodbye for now", 1),
    ("What is PyTorch?", 2),
    ("Help me with this problem", 4),
    ("Yes, absolutely correct", 6),
]


def vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def _bar(pct: float, width: int = 20) -> str:
    filled = int(pct * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {pct*100:.0f}%"


def main() -> int:
    print("=" * 62)
    print(f"  Visual Neural Network — Stress Test  ({N_INPUTS} inputs + 5 learn)")
    print(f"  Device : {DEVICE}  |  VRAM at start: {vram_mb():.0f} MB")
    print("=" * 62)

    t0 = time.monotonic()

    # ── 1. Load components ────────────────────────────────────────────────────
    print("\n[1/4] Loading SemanticEncoder…", end=" ", flush=True)
    from core.semantic_encoder import SemanticEncoder
    encoder = SemanticEncoder()
    print(f"OK  ({encoder.embedding_dim}-dim on {encoder.device})")

    print("[2/4] Building Trainer…", end=" ", flush=True)
    from core.trainer import Trainer
    trainer = Trainer(encoder)
    loaded = trainer.load()
    if loaded:
        print(f"OK  (saved weights loaded, {len(trainer.loss_history)} prior steps)")
    else:
        print("OK  (fresh — pre-training 300 epochs…)", end=" ", flush=True)
        trainer.pretrain()
        print("done")

    from core.memory     import BrainMemory
    from core.projection import ActivationProjector
    memory    = BrainMemory()
    projector = ActivationProjector()

    # ── 2. Inference loop ─────────────────────────────────────────────────────
    print(f"[3/4] Running {N_INPUTS} inference passes…\n")
    vram_start = vram_mb()
    errors: list[str] = []
    latencies: list[float] = []

    for i, text in enumerate(INPUTS, 1):
        try:
            ts = time.monotonic()

            embedding            = encoder.encode(text)
            bias, similar, sim   = memory.recall(embedding)
            probs, acts, pred, _ = trainer.infer(embedding, memory_bias=bias)
            memory.store(embedding, text, pred, probs)
            visual_acts = projector.project_activations(acts)
            visual_wm   = projector.project_all_weights(trainer.get_weight_magnitudes())

            te = time.monotonic()
            latencies.append((te - ts) * 1000)

            conf       = float(max(probs)) * 100
            recall_tag = f" ← recall {sim:.2f}" if sim >= MEMORY_THRESHOLD else ""
            mem_tag    = f"mem={memory.size():2d}/{MEMORY_CAPACITY}"

            # Sanity checks
            assert len(visual_acts) == len(VISUAL_LAYER_SIZES), "Wrong visual layer count"
            assert 0 <= pred < 12, f"pred={pred} out of range"
            assert abs(sum(probs) - 1.0) < 1e-3, f"probs don't sum to 1: {sum(probs):.4f}"

            print(f"  [{i:02d}/{N_INPUTS}] pred={pred:2d}  conf={conf:5.1f}%  {mem_tag}"
                  f"  {latencies[-1]:5.1f}ms{recall_tag}")

        except Exception as exc:
            errors.append(f"input {i}: {exc}")
            print(f"  [{i:02d}/{N_INPUTS}] ERROR: {exc}")

    vram_after_infer = vram_mb()

    # ── 3. Learn loop ─────────────────────────────────────────────────────────
    print("\n[4/4] Running 5 learn steps…")
    for text, label in LEARN_EXAMPLES:
        try:
            emb  = encoder.encode(text)
            loss = trainer.learn_from_input(emb, label)
            print(f"  learn label={label}  loss={loss:.4f}  '{text[:35]}'  OK")
        except Exception as exc:
            errors.append(f"learn '{text[:20]}': {exc}")
            print(f"  learn ERROR: {exc}")

    vram_end = vram_mb()

    # ── 4. Report ─────────────────────────────────────────────────────────────
    elapsed  = time.monotonic() - t0
    avg_lat  = sum(latencies) / len(latencies) if latencies else 0
    min_lat  = min(latencies, default=0)
    max_lat  = max(latencies, default=0)
    p95_lat  = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    print("\n" + "=" * 62)
    print("  RESULTS")
    print("=" * 62)
    print(f"  Total time         : {elapsed:.1f}s")
    print(f"  Inference latency  : avg={avg_lat:.1f}ms  "
          f"min={min_lat:.1f}  p95={p95_lat:.1f}  max={max_lat:.1f}ms")
    print(f"  VRAM start→infer→end: "
          f"{vram_start:.0f}→{vram_after_infer:.0f}→{vram_end:.0f} MB"
          f"  (delta {vram_end-vram_start:+.0f} MB)")
    print(f"  Memory buffer      : {memory.size()}/{MEMORY_CAPACITY} stored  OK")
    print(f"  Errors             : {len(errors)}")
    if errors:
        for e in errors:
            print(f"    ✗ {e}")

    status = "PASS ✓" if not errors else f"FAIL ✗  ({len(errors)} error(s))"
    print(f"\n  Status             : {status}")
    print("=" * 62)

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
