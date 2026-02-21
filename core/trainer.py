"""
Trainer — manages training lifecycle for the deep semantic network.

Changes from v1:
  • encoder is injected (SemanticEncoder) — created outside in StartupWorker
    so model loading happens off the main thread.
  • 12-class dataset (8 examples × 12 intents = 96 samples).
  • pretrain() batch-encodes all texts at once — ~10× faster than sequential.
  • infer() accepts an optional memory_bias np.ndarray; when supplied it is added
    to the raw embedding before the forward pass (associative memory nudge).
  • infer() returns a 4-tuple: (probs, acts, pred, max_sim).  max_sim carries the
    highest cosine similarity from the memory search, used to decide whether to
    trigger the teal canvas flash.  If no bias was provided, max_sim = 0.0.
  • All tensors are created on config.DEVICE.
  • weight_deltas normalised per-layer (unchanged from v1).
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import DEVICE, LEARNING_RATE, PRETRAIN_EPOCHS
from core.network         import DeepBrainNetwork
from core.semantic_encoder import SemanticEncoder


# ── 12-class training dataset ─────────────────────────────────────────────────
# 8 examples per intent × 12 intents = 96 samples
# With SentenceTransformer embeddings, this is sufficient for good accuracy.

_DATASET: List[Tuple[str, int]] = [
    # 0: Greeting
    ("hello",                           0),
    ("hi there",                        0),
    ("hey how are you",                 0),
    ("good morning",                    0),
    ("good evening everyone",           0),
    ("greetings and salutations",       0),
    ("what's up",                       0),
    ("howdy partner",                   0),
    # 1: Farewell
    ("goodbye",                         1),
    ("bye see you later",               1),
    ("take care",                       1),
    ("farewell",                        1),
    ("see you tomorrow",                1),
    ("good night",                      1),
    ("later",                           1),
    ("I have to go now",                1),
    # 2: What-Question
    ("what is this",                    2),
    ("what are you",                    2),
    ("what does that mean",             2),
    ("what time is it",                 2),
    ("what is a neural network",        2),
    ("what is machine learning",        2),
    ("what happened here",              2),
    ("what should I do",                2),
    # 3: How-Question
    ("how does this work",              3),
    ("how can I do that",               3),
    ("how many neurons does it have",   3),
    ("how does the brain process",      3),
    ("how to train a neural network",   3),
    ("how does backpropagation work",   3),
    ("how much memory does it use",     3),
    ("how do I get started",            3),
    # 4: Help Request
    ("can you help me",                 4),
    ("I need help with this",           4),
    ("please assist me",                4),
    ("help me understand",              4),
    ("I require your assistance",       4),
    ("could you please help",           4),
    ("I am stuck can you help",         4),
    ("support me with this task",       4),
    # 5: Info Request
    ("tell me about neural networks",   5),
    ("explain how this works",          5),
    ("give me more information",        5),
    ("describe the architecture",       5),
    ("I want to know more",             5),
    ("provide details about this",      5),
    ("inform me about the process",     5),
    ("what can you tell me about AI",   5),
    # 6: Positive Confirm
    ("yes that is correct",             6),
    ("absolutely right",                6),
    ("confirmed that is true",          6),
    ("I agree with you",                6),
    ("you are correct",                 6),
    ("that is accurate",                6),
    ("indeed that is right",            6),
    ("exactly what I meant",            6),
    # 7: Positive Praise
    ("great job well done",             7),
    ("excellent work",                  7),
    ("that is wonderful",               7),
    ("fantastic result",                7),
    ("amazing output",                  7),
    ("brilliant performance",           7),
    ("superb I love it",                7),
    ("outstanding achievement",         7),
    # 8: Negative Deny
    ("no that is wrong",                8),
    ("incorrect",                       8),
    ("I disagree",                      8),
    ("that is not right",               8),
    ("false negative",                  8),
    ("you are mistaken",                8),
    ("that is incorrect",               8),
    ("not at all",                      8),
    # 9: Negative Complaint
    ("this is terrible",                9),
    ("I do not like this",              9),
    ("it is broken",                    9),
    ("very disappointing",              9),
    ("this does not work properly",     9),
    ("awful experience",                9),
    ("this is frustrating",             9),
    ("I am not satisfied",              9),
    # 10: Command
    ("show me the result",              10),
    ("open the file",                   10),
    ("start the process",               10),
    ("run the program",                 10),
    ("execute this command",            10),
    ("launch the application",          10),
    ("display the output",              10),
    ("stop everything now",             10),
    # 11: Small Talk
    ("nice weather today",              11),
    ("how was your day",                11),
    ("I had a great lunch",             11),
    ("it has been a long day",          11),
    ("lovely afternoon",                11),
    ("just passing time",               11),
    ("random thoughts",                 11),
    ("chatting casually",               11),
]


class Trainer:
    """
    Owns DeepBrainNetwork and the injected SemanticEncoder.
    Provides pretrain(), infer(), learn_from_input().
    """

    def __init__(self, encoder: SemanticEncoder):
        self.encoder   = encoder
        self.network   = DeepBrainNetwork()          # moves to DEVICE in __init__
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4,
        )
        self.loss_history: List[float]               = []
        self.weight_deltas: Optional[List[np.ndarray]] = None

        # GPU optimisations for RTX 3050 (Ampere, CUDA 12.8)
        if DEVICE.type == "cuda":
            # Let CuDNN benchmark & cache the fastest kernel for our fixed input shape
            torch.backends.cudnn.benchmark = True
            # TF32: Ampere-native tensor format — same exponent range as FP32
            # but 10× faster matmuls, negligible accuracy loss for inference
            torch.backends.cuda.matmul.allow_tf32  = True
            torch.backends.cudnn.allow_tf32         = True

    # ── startup pre-training ──────────────────────────────────────────────────

    def pretrain(
        self,
        epochs: int = PRETRAIN_EPOCHS,
        progress_cb: Optional[Callable[[int, float], None]] = None,
    ):
        """
        Batch-encode all 96 training samples, then run gradient descent.
        Batch encoding is ~10× faster than encoding one sentence at a time.
        """
        texts, labels = zip(*_DATASET)

        # Encode all texts in one batch (returns (96, 384) numpy array)
        X_np = self.encoder.encode_batch(list(texts))
        X    = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
        y    = torch.tensor(labels, dtype=torch.long).to(DEVICE)

        self.network.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            out, _  = self.network(X)
            loss    = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            lv = float(loss.item())
            self.loss_history.append(lv)

            if progress_cb and epoch % 10 == 0:
                progress_cb(epoch, lv)

    # ── inference ─────────────────────────────────────────────────────────────

    def infer(
        self,
        text: str,
        memory_bias: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray], int, float]:
        """
        No-gradient forward pass.

        memory_bias (optional): np.ndarray (384,) — a small associative nudge
        from BrainMemory.recall().  When supplied it is added to the raw
        embedding before the network sees it.

        Returns
        -------
        probs           : (12,) float32 numpy array — softmax probabilities
        activations     : list of 6 numpy arrays (one per layer, on CPU)
        predicted_class : argmax index 0..11
        max_sim         : float — caller supplies this from BrainMemory;
                          0.0 if no bias was provided (used for UI display only)
        """
        embedding = self.encoder.encode(text)   # (384,) float32

        if memory_bias is not None:
            embedding = embedding + memory_bias  # gentle associative nudge

        self.network.eval()
        with torch.no_grad():
            x        = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            out, acts = self.network(x)

        probs   = out.squeeze(0).cpu().numpy()
        acts_np = [a.squeeze(0).numpy() for a in acts]
        return probs, acts_np, int(np.argmax(probs)), 0.0

    # ── online learning ───────────────────────────────────────────────────────

    def learn_from_input(self, text: str, label: int) -> float:
        """
        One gradient-descent step on a single (text, label) pair.
        Captures per-layer weight deltas for the backprop flash visualisation.
        """
        weights_before = self.network.get_raw_weights()

        self.network.train()
        embedding = self.encoder.encode(text)
        x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        y = torch.tensor([label], dtype=torch.long).to(DEVICE)

        self.optimizer.zero_grad()
        out, _  = self.network(x)
        loss    = self.criterion(out, y)
        loss.backward()
        self.optimizer.step()

        # Per-layer normalised weight deltas for orange flash intensity
        weights_after    = self.network.get_raw_weights()
        self.weight_deltas = []
        for wb, wa in zip(weights_before, weights_after):
            delta = np.abs(wa - wb)
            self.weight_deltas.append(delta / (delta.max() + 1e-8))

        lv = float(loss.item())
        self.loss_history.append(lv)
        return lv

    # ── helpers ───────────────────────────────────────────────────────────────

    def get_weight_magnitudes(self) -> List[np.ndarray]:
        return self.network.get_weight_magnitudes()
