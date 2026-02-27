"""
Trainer — manages training lifecycle for the deep semantic network.

v3 changes (Ollama):
  • encoder is now OllamaEncoder (was SemanticEncoder) — injected from StartupWorker.
  • 12-class dataset (20 examples × 12 intents = 240 samples).
  • pretrain() batch-encodes all texts at once via Ollama embed API.
  • infer() accepts an optional memory_bias np.ndarray; when supplied it is added
    to the raw embedding before the forward pass (associative memory nudge).
  • infer() returns a 4-tuple: (probs, acts, pred, max_sim).
  • load() validates that saved weights match the current embedding dimension.
  • All tensors are created on config.DEVICE.
  • save() / load() persist network + optimizer state across sessions.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import DEVICE, LEARNING_RATE, PRETRAIN_EPOCHS, BRAIN_SAVE_PATH, ACTUAL_LAYER_SIZES
from core.network         import DeepBrainNetwork
from core.ollama_encoder  import OllamaEncoder


# ── 12-class training dataset ─────────────────────────────────────────────────
# 20 examples per intent × 12 intents = 240 samples

_DATASET: List[Tuple[str, int]] = [
    # 0: Greeting
    ("hello",                                   0),
    ("hi there",                                0),
    ("hey how are you",                         0),
    ("good morning",                            0),
    ("good evening everyone",                   0),
    ("greetings and salutations",               0),
    ("what's up",                               0),
    ("howdy partner",                           0),
    ("hey there friend",                        0),
    ("good afternoon",                          0),
    ("hi how are things",                       0),
    ("yo what is going on",                     0),
    ("hello there nice to meet you",            0),
    ("hey long time no see",                    0),
    ("welcome back",                            0),
    ("oh hi I did not see you there",           0),
    ("morning how did you sleep",               0),
    ("salutations friend",                      0),
    ("hey I am glad you are here",              0),
    ("hi everyone good to see you",             0),
    # 1: Farewell
    ("goodbye",                                 1),
    ("bye see you later",                       1),
    ("take care",                               1),
    ("farewell",                                1),
    ("see you tomorrow",                        1),
    ("good night",                              1),
    ("later",                                   1),
    ("I have to go now",                        1),
    ("catch you later",                         1),
    ("until next time",                         1),
    ("I am heading out",                        1),
    ("see ya",                                  1),
    ("have a good one",                         1),
    ("time to leave",                           1),
    ("I will be off now",                       1),
    ("so long",                                 1),
    ("bye take care of yourself",               1),
    ("I must be going now",                     1),
    ("peace out",                               1),
    ("signing off goodbye",                     1),
    # 2: What-Question
    ("what is this",                            2),
    ("what are you",                            2),
    ("what does that mean",                     2),
    ("what time is it",                         2),
    ("what is a neural network",                2),
    ("what is machine learning",                2),
    ("what happened here",                      2),
    ("what should I do",                        2),
    ("what exactly is deep learning",           2),
    ("what is the purpose of this",             2),
    ("what kind of model is this",              2),
    ("what is an activation function",          2),
    ("what is the output",                      2),
    ("what does this button do",                2),
    ("what is the difference between AI and ML",2),
    ("what language is this written in",        2),
    ("what are the layers for",                 2),
    ("what is gradient descent",                2),
    ("what does backpropagation do",            2),
    ("what is a transformer",                   2),
    # 3: How-Question
    ("how does this work",                      3),
    ("how can I do that",                       3),
    ("how many neurons does it have",           3),
    ("how does the brain process",              3),
    ("how to train a neural network",           3),
    ("how does backpropagation work",           3),
    ("how much memory does it use",             3),
    ("how do I get started",                    3),
    ("how is the prediction made",              3),
    ("how does it learn",                       3),
    ("how do weights update",                   3),
    ("how fast is inference",                   3),
    ("how many parameters does it have",        3),
    ("how does attention work",                 3),
    ("how do I improve accuracy",               3),
    ("how does the loss decrease",              3),
    ("how is the embedding computed",           3),
    ("how do I add more classes",               3),
    ("how does memory recall work",             3),
    ("how is cosine similarity calculated",     3),
    # 4: Help Request
    ("can you help me",                         4),
    ("I need help with this",                   4),
    ("please assist me",                        4),
    ("help me understand",                      4),
    ("I require your assistance",               4),
    ("could you please help",                   4),
    ("I am stuck can you help",                 4),
    ("support me with this task",               4),
    ("I need guidance here",                    4),
    ("can someone explain this to me",          4),
    ("I am confused please help",               4),
    ("assist me please",                        4),
    ("help me figure this out",                 4),
    ("I need a hand with this",                 4),
    ("can you walk me through this",            4),
    ("please guide me",                         4),
    ("I do not understand help me",             4),
    ("need some help over here",                4),
    ("can you lend me a hand",                  4),
    ("I am lost please assist",                 4),
    # 5: Info Request
    ("tell me about neural networks",           5),
    ("explain how this works",                  5),
    ("give me more information",                5),
    ("describe the architecture",               5),
    ("I want to know more",                     5),
    ("provide details about this",              5),
    ("inform me about the process",             5),
    ("what can you tell me about AI",           5),
    ("share some facts about deep learning",    5),
    ("elaborate on that",                       5),
    ("can you give me an overview",             5),
    ("I am curious about the training",         5),
    ("tell me more about embeddings",           5),
    ("describe what a neuron does",             5),
    ("explain the loss function",               5),
    ("give me an introduction to PyTorch",      5),
    ("teach me about transformers",             5),
    ("I want details on the memory system",     5),
    ("can you summarize how it works",          5),
    ("describe the forward pass",               5),
    # 6: Confirm
    ("yes that is correct",                     6),
    ("absolutely right",                        6),
    ("confirmed that is true",                  6),
    ("I agree with you",                        6),
    ("you are correct",                         6),
    ("that is accurate",                        6),
    ("indeed that is right",                    6),
    ("exactly what I meant",                    6),
    ("yes exactly",                             6),
    ("correct",                                 6),
    ("that is right",                           6),
    ("affirmative",                             6),
    ("yep that is it",                          6),
    ("yes I confirm that",                      6),
    ("spot on",                                 6),
    ("you got it right",                        6),
    ("that matches perfectly",                  6),
    ("100 percent correct",                     6),
    ("yes agreed",                              6),
    ("precisely",                               6),
    # 7: Praise
    ("great job well done",                     7),
    ("excellent work",                          7),
    ("that is wonderful",                       7),
    ("fantastic result",                        7),
    ("amazing output",                          7),
    ("brilliant performance",                   7),
    ("superb I love it",                        7),
    ("outstanding achievement",                 7),
    ("you did great",                           7),
    ("I am impressed",                          7),
    ("wonderful job",                           7),
    ("that is exceptional",                     7),
    ("very well done",                          7),
    ("I am really pleased",                     7),
    ("top notch work",                          7),
    ("this is awesome",                         7),
    ("terrific output",                         7),
    ("magnificent result",                      7),
    ("you nailed it",                           7),
    ("kudos excellent job",                     7),
    # 8: Deny
    ("no that is wrong",                        8),
    ("incorrect",                               8),
    ("I disagree",                              8),
    ("that is not right",                       8),
    ("you are mistaken",                        8),
    ("that is incorrect",                       8),
    ("not at all",                              8),
    ("nope",                                    8),
    ("absolutely not",                          8),
    ("wrong answer",                            8),
    ("I do not think so",                       8),
    ("that is false",                           8),
    ("no way",                                  8),
    ("I reject that",                           8),
    ("definitely not",                          8),
    ("that cannot be right",                    8),
    ("no I disagree entirely",                  8),
    ("that is a mistake",                       8),
    ("you have it backwards",                   8),
    ("negative",                                8),
    # 9: Complaint
    ("this is terrible",                        9),
    ("I do not like this",                      9),
    ("it is broken",                            9),
    ("very disappointing",                      9),
    ("this does not work properly",             9),
    ("awful experience",                        9),
    ("this is frustrating",                     9),
    ("I am not satisfied",                      9),
    ("this is unacceptable",                    9),
    ("I hate this",                             9),
    ("poor quality output",                     9),
    ("this keeps failing",                      9),
    ("I expected better",                       9),
    ("this is a mess",                          9),
    ("nothing is working",                      9),
    ("I am really annoyed",                     9),
    ("this is a disaster",                      9),
    ("horrible performance",                    9),
    ("completely useless",                      9),
    ("I give up this is too bad",               9),
    # 10: Command
    ("show me the result",                      10),
    ("open the file",                           10),
    ("start the process",                       10),
    ("run the program",                         10),
    ("execute this command",                    10),
    ("launch the application",                  10),
    ("display the output",                      10),
    ("stop everything now",                     10),
    ("begin training",                          10),
    ("reset the weights",                       10),
    ("clear the memory",                        10),
    ("save the model",                          10),
    ("load the previous state",                 10),
    ("print the results",                       10),
    ("close this window",                       10),
    ("go back to the beginning",                10),
    ("apply the changes",                       10),
    ("delete that entry",                       10),
    ("refresh the display",                     10),
    ("export the data",                         10),
    # 11: Small Talk
    ("nice weather today",                      11),
    ("how was your day",                        11),
    ("I had a great lunch",                     11),
    ("it has been a long day",                  11),
    ("lovely afternoon",                        11),
    ("just passing time",                       11),
    ("random thoughts",                         11),
    ("chatting casually",                       11),
    ("not much happening today",                11),
    ("just killing time",                       11),
    ("thinking out loud",                       11),
    ("this is relaxing",                        11),
    ("I enjoy these conversations",             11),
    ("life is good today",                      11),
    ("just hanging around",                     11),
    ("lovely day outside",                      11),
    ("I am just browsing",                      11),
    ("nothing serious just chatting",           11),
    ("have you seen any good movies lately",    11),
    ("just wanted to say hi",                   11),
]


class Trainer:
    """
    Owns DeepBrainNetwork and the injected SemanticEncoder.
    Provides pretrain(), infer(), learn_from_input().
    """

    def __init__(self, encoder: OllamaEncoder):
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

    def save(self, path: str = BRAIN_SAVE_PATH):
        """Persist network weights + optimizer state so learning survives restarts."""
        torch.save({
            "network":       self.network.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "loss_history":  self.loss_history,
        }, path)

    def load(self, path: str = BRAIN_SAVE_PATH) -> bool:
        """
        Load saved state from disk.
        Returns True if successful, False if file not found or incompatible.
        """
        import logging
        import os
        log = logging.getLogger(__name__)

        if not os.path.isfile(path):
            return False

        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        # Verify saved weights match current architecture (embedding dim may have changed)
        saved_fc1 = checkpoint["network"].get("fc1.weight")
        if saved_fc1 is not None and saved_fc1.shape[1] != ACTUAL_LAYER_SIZES[0]:
            log.warning(
                "Saved weights incompatible (fc1 input %d != current %d) — retraining.",
                saved_fc1.shape[1], ACTUAL_LAYER_SIZES[0],
            )
            os.remove(path)
            return False

        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.loss_history = checkpoint.get("loss_history", [])
        return True

    def pretrain(
        self,
        epochs: int = PRETRAIN_EPOCHS,
        progress_cb: Optional[Callable[[int, float], None]] = None,
    ):
        """
        Batch-encode all 240 training samples, then run gradient descent.
        Batch encoding is ~10× faster than encoding one sentence at a time.
        """
        texts, labels = zip(*_DATASET)

        # Encode all texts in one batch (returns (240, 384) numpy array)
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
        embedding: np.ndarray,
        memory_bias: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray], int, float]:
        """
        No-gradient forward pass.

        embedding   : (384,) float32 numpy array — pre-computed by SemanticEncoder.
                      Caller encodes once and reuses for memory store and infer.
        memory_bias : (384,) float32 or None — associative nudge from BrainMemory.

        Returns
        -------
        probs           : (12,) float32 numpy array — softmax probabilities
        activations     : list of 6 numpy arrays (one per layer, on CPU)
        predicted_class : argmax index 0..11
        max_sim         : 0.0 (kept for API compatibility; caller owns max_sim)
        """
        if memory_bias is not None:
            embedding = embedding + memory_bias  # gentle associative nudge

        self.network.eval()
        with torch.inference_mode():
            x         = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            out, acts = self.network(x)

        probs   = out.squeeze(0).cpu().numpy()
        acts_np = [a.squeeze(0).numpy() for a in acts]
        return probs, acts_np, int(np.argmax(probs)), 0.0

    # ── online learning ───────────────────────────────────────────────────────

    def learn_from_input(self, embedding: np.ndarray, label: int) -> float:
        """
        One gradient-descent step on a single (embedding, label) pair.
        Captures per-layer weight deltas for the backprop flash visualisation.

        embedding : (384,) float32 — same pre-computed embedding used for infer().
        """
        weights_before = self.network.get_raw_weights()

        self.network.train()
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
