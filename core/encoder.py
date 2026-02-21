"""
TextEncoder — converts raw user text into an 8-dimensional float feature vector.

Design rationale:
  The encoder uses hand-crafted linguistic features rather than an embedding
  model.  This is intentional for an exhibition context:
    1. Each dimension has a human-readable meaning — students can understand
       exactly what the network is 'seeing'.
    2. Zero external ML dependencies at the input boundary.
    3. Deterministic → same input always produces the same vector, so the
       visualisation is reproducible during demos.

Feature vector layout (all values normalised to [0, 1]):
  [0]  greeting_score  — fraction of greeting keywords found
  [1]  question_score  — fraction of request/question keywords found
  [2]  positive_score  — fraction of positive-affirmation keywords found
  [3]  char_density    — character count / 100 (capped at 1.0)
  [4]  has_question    — 1.0 if '?' present, else 0.0
  [5]  has_exclaim     — 1.0 if '!' present, else 0.0
  [6]  word_density    — word count / 20 (capped at 1.0)
  [7]  cap_ratio       — upper-case letter fraction (emphasis signal)
"""

import re
import numpy as np


class TextEncoder:
    """Converts a text string into a fixed-width (8-dim) float32 feature vector."""

    # ── Vocabulary sets ───────────────────────────────────────────────────────

    _GREETINGS = {
        "hi", "hello", "hey", "greetings", "howdy", "sup", "yo",
        "hiya", "salut", "namaste", "morning", "evening", "afternoon",
    }

    _QUESTIONS = {
        "what", "how", "where", "when", "why", "who", "which",
        "can", "could", "would", "should", "is", "are", "do",
        "does", "did", "will", "please", "help", "need", "want",
        "request", "ask", "tell", "explain", "show", "give", "describe",
    }

    _POSITIVES = {
        "yes", "yeah", "yep", "sure", "okay", "ok", "great",
        "wonderful", "love", "like", "good", "excellent", "perfect",
        "awesome", "fantastic", "nice", "fine", "thanks", "thank",
        "appreciate", "agree", "correct", "right", "true", "absolutely",
        "definitely", "certainly", "indeed", "of course", "no problem",
    }

    # Multi-word greeting phrases checked against the raw (lowercased) string
    _GREETING_PHRASES = [
        "good morning", "good afternoon", "good evening",
        "what's up", "whats up", "how are you",
    ]

    # ── public API ────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """
        Return an 8-dimensional float32 numpy array for the given text.
        Always returns a valid vector (zeros for empty input).
        """
        if not text or not text.strip():
            return np.zeros(8, dtype=np.float32)

        low   = text.lower().strip()
        words = re.findall(r'\b\w+\b', low)

        if not words:
            return np.zeros(8, dtype=np.float32)

        # ── Feature 0: greeting score ─────────────────────────────────────
        g_count = sum(1.0 for w in words if w in self._GREETINGS)
        g_count += sum(1.5 for ph in self._GREETING_PHRASES if ph in low)
        f0 = min(g_count / 2.5, 1.0)

        # ── Feature 1: question / request score ───────────────────────────
        q_count = sum(1.0 for w in words if w in self._QUESTIONS)
        f1 = min(q_count / 3.5, 1.0)

        # ── Feature 2: positive affirmation score ─────────────────────────
        p_count = sum(1.0 for w in words if w in self._POSITIVES)
        f2 = min(p_count / 2.5, 1.0)

        # ── Feature 3: character density ──────────────────────────────────
        f3 = min(len(text) / 100.0, 1.0)

        # ── Feature 4: question mark ──────────────────────────────────────
        f4 = 1.0 if '?' in text else 0.0

        # ── Feature 5: exclamation mark ───────────────────────────────────
        f5 = 1.0 if '!' in text else 0.0

        # ── Feature 6: word density ───────────────────────────────────────
        f6 = min(len(words) / 20.0, 1.0)

        # ── Feature 7: capitalisation ratio ──────────────────────────────
        letters = re.findall(r'[a-zA-Z]', text)
        f7 = (sum(1 for c in letters if c.isupper()) / len(letters)
              if letters else 0.0)
        f7 = min(f7, 1.0)

        return np.array([f0, f1, f2, f3, f4, f5, f6, f7], dtype=np.float32)

    @staticmethod
    def feature_names() -> list[str]:
        """Human-readable names for each dimension — used in the UI."""
        return [
            "Greeting Score",
            "Question Score",
            "Positive Score",
            "Text Length",
            "Has '?'",
            "Has '!'",
            "Word Count",
            "Capitalisation",
        ]
