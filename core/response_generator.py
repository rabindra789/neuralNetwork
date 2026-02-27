"""
ResponseGenerator — uses Ollama's chat API to produce natural language replies.

After the neural network classifies user input into one of 12 intent classes,
this module asks llama3.2:3b to craft a short, contextual response.
"""

import logging

import ollama

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, CLASS_NAMES

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are the conversational module of an Artificial Brain simulator. "
    "A neural network has just classified the user's message. "
    "You receive the classification result and must respond helpfully "
    "in 1-3 sentences. Be concise, friendly, and relevant to what the user said. "
    "Do NOT mention the classification system or confidence scores to the user — "
    "just respond naturally as if you understood their message."
)


class ResponseGenerator:
    """Wraps Ollama chat for post-classification response generation."""

    def __init__(self):
        self._client = ollama.Client(host=OLLAMA_BASE_URL)
        self._model = OLLAMA_MODEL

    def generate(
        self,
        user_text: str,
        predicted_class: int,
        confidence: float,
    ) -> str:
        """
        Generate a conversational response.

        Parameters
        ----------
        user_text        : the original sentence the user typed
        predicted_class  : 0..11 intent index
        confidence       : max softmax probability (0.0 .. 1.0)
        """
        class_name = CLASS_NAMES[predicted_class]

        context = (
            f"[Internal context — do not reveal to user]\n"
            f"Classification: {class_name} ({confidence:.0%} confidence)\n"
            f"User message: \"{user_text}\""
        )

        try:
            resp = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                options={"temperature": 0.7, "num_predict": 150},
            )
            return resp["message"]["content"].strip()
        except Exception as exc:
            log.error("Ollama chat failed: %s", exc)
            return f"(Response generation failed: {exc})"
