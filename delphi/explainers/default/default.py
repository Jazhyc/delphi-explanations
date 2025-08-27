import asyncio
from dataclasses import dataclass

from ..explainer import ActivatingExample, Explainer
from .prompt_builder import build_prompt


@dataclass
class DefaultExplainer(Explainer):
    activations: bool = True
    """Whether to show activations to the explainer."""
    cot: bool = False
    """Whether to use chain of thought reasoning."""

    def _build_prompt(self, examples: list[ActivatingExample]) -> list[dict]:
        # Build a single user-facing prompt following the tuned model format
        prompt_lines = ["Neuron 1:\n"]

        for i, example in enumerate(examples):
            str_toks = example.str_tokens or []
            activations = example.activations.tolist()

            # Use parent's _highlight which now emits {{ }} delimiters
            highlighted_text = self._highlight(str_toks, activations)

            # Format as 'Excerpt X:' lines
            prompt_lines.append(f"Excerpt {i + 1}: {highlighted_text}")

        final_user_prompt = "\n".join(prompt_lines)

        return build_prompt(
            examples=final_user_prompt,
            activations=self.activations,
            cot=self.cot,
        )

    def call_sync(self, record):
        return asyncio.run(self.__call__(record))
