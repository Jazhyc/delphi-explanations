import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple

import aiofiles

from delphi import logger

from ..clients.client import Client, Response
from ..latents.latents import ActivatingExample, LatentRecord


class ExplainerResult(NamedTuple):
    record: LatentRecord
    """Latent record passed through to scorer."""

    explanation: str
    """Generated explanation for latent."""


@dataclass
class Explainer(ABC):
    """
    Abstract base class for explainers.
    """

    client: Client
    """Client to use for explanation generation. """
    verbose: bool = False
    """Whether to print verbose output."""
    threshold: float = 0.3
    """The activation threshold to select tokens to highlight."""
    temperature: float = 0.0
    """The temperature for explanation generation."""
    generation_kwargs: dict = field(default_factory=dict)
    """Additional keyword arguments for the generation client."""
    stats: dict = field(
        default_factory=lambda: defaultdict(
            lambda: {
                "total_time": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "count": 0,
            }
        )
    )

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        messages = self._build_prompt(record.train)

        start_time = time.time()
        response = await self.client.generate(
            messages, temperature=self.temperature, **self.generation_kwargs
        )
        end_time = time.time()

        explainer_name = self.__class__.__name__
        self.stats[explainer_name]["total_time"] += end_time - start_time
        self.stats[explainer_name]["count"] += 1

        try:
            if isinstance(response, Response):
                response_text = response.text
                if response.prompt_tokens:
                    self.stats[explainer_name]["prompt_tokens"] += response.prompt_tokens
                if response.completion_tokens:
                    self.stats[explainer_name]["completion_tokens"] += response.completion_tokens
            else:
                response_text = response
            explanation = self.parse_explanation(response_text)
            if self.verbose:
                logger.info(f"Explanation: {explanation}")
                logger.info(f"Messages: {messages[-1]['content']}")
                logger.info(f"Response: {response}")

            return ExplainerResult(record=record, explanation=explanation)
        except Exception as e:
            logger.error(f"Explanation parsing failed: {repr(e)}")
            return ExplainerResult(
                record=record, explanation="Explanation could not be parsed."
            )

    def parse_explanation(self, text: str) -> str:
        try:
            match = re.search(r"\[DESCRIPTION\]:\s*(.*)", text, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return "Explanation could not be parsed."
        except Exception as e:
            logger.error(f"Explanation parsing regex failed: {repr(e)}")
            raise

    def _highlight(self, str_toks: list[str], activations: list[float]) -> str:
        result = ""
        threshold = max(activations) * self.threshold

        def check(i):
            return activations[i] > threshold

        i = 0
        while i < len(str_toks):
            if check(i):
                result += "{{"

                while i < len(str_toks) and check(i):
                    result += str_toks[i]
                    i += 1
                result += "}}"
            else:
                result += str_toks[i]
                i += 1

        return result

    def _join_activations(
        self,
        str_toks: list[str],
        token_activations: list[float],
        normalized_activations: list[float],
    ) -> str:
        acts = ""
        activation_count = 0
        for str_tok, token_activation, normalized_activation in zip(
            str_toks, token_activations, normalized_activations
        ):
            if token_activation > max(token_activations) * self.threshold:
                # TODO: for each example, we only show the first 10 activations
                # decide on the best way to do this
                if activation_count > 10:
                    break
                acts += f'("{str_tok}" : {int(normalized_activation)}), '
                activation_count += 1

        return "Activations: " + acts

    @abstractmethod
    def _build_prompt(self, examples: list[ActivatingExample]) -> list[dict]:
        pass


async def explanation_loader(
    record: LatentRecord, explanation_dir: str
) -> ExplainerResult:
    try:
        async with aiofiles.open(f"{explanation_dir}/{record.latent}.txt", "r") as f:
            explanation = json.loads(await f.read())
        return ExplainerResult(record=record, explanation=explanation)
    except FileNotFoundError:
        logger.info(f"No explanation found for {record.latent}")
        return ExplainerResult(record=record, explanation="No explanation found")


async def random_explanation_loader(
    record: LatentRecord, explanation_dir: str
) -> ExplainerResult:
    explanations = [f for f in os.listdir(explanation_dir) if f.endswith(".txt")]
    if str(record.latent) in explanations:
        explanations.remove(str(record.latent))
    random_explanation = random.choice(explanations)
    async with aiofiles.open(f"{explanation_dir}/{random_explanation}", "r") as f:
        explanation = json.loads(await f.read())

    return ExplainerResult(record=record, explanation=explanation)
