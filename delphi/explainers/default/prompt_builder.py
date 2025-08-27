from .prompts import system, system_single_token


def build_prompt(
    examples: str,
    activations: bool = False,
    cot: bool = False,
) -> list[dict]:
    messages = system(cot=cot)

    # Hard-coded few-shot user prompt and assistant response for the tuned model
    few_shot_user_prompt = """Neuron 1:

Excerpt 1: The two men fought fiercely, swords clashing in a deadly duel. But in the end, only one could emerge victorious. With a final powerful swing, John ran his blade through the other man's chest, {{ending his life}}.
Excerpt 2: Adeline knew she wouldn't make it out of the burning building. The flames were too intense, and the smoke was choking her. As the flames engulfed her, she thought of her loved ones one last time before succ{{umbing to the fire}}.
Excerpt 3: The gunshot rang out in the empty alley, and Jim fell to the ground, blood pooling around him. He tried to hold on, to fight for life, but the wound was too severe. As his vision {{faded to black}}, he wondered who would take care of his little girl now.
Excerpt 4: The doctor told Tessa she only had a few months left to live. But she was determined to make the most of it. She traveled, spent time with loved ones, and tried new things. And against all odds, her health improved, and she went on to live a long and fulfilling life."""

    few_shot_assistant_response = "[DESCRIPTION]: phrases in a passage that indicate a character has died"

    messages.extend([
        {"role": "user", "content": few_shot_user_prompt},
        {"role": "assistant", "content": few_shot_assistant_response},
    ])

    # Add the final user prompt with the actual examples for the current latent
    messages.append({"role": "user", "content": examples})

    return messages


def build_single_token_prompt(examples):
    messages = system_single_token()

    user_start = f"WORDS: {examples}"

    messages.append({"role": "user", "content": user_start})

    return messages
