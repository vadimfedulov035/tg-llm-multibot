"""LLM Output Validation Module.

Validate module implements multi-criteria quality assurance for LLM responses
through lexical analysis and context-aware repetition detection.

Key Components:
- Length Enforcement:   Context-relative minimum response length requirements
- Content Policies:     Web reference detection, excessive questioning filters
- Repetition Checks:    Phrase-level Jaccard index repetition comparisons

Implementation Notes:
- Phrase-Based Analysis: Sentence splitting via terminal punctuation boundaries
- Threshold-Driven:      Configurable MAX_JACCARD_IDX and MIN_MSG_LEN constants
"""

import re

MAX_JACCARD_IDX = 0.5
MIN_MSG_LEN = 2 / 3


def _split_to_phrases(text: str) -> list[str]:
    """Split phrases based on delimeters.

    Args:
        text: str

    Returns: list[str]
    """
    delimiters = (".", "!", "?")
    pattern = "|".join(map(re.escape, delimiters))
    return re.split(pattern, text)


def _get_jaccard_idx(str1: str, str2: str) -> float:
    """Calculate Jaccard index to measure how one string repeats the other.

    Args:
        str1: str
        str2: str

    Returns: float
    """
    jaccard_idx = 0

    set1 = set(str1.split())
    set2 = set(str2.split())

    intersection_length = len(set1.intersection(set2))
    union_length = len(set1.union(set2))

    if union_length != 0:
        jaccard_idx = intersection_length / union_length

    return jaccard_idx


def _is_itself_repetitive(response: str) -> bool:
    """Check if response is repetitive to itself via Jaccard index.

    Args:
        response: str

    Returns: bool
    """
    jaccard_idx = 0
    phrases = _split_to_phrases(response)

    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            jaccard_idx = _get_jaccard_idx(phrases[i], phrases[j])
            if jaccard_idx > MAX_JACCARD_IDX:
                return True

    return False


def _is_prompt_repetitive(response: str, prompt: str) -> bool:
    """Check if response is repetitive to prompt via Jaccard index.

    Args:
        response: str
        prompt: str

    Returns: bool
    """
    jaccard_idx = 0

    response_phrases = _split_to_phrases(response)
    prompt_phrases = _split_to_phrases(prompt)

    for prompt_phrase in prompt_phrases:
        for response_phrase in response_phrases:
            jaccard_idx = _get_jaccard_idx(prompt_phrase, response_phrase)
            if jaccard_idx > MAX_JACCARD_IDX:
                return True

    return False


def _is_dialog_repetitive(response: str, dialog: list[str]) -> bool:
    """Check if response is repetitive to dialog via Jaccard index.

    Args:
        response: str
        dialog: list[str]

    Returns: bool
    """
    jaccard_idx = 0

    response_phrases = _split_to_phrases(response)

    for message in dialog:
        message_phrases = _split_to_phrases(message)

        for message_phrase in message_phrases:
            for response_phrase in response_phrases:
                jaccard_idx = _get_jaccard_idx(message_phrase, response_phrase)
                if jaccard_idx > MAX_JACCARD_IDX:
                    return True

    return False


def _is_repetitive(response: str, prompt: str, dialog: list[str]) -> bool:
    """Check if response is repetitive to itself, prompt or dialog.

    Args:
        response: str
        propmt: str
        dialog: list[str]

    Returns: bool
    """
    is_repetitive = False

    is_repetitive = _is_itself_repetitive(response)
    if is_repetitive:
        return is_repetitive

    is_repetitive = _is_prompt_repetitive(response, prompt)
    if is_repetitive:
        return is_repetitive

    is_repetitive = _is_dialog_repetitive(response, dialog)
    if is_repetitive:
        return is_repetitive

    return is_repetitive


def validate(resp: str, prompt: str, dialog: list[str]) -> tuple[bool, str]:
    """Check if response is web-related, questioning, short or repetitive.

    Args:
        response: str
        propmt: str
        dialog: list[str]

    Returns: tuple[bool, str]

    Note: prompt should be from user, as checking against system prompt
    can invalidate basic self-presentation.
    """
    err = ""
    last_msg = dialog[-1]

    web = "[RETEJO]" in resp
    short = len(resp) < (len(last_msg) * MIN_MSG_LEN)
    questioning = resp.count("?") > last_msg.count("?") + 1
    repetitive = _is_repetitive(resp, prompt, dialog)

    if web:
        err += " <Web>"
    if short:
        err += " <Short>"
    if questioning:
        err += " <Questioning>"
    if repetitive:
        err += " <Repetitive>"

    bad = web or short or questioning or repetitive
    ok = not bad

    return ok, err
