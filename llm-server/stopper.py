"""LLM Generation Termination Module.

Termination module provides dialog-aware early stopping mechanisms fortext
generation through dynamic stop sequence detection and post-processing cleanup.

Key Components:
- Sequence Detection:  Real-time token sequence matching during generation
- Dialog Processing:   Speaker identifyer extraction from conversation history
- Output Sanitization: Post-generation removal of matched stop sequences
"""

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnTokens(StoppingCriteria):
    """Dynamic stopping criteria based on token sequence detection.

    Monitors generation output for predefined token sequences, flagging
    sequences for termination when any configured stop pattern is matched.

    Args:
        stop_token_ids: List of token sequences to detect (list[list[int]])

    Methods_
        __call__: Batch-enabled sequence matching check
    """

    def __init__(self, stop_token_ids: list):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor,
                 _scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        """Evaluate current generation state for stop sequence matches.

        Args:
            input_ids: torch.LongTensor
            _scores: torch.FloatTensor

        Returns_
            torch.BoolTensor
        """
        # Initialize all sequences as not stopping
        stop_flags = torch.zeros((input_ids.shape[0],),
                                 dtype=torch.bool, device=input_ids.device)

        for stop_seq in self.stop_token_ids:
            if len(stop_seq) == 0:
                continue
            # Convert stop sequence to tensor on correct device
            stop_tokens = torch.tensor(stop_seq, device=input_ids.device)
            seq_len = stop_tokens.shape[0]

            # Skip if input is shorter than current stop sequence
            if input_ids.shape[-1] < seq_len:
                continue

            # Get last 'seq_len' tokens from input
            window = input_ids[:, -seq_len:]
            # Check for match with current stop sequence
            matches = (window == stop_tokens).all(dim=1)
            # Update stop flags
            stop_flags |= matches

        return stop_flags


def get_stop_vars(tokenizer, dialog):
    """Generate dialog-derived stopping criteria and token sequences.

    Args:
        tokenizer: HF Tokenizer for text->token conversion
        dialog: Conversation history for stop pattern extraction

    Returns_
        tuple: (configured stopping criteria, detected stop token sequences)

    Processes dialog lines to extract speaker prefixes as stop patterns.
    Example: "User: Hello" -> stop sequence ["U", "ser", ":"]
    """
    stop_strings = []
    for line in dialog:
        colon_pos = line.find(":")
        if colon_pos > 0:  # Only consider lines with non-empty speaker names
            stop_str = line[:colon_pos+1]
            if len(stop_str) > 1:  # Exclude patterns like ":"
                stop_strings.append(stop_str)

    stop_token_ids = [
        tokenizer.encode(stop_string, add_special_tokens=False)
        for stop_string in stop_strings
    ]

    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

    return stopping_criteria, stop_token_ids


def trim_stop_sequences(output_ids, stop_token_ids):
    """Remove detected stop sequences from generated outputs.

    Args:
        output_ids: Raw generated token sequences
        stop_token_ids: Stop patterns detected during generation

    Returns_
        list: Cleaned token sequences with stop patterns removed

    Longest-match-first removal prevents partial sequence retention.
    Processes sequences in reverse length order for optimal pattern erasure.
    """
    processed = []
    for output_id in output_ids:
        seq = output_id.tolist()
        removed = 0

        # Check all stop sequences from longest to shortest
        for stop_seq in sorted(stop_token_ids, key=len, reverse=True):
            stop_len = len(stop_seq)
            if stop_len == 0:
                continue

            # Check if sequence ends with this stop pattern
            if len(seq) >= stop_len and seq[-stop_len-removed:] == stop_seq:
                removed += stop_len

        processed.append(seq[:-removed] if removed > 0 else seq)

    return processed
