"""Models used for LLM server."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class LLM:
    """Config for LLM."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    mode: Literal["response", "rate"] | None
    verbose: bool

@dataclass
class Settings:
    """Settings from bot/cmd config."""

    system_prompt: str
    chain_prompts: list[str]
    rate_prompt: str
    temperature: float
    repetition_penalty: float
    top_p: float
    top_k: int
    response_tokens: int
    response_token_shift: int
    response_batch_size: int
    rate_tokens: int
    rate_batch_size: int

class RequestBody(BaseModel):
    """Request passed to API endpoint."""

    dialog: list[str]
    settings: Settings


class ResponseBody(BaseModel):
    """Response from API endpoint."""

    response: str
