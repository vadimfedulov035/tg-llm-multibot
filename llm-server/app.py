#!/usr/bin/env python

"""LLM Inference Server App.

FastAPI-based web service for tunable LLM inference.
Two-stage processing pipeline with configurable parameters.

Key Components:
1. API Endpoint:
   - POST /v1/generate: RequestBody(dialog, settings) -> ResponseBody(response)

2. Processing:
   - LLM Inference: Load LLM from './model' according to its config
   - Response Generation: Use Responder class to generate response candidates
   - Response Selection: Use Selector class to select response from candidates
   - Timing Metrics: Track processing time

3. Error Handling:
   - Return ResponseBody(ERR_MSG) for:
     * No response/rate candidates generated

Usage Example:
    $ uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    POST /v1/generate with JSON body containing dialog and settings
"""

import timeit
from collections.abc import Callable
from functools import wraps

import torch
from fastapi import FastAPI
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
)

from models import LLM, RequestBody, ResponseBody
from responder import Responder
from selector import Selector

MODEL = "./model"
ERR_MSG = "Server error."

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = MllamaForConditionalGeneration.from_pretrained(
    MODEL,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    torch_dtype=torch.float16,
    device_map="cuda",
)


llm = LLM(
    model=model,
    tokenizer=tokenizer,
    mode=None,
    verbose=True
)

app = FastAPI()


def timing(func: Callable) -> Callable:
    """Measure function runtime."""

    @wraps(func)
    def inner(*args: RequestBody) -> dict[str, str]:
        ts = timeit.default_timer()

        result = func(*args)

        te = timeit.default_timer()
        time = int(te - ts)

        minutes, seconds = time // 60, time % 60
        print(f"Time generating: {minutes}m {seconds}s")

        return result

    return inner


@timing
def generate(request: RequestBody) -> ResponseBody:
    """Generate response with specified LLM parameters."""
    error = ResponseBody(response=ERR_MSG)

    # base data objects
    settings = request.settings
    dialog = request.dialog

    # Generate responses
    responses = []
    llm.mode = "response"
    responder = Responder(llm, settings, dialog, responses=None)
    try:
        responses = responder.respond()
    except ValueError:
        return error

    # Select best response via rating
    response_str = ""
    llm.mode = "rate"
    selector = Selector(llm, settings, dialog, responses)
    try:
        response_str = selector.select()
    except ValueError:
        return error

    return ResponseBody(response=response_str)


@app.post("/v1/generate")
async def chat(request: RequestBody) -> ResponseBody:
    """Return response from LLM Server."""
    return generate(request)
