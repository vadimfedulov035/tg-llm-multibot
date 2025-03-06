"""LLM Text Generation Base Module.

Generator superclass orchestrates dialog processing, token management, and
LLM inference for specialized Responder/Selector subclasses.

Key Components:
- Dialog Processing:   Normalization and string convertion
- Token Management:    Mode-based configuration (respond/rate)
- Generation Pipeline: Prompt templating and stop sequence utilization
"""

import re
from typing import Optional

import torch

from models import LLM, Settings
from stopper import get_stop_vars, trim_stop_sequences


class Generator:
    """Superclass for Responder and Selector classes.

    self.generate(user_prompt: str) to generate LLM response.
    """

    SYSTEM_TEMPLATE = "<|start_header_id|>system<|end_header_id|>%s<|eot_id|>"
    USER_TEMPLATE = "<|start_header_id|>user<|end_header_id|>%s<|eot_id|>"
    ASSISTANT_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>"

    RESP_TOKEN_ERR_MSG = "No token number specified for response."
    MODE_ERR_MSG = "Mode is %s. Set to 'respond' or 'rate'!"
    RATE_TOKEN_ERR_MSG = "No token number for rate."

    def __init__(
        self,
        llm: LLM,
        settings: Settings,
        dialog: list[str],
        responses: Optional[list[str]],
    ) -> None:
        """Initialize Generator class.

        Args:
            self
            llm: LLM
            settings: Settings
            dialog: list[str]
            responses: Optional[list[str]]

        Returns: None
        """
        self._llm = llm
        self._settings = settings
        self._dialog = self.normalize_dialog(dialog)

        self._set_stopping_criteria()
        self._set_token_num()

        self.responses = responses


    @staticmethod
    def normalize_dialog(dialog: str) -> list[str]:
        """Normalize self.dialog by terminating new line characters.

        Args:
            self
            dialog: str

        Returns: list[str]
        """
        for i, msg in enumerate(dialog):
            if "\n" in msg:
                dialog[i] = re.sub("\n+", r"\\n", msg)
        return dialog


    @property
    def dialog(self) -> str:
        """Set self.dialog as property.

        Args:
            self

        Returns: str
        """
        return self._dialog


    @property
    def dialog_str(self) -> str:
        """Set self.dialog_str as property.

        Args:
            self

        Returns: str
        """
        return "\n".join(self.dialog)


    def _set_stopping_criteria(self) -> None:
        """Set self.stopping_criteria and self.stop_token_ids.

        Args:
            self

        Returns: None
        """
        tokenizer = self._llm.tokenizer
        dialog = self.dialog

        stopping_criteria, stop_token_ids = get_stop_vars(tokenizer, dialog)
        self.stopping_criteria = stopping_criteria
        self.stop_token_ids = stop_token_ids


    def _set_token_num(self) -> None:
        """Set self.max_new_tokens token number based on mode and values.

        select mode: rate_tokens* ?
        respond mode: response_tokens*/inputs.size(1) + response_token_shift ?
        * Must be non-zero; ? Raise error on failure

        Args:
            self

        Returns: None

        Raises
        ------
            ValueError: problem

        """
        tokenizer = self._llm.tokenizer
        mode = self._llm.mode
        settings = self._settings
        dialog = self.dialog

        match mode:
            case "rate":
                rate_tokens = settings.rate_tokens
                # Set static (specified)
                if rate_tokens != 0:
                    self.max_new_tokens = settings.rate_tokens
                else:
                    err = ValueError(self.RATE_TOKEN_ERR_MSG)
                    raise err
            case "response":
                response_tokens = settings.response_tokens
                response_token_shift = settings.response_token_shift
                # Set static
                if settings.response_tokens != 0:
                    self.max_new_tokens = response_tokens
                # Set with shift
                elif settings.response_token_shift != 0:
                    inputs = tokenizer.encode(dialog[-1], return_tensors="pt")
                    self.max_new_tokens = inputs.size(1) + response_token_shift
                else:
                    err = ValueError(self.RESP_TOKEN_ERR_MSG)
                    raise err
            case _:
                err = ValueError(self.MODE_ERR_MSG % mode)
                raise err


    def _new_prompt(self, user_prompt: str) -> str:
        """Format new prompt based on class templates and user prompt.

        Args:
            self
            user_prompt: str

        Returns: str
        """
        settings = self._settings

        system_prompt = settings.system_prompt

        prompt  = self.SYSTEM_TEMPLATE % system_prompt
        prompt += self.USER_TEMPLATE % user_prompt
        prompt += self.ASSISTANT_TEMPLATE

        return prompt


    def generate(self, user_prompt: str) -> str:
        """Generate response based on user prompt.

        Args:
            self
            user_prompt: str

        Returns: str
        """
        model = self._llm.model
        tokenizer = self._llm.tokenizer
        settings = self._settings

        max_new_tokens = self.max_new_tokens
        stop_token_ids = self.stop_token_ids
        stopping_criteria = self.stopping_criteria

        prompt = self._new_prompt(user_prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=True,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                temperature=settings.temperature,
                repetition_penalty=settings.repetition_penalty,
                top_p=settings.top_p,
                top_k=settings.top_k,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
            )
            processed_ids = trim_stop_sequences(output_ids, stop_token_ids)
            response_raw = tokenizer.batch_decode(
                processed_ids, skip_special_tokens=True
            )[0]
            response_raw = response_raw.split("assistant")[-1].strip()

        return response_raw
