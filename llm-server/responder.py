"""Chain of Thought Response Generation Module.

Responder class manages parallelized Chain of Thought reasoning for response
generation with cleaning, validation, retries and batch processing.

Key Components:
- Responder:      Orchestration of CoT workflow with colored console feedback
- Thought Chains: Parallel reasoning paths managed via NumPy arrays
- Clean/Validate: Imported validation and cleaning pipelines for LLM output
"""

from contextlib import contextmanager

import numpy as np
from colorama import Fore, init
from numpy.typing import NDArray

from clean import clean
from generator import Generator
from validate import validate

init(autoreset=True)


class Responder(Generator):
    """Responder initialized with Generator initialize method.

    self.respond() to respond based on passed dialog.
    """

    SUCCESS_MSG = Fore.GREEN + "[Success]"
    FAILURE_MSG = Fore.RED + "[Failure]"
    OVERFLOW_MSG = Fore.RED + "[Fatal] Max attempts exceeded. Generation skip."
    FATAL_MSG = "No thoughts generated with prompt: %s."

    def _think(self, chain_prompt: str, thought_chain: NDArray) -> list[str]:
        """Generate thought based on chain prompt with expanded thought chain.

        Args:
            chain_prompt: str
            thought_chain: NDArray

        Returns: list[str]

        Raises
        ------
            ValueError: problem

        """
        thoughts = []

        settings = self._settings
        dialog, dialog_str = self.dialog, self.dialog_str
        verbose = self._llm.verbose

        batch_size = settings.response_batch_size
        max_attempts = batch_size * 3

        user_prompt = chain_prompt.format(dialog_str, *thought_chain)
        if verbose:
            print(user_prompt)

        for attempt in range(max_attempts):
            current_try = attempt + 1
            print(f"Try {current_try:02}:", end=" ")

            thought_raw = self.generate(user_prompt)
            thought = clean(thought_raw)

            ok, err = validate(thought, chain_prompt, dialog)
            if ok:
                print(self.SUCCESS_MSG)
                thoughts.append(thought)
                if verbose:
                    print(thought_raw)
                if len(thoughts) >= batch_size:
                    break
            else:
                print(self.FAILURE_MSG, end=" ")
                print(err)

        if len(thoughts) < batch_size:
            print(self.OVERFLOW_MSG)

        if len(thoughts) == 0:
            raise ValueError(self.FATAL_MSG % chain_prompt)

        return thoughts


    @contextmanager
    def _temp_batch_size(self, new_size: int) -> None:
        """Temporarily modify the batch size.

        Args:
            self
            new_size: int
        """
        original = self._settings.response_batch_size
        self._settings.response_batch_size = new_size
        try:
            yield
        finally:
            self._settings.response_batch_size = original


    def respond(self) -> list[str]:
        """Implement Chain of Thought algorithm.

        Create and independently continue parrallel thought chains.
        Treat thoughts as responses if think prompts exhausted.

        Args:
            self
            chain_prompt: str
            thought_chain: NDArray

        Returns: list[str]

        Raises
        ------
            ValueError: problem
        """
        settings = self._settings
        dialog_str = self.dialog_str

        chain_prompts = settings.chain_prompts
        batch_size = settings.response_batch_size

        # For the initial think prompt
        # batch generate first thoughts in the chains
        print("Step 1: CoT start")
        thoughts = self._think(chain_prompts[0], np.array([dialog_str]))

        if len(chain_prompts) == 1:
            return thoughts
        thought_chains = np.array([thoughts])
        dim_error = "Chain dimension mismatch"
        assert thought_chains.shape[0] == len(chain_prompts), dim_error

        # For every non-initial think prompt
        # accumulate next thoughts to continue the chains in parallel
        print("Step 2: CoT continue")
        with self._temp_batch_size(self, 1):
            for chain_prompt in chain_prompts[1:]:
                thoughts.clear()

                for j in range(batch_size):
                    thoughts += self._think(chain_prompt, thought_chains[:, j])
                thought_chains = np.vstack((thought_chains, thoughts))

            responses = thoughts

        return responses
