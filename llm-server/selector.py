"""LLM Response Selection Module.

Selector subclass implements batch rating and quality-based response selection
through multi-attempt validation and statistical rate averaging.

Key Components:
- Rating Pipeline:   Regex-based score extraction with validity checks
- Batch Rating:      Parallel rating attempts with tqdm progress tracking
- Quality Selection: Descending-order response ranking with colored output

Note:
- Rate Limits: Strict [0-10] rating bounds with regex filtering
- Mode: 'rate' as the LLM part only rates, while selection is done by Selector
"""

import re

from colorama import Fore, init
from tqdm.auto import tqdm

from generator import Generator

init(autoreset=True)


class Selector(Generator):
    """Selector uses Generator initialize method.

    self.select() to rate response candidates and return the best one.
    """

    OVERFLOW_MSG = "Fatal: 10 times the batch size. Skipped generations."
    FATAL_MSG = "No rates generated with prompt: %s."
    MIN_RATE = 0
    MAX_RATE = 10

    def _to_rate(self, response: str) -> tuple[str, bool]:
        """Convert response to rate.

        Args:
            self
            response: str

        Returns: tuple[str, bool]
        """
        ok = True

        # Export rate from "Rate: 5.5/10"
        response = re.sub(r"^[Rr]ate:\s?", "", response)
        response = re.split(r"(?:\s|\.|/)", response)[0]

        digits = ''.join(char for char in response if char.isdigit())
        if not digits:
            ok = False
            return 0, ok

        rate = int(digits)
        if rate < self.MIN_RATE or rate > self.MAX_RATE:
            ok = False

        return rate, ok


    def _rate_average(self, pbar, response: str, idx: int, size: int) -> float:
        """Calculate average rate from up to batch size rates.

        Args:
            self
            pbar
            response: str
            idx: int (response index)
            size: int (batch size)

        Returns: float

        Raises
        ------
            ValueError: problem
        """
        rates = []

        settings = self._settings

        user_prompt = settings.rate_prompt.format(self.dialog_str, response)
        rate_batch_size = settings.rate_batch_size

        max_attempts = rate_batch_size * 10
        for attempt in range(max_attempts):
            # Update progress bar description with current status
            pbar.set_description(f"Batch: {idx:02}/{size:02}")
            pbar.set_postfix_str(f"Try: {attempt + 1:02}/{max_attempts:02}")

            rate_raw = self.generate(user_prompt)
            rate, ok = self._to_rate(rate_raw)

            # Update main progress bar if okay
            if ok:
                rates.append(rate)
                pbar.update(1)
                if len(rates) >= rate_batch_size:
                    break
        if len(rates) < rate_batch_size:
            pbar.write(self.OVERFLOW_MSG)

        if len(rates) == 0:
            raise ValueError(self.FATAL_MSG % user_prompt)

        # Clear attempt information when moving to next response
        pbar.set_postfix_str()

        average_rate = sum(rates) / len(rates) if rates else 0.0

        return average_rate

    def _rate_average_all(self) -> list[str]:
        """Calculate average rates for all responses with single progress bar.

        Args:
            self

        Returns: list[str]
        """
        rates = []
        responses = self.responses or []

        rate_batch_size = self._settings.rate_batch_size

        total_steps = len(responses) * rate_batch_size
        with tqdm(total=total_steps, desc="Total Progress") as pbar:
            for idx, response in enumerate(responses, 1):
                rate = self._rate_average(pbar, response, idx, len(responses))
                rates.append(rate)

        return rates

    def select(self) -> str:
        """Select the best response and print response in descending order.

        Args:
            self

        Returns: str
        """
        best_response = ""

        rates = self._rate_average_all()
        responses = self.responses or []

        verbose = self._llm.verbose

        zipped = zip(rates, responses)
        sorted_zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
        for i, (rate, response) in enumerate(sorted_zipped):
            if i == 0:
                best_response = response
            if verbose:
                colorer = Fore.LIGHTGREEN_EX if i == 0 else ""
                print(f"{colorer}Rate: {rate:.1f}/10\n{response}\n")
            else:
                break

        return best_response
