from __future__ import annotations

from typing import Iterable

from .types import InputSource
from .matchers import (
    SourceMatcher,
    match_local_directory,
    match_local_file,
)


class SourceDetector:
    def __init__(self, matchers: Iterable[SourceMatcher] | None = None) -> None:
        self._matchers: list[SourceMatcher] = list(matchers) if matchers is not None else []

    def register(self, matcher: SourceMatcher) -> None:
        self._matchers.append(matcher)

    def detect(self, raw_input: str) -> InputSource:
        for matcher in self._matchers:
            source = matcher(raw_input)
            if source is not None:
                return source

        raise ValueError(f"Unable to determine input source for: '{raw_input}'")

    @classmethod
    def default(cls) -> SourceDetector:
        return cls(
            matchers=[
                match_local_directory,
                match_local_file,
            ]
        )
