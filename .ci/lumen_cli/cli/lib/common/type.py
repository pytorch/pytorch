from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseRunner(ABC):
    def __init__(self, args: Optional[Any] = None) -> None:
        self.args = args

    @abstractmethod
    def run(self, args: Optional[Any] = None) -> None:
        """runs main logics, required"""
