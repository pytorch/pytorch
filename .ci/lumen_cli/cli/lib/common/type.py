from abc import ABC, abstractmethod
from typing import Any


class BaseRunner(ABC):
    def __init__(self, args: Any) -> None:
        self.args = args

    @abstractmethod
    def run(self, args: Any) -> None:
        """runs main logics, required"""
