from abc import ABC, abstractmethod

class BaseRunner(ABC):
    @abstractmethod
    def run(self) -> None:
        """runs main logics, required"""
        pass
