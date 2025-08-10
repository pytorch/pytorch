from abc import ABC, abstractmethod


class BaseRunner(ABC):
    @abstractmethod
    def run(self) -> None:
        """执行主逻辑"""
        pass
