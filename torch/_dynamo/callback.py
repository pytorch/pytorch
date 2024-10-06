from dataclasses import dataclass, field  # noqa: F811
from typing import Callable, List


@dataclass
class CompilationCallbackHandler:
    start_callbacks: List[Callable[[], None]] = field(default_factory=list)
    end_callbacks: List[Callable[[], None]] = field(default_factory=list)

    def register_start_callback(
        self, callback: Callable[[], None]
    ) -> Callable[[], None]:
        """
        Register a callback function to be called when the compilation starts.

        Args:
        - callback (Callable): The callback function to register.
        """
        self.start_callbacks.append(callback)
        return callback

    def register_end_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
        """
        Register a callback function to be called when the compilation ends.

        Args:
        - callback (Callable): The callback function to register.
        """
        self.end_callbacks.append(callback)
        return callback

    def remove_start_callback(self, callback: Callable[[], None]) -> None:
        """
        Remove a registered start callback function.

        Args:
        - callback (Callable): The callback function to remove.
        """
        self.start_callbacks.remove(callback)

    def remove_end_callback(self, callback: Callable[[], None]) -> None:
        """
        Remove a registered end callback function.

        Args:
        - callback (Callable): The callback function to remove.
        """
        self.end_callbacks.remove(callback)

    def run_start_callbacks(self) -> None:
        """
        Execute all registered start callbacks.
        """
        for callback in self.start_callbacks:
            callback()

    def run_end_callbacks(self) -> None:
        """
        Execute all registered end callbacks.
        """
        for callback in self.end_callbacks:
            callback()

    def clear(self) -> None:
        """
        Clear all registered callbacks.
        """
        self.start_callbacks.clear()
        self.end_callbacks.clear()


callback_handler = CompilationCallbackHandler()


def on_compile_start(callback: Callable[[], None]) -> Callable[[], None]:
    """
    Decorator to register a callback function for the start of the compilation.
    """
    callback_handler.register_start_callback(callback)
    return callback


def on_compile_end(callback: Callable[[], None]) -> Callable[[], None]:
    """
    Decorator to register a callback function for the end of the compilation.
    """
    callback_handler.register_end_callback(callback)
    return callback
