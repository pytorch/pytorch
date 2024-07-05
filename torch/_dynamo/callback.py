# mypy: allow-untyped-defs
class CompilationCallbackHandler:
    def __init__(self):
        self.start_callbacks = []
        self.end_callbacks = []

    def register_start_callback(self, callback):
        """
        Register a callback function to be called when the compilation starts.

        Args:
        - callback (callable): The callback function to register.
        """
        self.start_callbacks.append(callback)
        return callback

    def register_end_callback(self, callback):
        """
        Register a callback function to be called when the compilation ends.

        Args:
        - callback (callable): The callback function to register.
        """
        self.end_callbacks.append(callback)
        return callback

    def remove_start_callback(self, callback):
        """
        Remove a registered start callback function.

        Args:
        - callback (callable): The callback function to remove.
        """
        self.start_callbacks.remove(callback)

    def remove_end_callback(self, callback):
        """
        Remove a registered end callback function.

        Args:
        - callback (callable): The callback function to remove.
        """
        self.end_callbacks.remove(callback)

    def run_start_callbacks(self):
        """
        Execute all registered start callbacks.
        """
        for callback in self.start_callbacks:
            callback()

    def run_end_callbacks(self):
        """
        Execute all registered end callbacks.
        """
        for callback in self.end_callbacks:
            callback()

    def clear(self):
        """
        Clear all registered callbacks.
        """
        self.start_callbacks.clear()
        self.end_callbacks.clear()


callback_handler = CompilationCallbackHandler()


def on_compile_start(callback):
    """
    Decorator to register a callback function for the start of the compilation.
    """
    callback_handler.register_start_callback(callback)
    return callback


def on_compile_end(callback):
    """
    Decorator to register a callback function for the end of the compilation.
    """
    callback_handler.register_end_callback(callback)
    return callback
