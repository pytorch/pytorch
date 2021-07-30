import abc
import typing


class WorkerBase(abc.ABC):
    """Interface for the core worker abstraction.

    Conceptually, a worker is modeled as a remote interactive Python terminal.
    One can send code to be executed (analogous to writing to stdin), and
    perform basic stores and loads (analogous to RPC).

    It is the responsibility of higher layers of the stack (e.g. those that
    call the worker) to generate any code that the worker needs; the worker
    itself is deliberately dumb. As a result, there are several restrictions
    on the semantics of a worker:

     1) Workers are individually scoped, and one should not assume any state
        is shared between the caller and worker unless explicitly set with
        `store` and `load` calls. However, worker state does persist across
        calls.

     2) Stores and loads go through a serialization step. This means they
        should only be basic types (specifically those supported by the
        marshal library), and the results will be copies rather than references.

     3) Framework code will often live side-by-side with user code. Framework
        code should take care to scope implementation details to avoid leaking
        variables, and choose names defensively.

        Good:
        ```
            def _timer_impl_call():
                import my_lib
                my_value = 1
                my_lib.foo(my_value)
            _timer_impl_call()
            del _timer_impl_call
        ```

        Bad: (Leaks `my_lib` and `my_value` into user variable space.)
        ```
            import my_lib
            my_value = 1
            my_lib.foo(my_value)
        ```

     4) One must take care when importing code in the worker, because changes
        to `sys.path` in the parent may not be reflected in the child.
    """

    @abc.abstractmethod
    def run(self, snippet: str) -> None:
        """Execute snippet (Python code), and return when complete."""
        ...

    @abc.abstractmethod
    def store(self, name: str, value: typing.Any, *, in_memory: bool = False) -> None:
        """Assign `value` to `name` in the worker.

        (This will be a copy if `in_memory=False`)
        """
        ...

    @abc.abstractmethod
    def load(self, name: str) -> typing.Any:
        """Fetch a copy of `name` from worker, and return it to the caller."""
        ...

    @abc.abstractproperty
    def in_process(self) -> bool:
        """Is this worker in the same process as the caller.

        This property can be used to gate certain features. (Such as sharing
        in-memory objects) However it should be used sparringly, as it violates
        the abstraction of a purely remote worker.
        """
        ...
