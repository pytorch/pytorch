import marshal
import textwrap
import typing

from torch.utils.benchmark._impl.workers import base


class InProcessWorker(base.WorkerBase):
    """Worker which reuses the current Python process.

    The implementation of this worker borrows from the builtin `timeit.Timer`
    class, and simply reuses the current interpreter. (Making it comparatively
    simple.) Note that as a result, it offers no protection against the GIL.
    """

    def __init__(self, globals: typing.Dict[str, typing.Any]):
        super().__init__()
        self._globals: typing.Dict[str, typing.Any] = globals

    @property
    def in_process(self) -> bool:
        return True

    def run(self, snippet: str):
        code = compile(
            textwrap.dedent(snippet),
            "<in-process-worker>",
            "exec",
        )
        exec(code, self._globals)

    # Serialize and deserialize during store and load to match the behavior of
    # workers with `in_process=False`.
    def store(self, name: str, value: typing.Any, in_memory: bool = False) -> None:
        if not in_memory:
            value = marshal.loads(marshal.dumps(value))

        self._globals[name] = value

    def load(self, name: str) -> typing.Any:
        return marshal.loads(marshal.dumps(self._globals[name]))
