import textwrap
import timeit
import typing

from torch.utils.benchmark._impl import constants
from torch.utils.benchmark._impl.tasks import base as task_base
from torch.utils.benchmark._impl.workers import base as worker_base
from torch.utils.benchmark._impl.workers import in_process_worker


class TimeitTask(task_base.TaskBase):

    _worker: worker_base.WorkerBase
    _timer_arg: str

    def __init__(
        self,
        work_spec: constants.WorkSpec,
        timer: typing.Optional[typing.Callable[[],float]] = None,
        worker: typing.Optional[worker_base.WorkerBase] = None,
    ) -> None:
        self._work_spec = work_spec
        self._timer = timer
        self._worker = worker or in_process_worker.InProcessWorker({})

        if self._work_spec.language == constants.Language.CPP:
            raise NotImplementedError("TODO: Support C++ snippets")

        else:
            assert self._work_spec.language == constants.Language.PYTHON
            if self._timer is None or self._timer is timeit.default_timer:
                self.worker.run(textwrap.dedent("""
                    import timeit
                    _timeit_task_timer = timeit.default_timer
                """))
            else:
                self.worker.store("_timeit_task_timer", self._timer, in_memory=True)

            # Note: A later PR will provide a more ergonomic approach.
            self.worker.run("\n".join([
                "def _timeit_task_inner_f(_timeit_task_number: int):",
                textwrap.indent(self._work_spec.setup, " " * 4),
                "    _timeit_task_start_time = _timeit_task_timer()",
                f"    for _ in range(_timeit_task_number):",
                textwrap.indent(self._work_spec.stmt, " " * 8),
                "    return _timeit_task_timer() - _timeit_task_start_time",
            ]))

    @property
    def worker(self) -> worker_base.WorkerBase:
        return self._worker

    def timeit(self, number: int) -> float:
        self.worker.run(f"_timeit_task_result = _timeit_task_inner_f({number})")
        result = self.worker.load("_timeit_task_result")
        assert isinstance(result, float)
        return result
