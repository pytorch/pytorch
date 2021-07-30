import textwrap
import timeit
import typing

from torch.utils.benchmark._impl import constants
from torch.utils.benchmark._impl.tasks import base as task_base
from torch.utils.benchmark._impl.templates import jit as jit_template
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

        self.worker.run(jit_template.generate(work_spec=self._work_spec))

        if self._work_spec.language == constants.Language.CPP:
            assert self._timer is None
            self._maybe_timer_arg = ""

        elif self._timer in (None, timeit.default_timer):
            self._maybe_timer_arg = "timer=timeit.default_timer,"

        else:
            self.worker.store("_timeit_task_timer", self._timer, in_memory=True)
            self._maybe_timer_arg = "timer=_timeit_task_timer,"

    @property
    def worker(self) -> worker_base.WorkerBase:
        return self._worker

    def timeit(self, number: int) -> float:
        self.worker.run(textwrap.dedent(f"""
            _timeit_task_result = {constants.COMPILED_MODULE_NAME}.measure_wall_time(
                n_iter={number},
                n_warmup_iter=1,
                cuda_sync=False,
                {self._maybe_timer_arg}
            )
        """))
        result = self.worker.load("_timeit_task_result")
        assert isinstance(result, float)
        return result
