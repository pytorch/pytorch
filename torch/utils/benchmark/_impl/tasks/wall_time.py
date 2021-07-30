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

    def __init__(
        self,
        work_spec: constants.WorkSpec,
        timer: typing.Optional[typing.Callable[[],float]] = None,
        worker: typing.Optional[worker_base.WorkerBase] = None,
    ) -> None:
        self._work_spec = work_spec
        self._worker = worker or in_process_worker.InProcessWorker({})
        self.worker.run(jit_template.generate(work_spec=self._work_spec))

        # `timeit.Timer` allows users to override the timer used, so we have
        # to support that functionality.
        self._custom_timer: bool = False
        if self._work_spec.language == constants.Language.CPP:
            assert timer is None

        elif timer not in (None, timeit.default_timer):
            self.worker.store("_timeit_task_timer", timer, in_memory=True)
            self._custom_timer = True

    @property
    def worker(self) -> worker_base.WorkerBase:
        return self._worker

    def timeit(self, number: int) -> float:
        return self.measure(n_iter=number, custom_timer=self._custom_timer)

    @task_base.run_in_worker(scoped=True)
    def measure(n_iter: int, custom_timer: bool) -> float:
        from torch.utils.benchmark._impl.templates import jit as jit_template

        # This is placed in the global namespace during Task init.
        kwargs = {"timer": globals()["_timeit_task_timer"]} if custom_timer else {}

        return jit_template.get().measure_wall_time(
            n_iter=n_iter,
            n_warmup_iter=1,
            cuda_sync=False,
            **kwargs,
        )
