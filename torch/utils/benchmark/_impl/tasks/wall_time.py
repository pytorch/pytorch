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
    _should_cuda_sync: typing.Optional[bool]

    def __init__(
        self,
        work_spec: constants.WorkSpec,
        timer: typing.Optional[typing.Callable[[],float]] = None,
        worker: typing.Optional[worker_base.WorkerBase] = None,
    ) -> None:
        self._work_spec = work_spec
        self._worker = worker or in_process_worker.InProcessWorker({})
        self.worker.run(jit_template.generate(work_spec=self._work_spec))

        # See `measure` for more details.
        self._should_cuda_sync = None

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
        result, self._should_cuda_sync = self.measure(
            n_iter=number,
            num_threads=self._work_spec.num_threads,
            custom_timer=self._custom_timer,
            should_cuda_sync=self._should_cuda_sync
        )
        return result

    @task_base.run_in_worker(scoped=True)
    def measure(
        n_iter: int,
        num_threads: int,
        custom_timer: bool,
        should_cuda_sync: typing.Optional[bool],
    ) -> typing.Tuple[float, bool]:
        from torch.utils.benchmark._impl import runtime_utils
        from torch.utils.benchmark._impl.templates import jit as jit_template

        with runtime_utils.set_torch_threads(num_threads):
            # The first time `measure` is called, we must determine if it is
            # necessary to syncronize CUDA. In some cases this can be done
            # statically: if PyTorch is not built with CUDA or a GPU is not
            # present, we know a priori that we don't need to sync.
            #
            # Alternatively, if we cannot make such a static determination we
            # run a few times using the Kineto profier, and check if any CUDA
            # events are observed.
            #
            # We pass `should_cuda_sync` back to the caller so that TimeitTask
            # can remember if it needs to sync, and subsequent calls can skip
            # the overhead of checking.
            if should_cuda_sync is None:
                should_cuda_sync = runtime_utils.ShouldCudaSynchronize.cuda_present()

            if should_cuda_sync is None:
                with runtime_utils.ShouldCudaSynchronize() as watch_cuda:
                    jit_template.get().call(n_iter=3)
                should_cuda_sync = watch_cuda.cuda_detected

            if not isinstance(should_cuda_sync, bool):
                raise ValueError(
                    "`should_cuda_sync` should have been narrowed to a bool, "
                    f"instead got `{should_cuda_sync}`. ({type(should_cuda_sync)})")

            # This is placed in the global namespace during Task init.
            kwargs = {"timer": globals()["_timeit_task_timer"]} if custom_timer else {}

            return jit_template.get().measure_wall_time(
                n_iter=n_iter,
                n_warmup_iter=1,
                cuda_sync=should_cuda_sync,
                **kwargs,
            ), should_cuda_sync
