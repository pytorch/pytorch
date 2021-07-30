from torch.utils.benchmark._impl import constants
from torch.utils.benchmark._impl.tasks import base as task_base
from torch.utils.benchmark._impl.templates import jit as jit_template
from torch.utils.benchmark._impl.workers import callgrind_worker

class CallgrindTask(task_base.TaskBase):

    def __init__(
        self,
        work_spec: constants.WorkSpec,
        worker: callgrind_worker.CallgrindWorker,
    ) -> None:
        self._work_spec = work_spec

        assert isinstance(worker, callgrind_worker.CallgrindWorker)
        self._worker = worker

        self.worker.run(jit_template.generate(work_spec=self._work_spec))

    @property
    def worker(self) -> callgrind_worker.CallgrindWorker:
        return self._worker

    def collect(self, n_iter: int) -> None:
        self._collect(
            n_iter=n_iter,
            n_warmup_iter=min(n_iter, 10),
            num_threads=self._work_spec.num_threads,
        )

    @task_base.run_in_worker(scoped=True)
    def _collect(
        n_iter: int,
        n_warmup_iter: int,
        num_threads: int,
    ) -> None:
        from torch.utils.benchmark._impl import runtime_utils
        from torch.utils.benchmark._impl.templates import jit as jit_template

        with runtime_utils.set_torch_threads(num_threads):
            jit_template.get().collect_callgrind(
                n_iter=n_iter,
                n_warmup_iter=n_warmup_iter,
            )
