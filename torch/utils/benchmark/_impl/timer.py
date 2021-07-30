import timeit
from typing import Any, Callable, Dict, Optional, Union

from torch.utils.benchmark._impl import constants
from torch.utils.benchmark._impl.tasks import wall_time
from torch.utils.benchmark._impl.workers import in_process_worker


class Timer:
    def __init__(
        self,
        stmt: str = "pass",
        setup: str = "pass",
        global_setup: str = "",
        timer: Optional[Callable[[], float]] = None,
        globals: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
        sub_label: Optional[str] = None,
        description: Optional[str] = None,
        env: Optional[str] = None,
        num_threads: int = 1,
        language: Union[constants.Language, str] = constants.Language.PYTHON,
    ):
        self._work_spec = constants.WorkSpec(
            stmt=stmt,
            setup=setup,
            global_setup=global_setup,
            num_threads=num_threads,
            language=language,
        )

        self._metadata: constants.TaskSpec = constants.WorkMetadata(
            label=label,
            sub_label=sub_label,
            description=description,
            env=env,
        )

        if self._work_spec.language == constants.Language.CPP:
            if globals is not None:
                raise ValueError("Cannot pass `globals` for C++ snippet.")

            if timer is not None:
                raise ValueError("Cannot override `timer` for C++ snippet.")

        # We copy `globals` to prevent mutations from leaking.
        # (For instance, `eval` adds the `__builtins__` key)
        self._globals = dict(globals or {})
        self._timer = timer

    def timeit(self, number: int = 1000000) -> float:
        return wall_time.TimeitTask(
            work_spec=self._work_spec,
            timer=self._timer,
            worker=in_process_worker.InProcessWorker(globals=self._globals)
        ).timeit(number=number)
