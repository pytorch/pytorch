import asyncio
import sys
import weakref
from asyncio import AbstractEventLoop, Future
from collections.abc import Awaitable, Coroutine, Generator, Iterator
from contextlib import contextmanager, ExitStack
from contextvars import Context
from typing import Any, Callable, Optional, Protocol, TypeVar

from torch.utils._ordered_set import OrderedSet


T = TypeVar("T")
TCoro = Generator[Any, None, T]

if sys.version_info >= (3, 11):

    class TaskFactory(Protocol):
        def __call__(
            self,
            __loop: AbstractEventLoop,
            __factory: Coroutine[None, None, object] | Generator[None, None, object],
            __context: Context | None = None,
            /,
        ) -> asyncio.futures.Future[object]: ...

    TaskFactoryType = TaskFactory
else:
    TaskFactoryType = Callable[[AbstractEventLoop, Generator[TCoro, None, T]], Future]  # type: ignore[valid-type]


def await_sync(awaitable: Awaitable[T]) -> T:
    with get_loop() as loop:
        return loop.run_until_complete(awaitable)


@contextmanager
def get_loop(
    always_create_new_loop: bool = False,
) -> Iterator[AbstractEventLoop]:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as re:
        if "There is no current event loop in thread" in str(re):
            with _new_loop() as loop:
                yield loop
            return
        else:
            raise

    @contextmanager
    def _restore_loop(
        loop: asyncio.AbstractEventLoop,
    ) -> Iterator[None]:
        try:
            yield
        finally:
            asyncio.set_event_loop(loop)

    @contextmanager
    def _restore_running_loop() -> Iterator[None]:
        loop_from_events = asyncio.events._get_running_loop()
        asyncio.events._set_running_loop(None)
        try:
            yield
        finally:
            asyncio.events._set_running_loop(loop_from_events)

    with ExitStack() as stack:
        if loop.is_running():
            stack.enter_context(_restore_running_loop())
            stack.enter_context(_restore_loop(loop=loop))
            loop = stack.enter_context(_new_loop(loop.get_task_factory()))  # type: ignore[arg-type]
        elif loop.is_closed():
            loop = stack.enter_context(_new_loop())  # type: ignore[arg-type]
        elif always_create_new_loop:
            stack.enter_context(_restore_loop(loop=loop))
            loop = stack.enter_context(_new_loop())  # type: ignore[arg-type]
        yield loop


@contextmanager
def _new_loop(
    task_factory: Optional[TaskFactoryType] = None,
) -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    tasks = _patch_loop(loop)

    if task_factory:
        # pyre-ignore[6]
        loop.set_task_factory(task_factory)  # type: ignore[arg-type]

    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        try:
            _cancel_all_tasks(loop, tasks)
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def _cancel_all_tasks(
    loop: AbstractEventLoop,
    tasks: OrderedSet[Future],  # type: ignore[type-arg]
) -> None:
    to_cancel = [task for task in tasks if not task.done()]

    if not to_cancel:
        return

    # pyre-fixme[1001]: Awaitable assigned to `task` is never awaited.
    for task in to_cancel:
        task.cancel()

    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )


def _patch_loop(loop: AbstractEventLoop) -> OrderedSet[Future]:  # type: ignore[type-arg]
    tasks: weakref.WeakSet[Future] = weakref.WeakSet()  # type: ignore[type-arg]

    task_factories: list[Optional[TaskFactoryType]] = [None]

    def _set_task_factory(factory: Optional[TaskFactoryType]) -> None:
        task_factories[0] = factory

    def _get_task_factory() -> Optional[TaskFactoryType]:
        return task_factories[0]

    def _safe_task_factory(
        loop: AbstractEventLoop,
        coro: TCoro,  # type: ignore[type-arg]
        *,
        context: Context | None = None,
    ) -> asyncio.Future:  # type: ignore[valid-type, type-arg]
        task_factory = task_factories[0]
        if task_factory is None:
            if sys.version_info >= (3, 11):
                task = asyncio.Task(coro, loop=loop, context=context)
            else:
                task = asyncio.Task(coro, loop=loop)
            # pyre-ignore[16]: `Task` has no attribute `_source_traceback`.
            if task._source_traceback:  # type: ignore[attr-defined]
                del task._source_traceback[  # type: ignore[attr-defined]
                    -1
                ]  # pragma: no cover  # type: ignore[attr-defined]
        else:
            if sys.version_info >= (3, 11):
                task = task_factory(loop, coro, context=context)  # type: ignore[arg-type, call-arg, assignment]
            else:
                task = task_factory(loop, coro)  # type: ignore[arg-type]
        #  `Union[Task[Any], Future[Any]]`.
        tasks.add(task)
        return task

    # pyre-ignore[6]
    loop.set_task_factory(_safe_task_factory)  # type: ignore[method-assign, arg-type]
    # pyre-ignore[8]
    loop.set_task_factory = _set_task_factory  # type: ignore[method-assign, assignment]
    # pyre-ignore[8]
    loop.get_task_factory = _get_task_factory  # type: ignore[method-assign, assignment]

    return tasks  # type: ignore[return-value]
