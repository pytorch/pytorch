# Owner(s): ["module: inductor"]
from __future__ import annotations

import copy
import gc
import pickle
import threading
import time
import unittest
import weakref
from typing import Any

import torch
from torch._inductor.output_code import CompiledFxGraph
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


class TestCompiledFxGraph(TestCase):
    def _make_graph(
        self, current_callable: Any, fx_graph_cache_key: str | None = None
    ) -> CompiledFxGraph:
        graph = CompiledFxGraph.__new__(CompiledFxGraph)
        graph.fx_kwargs = {}
        graph._compile_context = None
        graph.recursively_apply_fns = None
        graph.compiled_fn_runner = None
        graph._original_gm = None
        graph._serialized_original_gm = None
        graph._fx_graph_cache_key = fx_graph_cache_key
        graph.current_callable = current_callable
        return graph

    def test_current_callable_is_thread_local(self) -> None:
        def main_callable(inputs: list[str]) -> tuple[str, list[str]]:
            return ("main", inputs)

        def worker_callable(inputs: list[str]) -> tuple[str, list[str]]:
            return ("worker", inputs)

        graph = self._make_graph(main_callable)
        worker_ready = threading.Event()
        main_checked = threading.Event()
        worker_result: list[tuple[str, list[str]]] = []
        worker_errors: list[BaseException] = []

        def worker_fn() -> None:
            try:
                graph.current_callable = worker_callable
                worker_ready.set()
                self.assertTrue(main_checked.wait(5))
                worker_result.append(graph(["worker"]))
            except BaseException as e:
                worker_errors.append(e)

        worker = threading.Thread(target=worker_fn)
        worker.start()

        self.assertTrue(worker_ready.wait(5))
        self.assertIs(graph.current_callable, main_callable)
        self.assertEqual(graph(["main"]), ("main", ["main"]))

        main_checked.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        if worker_errors:
            raise worker_errors[0]

        self.assertEqual(worker_result, [("worker", ["worker"])])
        self.assertIs(graph.current_callable, main_callable)

    def test_current_callable_fresh_thread_uses_latest_base_callable(self) -> None:
        def original_callable(inputs: list[str]) -> tuple[str, list[str]]:
            return ("original", inputs)

        def wrapped_callable(inputs: list[str]) -> tuple[str, list[str]]:
            return ("wrapped", inputs)

        graph = self._make_graph(original_callable, "fresh-thread-cache-key")
        graph.current_callable = wrapped_callable
        self.assertEqual(graph(["main"]), ("wrapped", ["main"]))

        worker_result: list[tuple[str, list[str]]] = []
        worker_errors: list[BaseException] = []

        def worker_fn() -> None:
            try:
                worker_result.append(graph(["worker"]))
            except BaseException as e:
                worker_errors.append(e)

        worker = threading.Thread(target=worker_fn)
        worker.start()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        if worker_errors:
            raise worker_errors[0]

        self.assertEqual(worker_result, [("wrapped", ["worker"])])

    def test_current_callable_follows_thread_across_cache_equivalent_graphs(
        self,
    ) -> None:
        def main_callable(inputs: list[str]) -> tuple[str, list[str]]:
            return ("main", inputs)

        def worker_callable(inputs: list[str]) -> tuple[str, list[str]]:
            return ("worker", inputs)

        cache_key = "same-fx-graph-cache-key"
        main_graph = self._make_graph(main_callable, cache_key)
        self.assertEqual(main_graph(["main"]), ("main", ["main"]))

        worker_ready = threading.Event()
        worker_graph: list[CompiledFxGraph] = []
        worker_errors: list[BaseException] = []

        def worker_fn() -> None:
            try:
                graph = self._make_graph(worker_callable, cache_key)
                self.assertEqual(graph(["worker"]), ("worker", ["worker"]))
                worker_graph.append(graph)
                worker_ready.set()
            except BaseException as e:
                worker_errors.append(e)

        worker = threading.Thread(target=worker_fn)
        worker.start()
        self.assertTrue(worker_ready.wait(5))
        worker.join(5)
        self.assertFalse(worker.is_alive())
        if worker_errors:
            raise worker_errors[0]

        self.assertIs(worker_graph[0].current_callable, main_callable)
        self.assertEqual(worker_graph[0](["main"]), ("main", ["main"]))

    def test_current_callable_keeps_partition_runner_state_by_thread(self) -> None:
        class FakeRunner:
            def __init__(self, name: str) -> None:
                self.partitions = [lambda inputs: (name, inputs)]

            def recursively_apply_fns(self, fns: list[Any]) -> None:
                self.partitions = [
                    fn(partition) for fn, partition in zip(fns, self.partitions)
                ]

            def call(self, inputs: list[str]) -> Any:
                return self.partitions[0](inputs)

        def wrapper(name: str) -> Any:
            def wrap_partition(fn: Any) -> Any:
                def wrapped(inputs: list[str]) -> Any:
                    return (name, fn(inputs))

                return wrapped

            return wrap_partition

        cache_key = "same-partitioned-fx-graph-cache-key"
        main_runner = FakeRunner("main")
        main_graph = self._make_graph(main_runner.call, cache_key)
        main_graph.compiled_fn_runner = main_runner
        main_graph.recursively_apply_fns = main_runner.recursively_apply_fns
        main_graph.recursively_apply_fns([wrapper("main-wrapper")])
        self.assertEqual(
            main_graph(["main"]),
            ("main-wrapper", ("main", ["main"])),
        )

        worker_ready = threading.Event()
        worker_graph: list[CompiledFxGraph] = []
        worker_errors: list[BaseException] = []

        def worker_fn() -> None:
            try:
                worker_runner = FakeRunner("worker")
                graph = self._make_graph(worker_runner.call, cache_key)
                graph.compiled_fn_runner = worker_runner
                graph.recursively_apply_fns = worker_runner.recursively_apply_fns
                graph.recursively_apply_fns([wrapper("worker-wrapper")])
                self.assertEqual(
                    graph(["worker"]),
                    ("worker-wrapper", ("worker", ["worker"])),
                )
                worker_graph.append(graph)
                worker_ready.set()
            except BaseException as e:
                worker_errors.append(e)

        worker = threading.Thread(target=worker_fn)
        worker.start()
        self.assertTrue(worker_ready.wait(5))
        worker.join(5)
        self.assertFalse(worker.is_alive())
        if worker_errors:
            raise worker_errors[0]

        self.assertEqual(
            worker_graph[0](["main"]),
            ("main-wrapper", ("main", ["main"])),
        )

    def test_current_callable_registry_is_weak_without_lookup(self) -> None:
        class FakeRunner:
            def __init__(self, name: str) -> None:
                self.partitions = [lambda inputs: (name, inputs)]

            def call(self, inputs: list[str]) -> Any:
                return self.partitions[0](inputs)

        cache_key = "deleted-partitioned-fx-graph-cache-key"

        main_runner = FakeRunner("main")
        main_graph = self._make_graph(main_runner.call, cache_key)
        main_graph.compiled_fn_runner = main_runner
        self.assertEqual(main_graph(["main"]), ("main", ["main"]))
        main_graph_ref = weakref.ref(main_graph)

        worker_ready = threading.Event()
        worker_graph: list[CompiledFxGraph] = []
        worker_errors: list[BaseException] = []

        def worker_fn() -> None:
            try:
                worker_runner = FakeRunner("worker")
                graph = self._make_graph(worker_runner.call, cache_key)
                graph.compiled_fn_runner = worker_runner
                self.assertEqual(graph(["worker"]), ("worker", ["worker"]))
                worker_graph.append(graph)
                worker_ready.set()
            except BaseException as e:
                worker_errors.append(e)

        worker = threading.Thread(target=worker_fn)
        worker.start()
        self.assertTrue(worker_ready.wait(5))
        worker.join(5)
        self.assertFalse(worker.is_alive())
        if worker_errors:
            raise worker_errors[0]

        del main_graph
        del main_runner
        gc.collect()
        self.assertIsNone(main_graph_ref())

        self.assertEqual(worker_graph[0](["main"]), ("worker", ["main"]))

    def test_current_callable_registry_pins_owner_after_lookup(self) -> None:
        class FakeRunner:
            def __init__(self, name: str) -> None:
                self.partitions = [lambda inputs: (name, inputs)]

            def call(self, inputs: list[str]) -> Any:
                return self.partitions[0](inputs)

        cache_key = "pinned-partitioned-fx-graph-cache-key"

        main_runner = FakeRunner("main")
        main_graph = self._make_graph(main_runner.call, cache_key)
        main_graph.compiled_fn_runner = main_runner
        self.assertEqual(main_graph(["main"]), ("main", ["main"]))
        main_graph_ref = weakref.ref(main_graph)

        worker_ready = threading.Event()
        worker_graph: list[CompiledFxGraph] = []
        worker_errors: list[BaseException] = []

        def worker_fn() -> None:
            try:
                worker_runner = FakeRunner("worker")
                graph = self._make_graph(worker_runner.call, cache_key)
                graph.compiled_fn_runner = worker_runner
                self.assertEqual(graph(["worker"]), ("worker", ["worker"]))
                worker_graph.append(graph)
                worker_ready.set()
            except BaseException as e:
                worker_errors.append(e)

        worker = threading.Thread(target=worker_fn)
        worker.start()
        self.assertTrue(worker_ready.wait(5))
        worker.join(5)
        self.assertFalse(worker.is_alive())
        if worker_errors:
            raise worker_errors[0]

        call = worker_graph[0].current_callable
        self.assertEqual(call(["before-delete"]), ("main", ["before-delete"]))

        del main_graph
        del main_runner
        gc.collect()
        self.assertIsNotNone(main_graph_ref())
        self.assertEqual(call(["after-lookup"]), ("main", ["after-lookup"]))

        worker_graph[0].current_callable = None
        del call
        gc.collect()
        self.assertIsNone(main_graph_ref())

    def test_prepare_for_serialization_drops_current_callable_state(self) -> None:
        def current_callable(inputs: list[str]) -> list[str]:
            return inputs

        graph = self._make_graph(current_callable)
        graph.prepare_for_serialization()

        loaded = pickle.loads(pickle.dumps(graph))
        self.assertIsNone(loaded.current_callable)

    def test_prepare_for_serialization_with_cache_key_does_not_resurrect_callable(
        self,
    ) -> None:
        def current_callable(inputs: list[str]) -> list[str]:
            return inputs

        graph = self._make_graph(current_callable, "serialization-cache-key")
        self.assertIs(graph.current_callable, current_callable)

        graph.prepare_for_serialization()

        self.assertIsNone(graph.current_callable)
        loaded = pickle.loads(pickle.dumps(graph))
        self.assertIsNone(loaded.current_callable)

    def test_deepcopy_preserves_current_callable(self) -> None:
        def original_callable(inputs: list[str]) -> tuple[str, list[str]]:
            return ("original", inputs)

        def wrapped_callable(inputs: list[str]) -> tuple[str, list[str]]:
            return ("wrapped", inputs)

        graph = self._make_graph(original_callable, "deepcopy-cache-key")
        graph.current_callable = wrapped_callable

        graph_copy = copy.deepcopy(graph)
        self.assertEqual(graph_copy(["copy"]), ("wrapped", ["copy"]))
        self.assertIsNotNone(graph_copy.current_callable)

    @unittest.skipIf(not HAS_CUDA_AND_TRITON, "requires CUDA and Triton")
    def test_current_callable_stable_in_multithreaded_torch_compile(self) -> None:
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        original_call = CompiledFxGraph.__call__
        self.addCleanup(setattr, CompiledFxGraph, "__call__", original_call)

        seen: list[tuple[int, int, int | None, tuple[int, ...] | None]] = []
        seen_lock = threading.Lock()

        def traced_call(graph: CompiledFxGraph, inputs: list[Any]) -> Any:
            current_callable = graph.current_callable
            runner = getattr(current_callable, "__self__", None)
            partitions = getattr(runner, "partitions", None)
            with seen_lock:
                seen.append(
                    (
                        threading.get_native_id(),
                        id(current_callable),
                        id(runner) if runner is not None else None,
                        (
                            tuple(id(partition) for partition in partitions)
                            if partitions is not None
                            else None
                        ),
                    )
                )
            return original_call(graph, inputs)

        CompiledFxGraph.__call__ = traced_call

        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        compiled_foo = torch.compile(foo, options={"triton.cudagraphs": True})

        start = threading.Barrier(2)
        first_call_done = threading.Barrier(2)
        thread_errors: list[BaseException] = []

        def worker_fn(worker_id: int) -> None:
            try:
                x = torch.ones(1, device="cuda")
                y = torch.ones(1, device="cuda")
                start.wait(5)
                for i in range(10):
                    result = compiled_foo(x, y)
                    result.cpu()
                    if i == 0:
                        first_call_done.wait(5)
                        if worker_id == 0:
                            time.sleep(0.1)
            except BaseException as e:
                thread_errors.append(e)

        threads = [threading.Thread(target=worker_fn, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
            self.assertFalse(thread.is_alive())
        if thread_errors:
            raise thread_errors[0]

        by_thread: dict[int, set[tuple[int, int | None, tuple[int, ...] | None]]] = {}
        for thread_id, callable_id, runner_id, partition_ids in seen:
            by_thread.setdefault(thread_id, set()).add(
                (callable_id, runner_id, partition_ids)
            )

        self.assertEqual(len(by_thread), 2)
        self.assertTrue(
            all(len(callables) == 1 for callables in by_thread.values()),
            by_thread,
        )


if __name__ == "__main__":
    run_tests()
