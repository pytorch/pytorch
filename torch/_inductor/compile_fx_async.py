from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING
from typing_extensions import final, override

import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._inductor.output_code import CompiledFxGraphConstants, OutputCode

from .compile_fx import _CompileFxKwargs, _InProcessFxCompile, FxCompile
from .output_code import complex_memory_overlap as complex_memory_overlap  # noqa: F401


# When async compile works with cache, remove the disabling below
BUG_CACHES_DONT_WORK_WITH_ASYNC = True


if TYPE_CHECKING:
    from collections.abc import Sequence
    from concurrent.futures import Future

    from torch._inductor.utils import InputType
    from torch.fx import GraphModule

    from .compile_fx_ext import _OutOfProcessFxCompile, _WireProtocolPickledOutput


@dataclass
class _PostCompileData:
    example_inputs: Sequence[InputType]
    constants: CompiledFxGraphConstants
    graph_kwargs: _CompileFxKwargs


@dataclass
class ProgressiveCompilationState:
    progression_futures: deque[Future[_WireProtocolPickledOutput]]
    callback: Callable[[_WireProtocolPickledOutput], OutputCode]
    post_compile_data: Optional[_PostCompileData]

    def check_and_get_ready_stage(self) -> int:
        """Check if any progression stage is ready and return its index, or -1 if none are ready."""
        if not self.progression_futures:
            return -1

        stage_index = -1
        if self.post_compile_data:
            for i, future in enumerate(self.progression_futures):
                if future.done():
                    stage_index = i

        return stage_index

    def switch_to_progression_stage(self, stage_index: int) -> tuple[OutputCode, bool]:
        """
        Switch to the specified progression stage and return the optimized output code.
        Returns a tuple of (optimized_output_code, should_clear_compilation_state).
        """
        future = self.progression_futures[stage_index]
        assert future is not None
        optimized_output_code = self.callback(future.result())

        if pcd := self.post_compile_data:
            optimized_output_code.post_compile(
                pcd.example_inputs, pcd.constants, pcd.graph_kwargs
            )

        # Clear earlier progression futures to free memory
        for _ in range(stage_index + 1):
            self.progression_futures.popleft()

        # Return whether all compilation state should be cleared
        should_clear_state = not self.progression_futures
        return optimized_output_code, should_clear_state


# _ProgressiveOutputCode handles running a fast compile first, then hot-swapping
# to a more optimized version when the expensive compile finishes.
@final
class _ProgressiveOutputCode(OutputCode):
    _fast_output_code: Optional[OutputCode]
    _optimized_output_code: Optional[OutputCode]
    _compilation_state: Optional[ProgressiveCompilationState]
    # _boxed_call state is effectively cached (we sometimes wrap unboxed w/
    # lambdas to box them) so we can't change it mid-way. Since _boxed_call=True
    # is more common let's default to that and we'll convert if necessary.
    _boxed_call: bool = True

    def __init__(
        self,
        # Fast compile that runs faster than the progressive compiles
        fast_output_code: OutputCode,
        # Futures for the progressive optimized compiles
        progression_futures: Sequence[Future[_WireProtocolPickledOutput]],
        # Callback to convert the optimized result to OutputCode
        callback: Callable[[_WireProtocolPickledOutput], OutputCode],
    ) -> None:
        self._fast_output_code = fast_output_code
        self._optimized_output_code = None
        self._compilation_state = ProgressiveCompilationState(
            progression_futures=deque(progression_futures),
            callback=callback,
            post_compile_data=None,
        )

    @override
    def __call__(self, args: Sequence[Any]) -> Any:
        # Check if any newer progression stage is ready and switch to it
        self._check_and_switch_progression()

        if self._optimized_output_code is not None:
            # Use legacy async stats if this was created in async mode
            # We can detect this by checking if the fast_output_code was from _InProcessFxCompile
            if hasattr(self._fast_output_code, '__class__') and self._fast_output_code.__class__.__name__ == '_InProcessFxCompile':
                _ProgressiveFxCompile._stat_compiled_runs += 1
            else:
                _ProgressiveFxCompile._stat_optimized_runs += 1
            output_code = self._optimized_output_code
        else:
            # Use legacy async stats if this was created in async mode
            if hasattr(self._fast_output_code, '__class__') and self._fast_output_code.__class__.__name__ == '_InProcessFxCompile':
                _ProgressiveFxCompile._stat_eager_runs += 1
            else:
                _ProgressiveFxCompile._stat_fast_runs += 1
            assert self._fast_output_code is not None
            output_code = self._fast_output_code

        boxed_call = getattr(output_code, "_boxed_call", False)
        if boxed_call:
            res = output_code.__call__(args)
        else:
            res = output_code.__call__(*args)
        return res

    def _check_and_switch_progression(self) -> None:
        if not self._compilation_state:
            return

        stage_index = self._compilation_state.check_and_get_ready_stage()
        if stage_index == -1:
            # no futures are ready
            return

        self._switch_to_progression_stage(stage_index)

    def _switch_to_progression_stage(self, stage_index: int) -> None:
        assert self._compilation_state is not None
        optimized_output_code, should_clear_state = (
            self._compilation_state.switch_to_progression_stage(stage_index)
        )

        self._optimized_output_code = optimized_output_code
        self._fast_output_code = None

        # Clear all compilation state if no more progression futures are left
        if should_clear_state:
            self._compilation_state = None

    @override
    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        assert self._fast_output_code is not None
        self._fast_output_code.post_compile(example_inputs, constants, graph_kwargs)

        assert self._compilation_state is not None
        # Store for later when optimized version is ready
        self._compilation_state.post_compile_data = _PostCompileData(
            example_inputs, constants, graph_kwargs
        )


# _ProgressiveFxCompile runs a fast compile immediately, then kicks off
# progressive compiles in the background and hot-swaps when they're ready.
# It also handles the async case (single future with eager execution).
@final
class _ProgressiveFxCompile(FxCompile):
    _fast_compile: Optional[FxCompile]
    _optimized_compile: _OutOfProcessFxCompile
    _progression_configs: list[dict[str, Any]]
    _is_async_mode: bool

    # Debugging stats
    _stat_bg_started: int = 0
    _stat_bg_finished: int = 0
    _stat_fast_runs: int = 0
    _stat_optimized_runs: int = 0
    # Legacy async stats (for backward compatibility)
    _stat_eager_runs: int = 0
    _stat_compiled_runs: int = 0

    def __init__(
        self,
        optimized_compile: _OutOfProcessFxCompile,
        fast_compile: Optional[FxCompile] = None,
        progression_configs: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self._optimized_compile = optimized_compile
        self._fast_compile = fast_compile
        self._progression_configs = progression_configs or [{}]
        # If no fast_compile is provided, we're in async mode (eager execution)
        self._is_async_mode = fast_compile is None

    @classmethod
    def _reset_stats(cls) -> None:
        cls._stat_bg_started = 0
        cls._stat_bg_finished = 0
        cls._stat_fast_runs = 0
        cls._stat_optimized_runs = 0
        cls._stat_eager_runs = 0
        cls._stat_compiled_runs = 0

    @classmethod
    def create_async(cls, optimized_compile: _OutOfProcessFxCompile) -> _ProgressiveFxCompile:
        """Create an async compile instance (legacy _AsyncFxCompile behavior)."""
        return cls(optimized_compile=optimized_compile, fast_compile=None, progression_configs=[{}])

    @classmethod
    def create_progressive(
        cls,
        fast_compile: FxCompile,
        optimized_compile: _OutOfProcessFxCompile,
        progression_configs: list[dict[str, Any]],
    ) -> _ProgressiveFxCompile:
        """Create a progressive compile instance."""
        return cls(
            optimized_compile=optimized_compile,
            fast_compile=fast_compile,
            progression_configs=progression_configs,
        )

    @override
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode:
        import torch._inductor.config as inductor_config

        progression_futures: list[Future[_WireProtocolPickledOutput]] = []

        for config in self._progression_configs:
            with inductor_config.patch(config):
                _ProgressiveFxCompile._stat_bg_started += 1

                # Start the progressive compiles in the background
                serialized = self._optimized_compile.serialize_compile(
                    gm, example_inputs, inputs_to_check, graph_kwargs
                )

                if not serialized:
                    continue

                inputs, constants = serialized
                future = self._optimized_compile._send_to_child_async(inputs)
                progression_futures.append(future)

        if self._is_async_mode:
            # Async mode: use eager execution instead of fast compile
            eager_output_code = _InProcessFxCompile().codegen_and_compile(
                gm, example_inputs, inputs_to_check, graph_kwargs
            )
            fallback_output_code = eager_output_code
        else:
            # Progressive mode: use fast compile
            assert self._fast_compile is not None
            fast_output_code = self._fast_compile.codegen_and_compile(
                gm, example_inputs, inputs_to_check, graph_kwargs
            )
            fallback_output_code = fast_output_code

        if not progression_futures:
            # All async compile attempts failed - just return the fallback version
            return fallback_output_code

        # Callback to handle the optimized result.
        # This callback may be called multiple times, once for each progressive level completed,
        # but may be skipped if a level either never completes or if a more optimal level
        # completes before a less optimal one is switched to.
        def callback(pickled_output: _WireProtocolPickledOutput) -> OutputCode:
            _ProgressiveFxCompile._stat_bg_finished += 1
            output = pickled_output.deserialize(constants)
            self._optimized_compile._postprocess(output)
            return output.graph

        return _ProgressiveOutputCode(fallback_output_code, progression_futures, callback)
