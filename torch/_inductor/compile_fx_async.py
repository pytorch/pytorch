from __future__ import annotations

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


# _ProgressiveOutputCode handles progressive compilation with configurable Stage 0.
# Stage 0 can be either eager execution or fast compile, followed by optimization stages.
@final
class _ProgressiveOutputCode(OutputCode):
    _stage0_output_code: OutputCode
    _optimized_output_code: Optional[OutputCode]
    _progression_futures: list[Optional[Future[_WireProtocolPickledOutput]]]
    _callback: Callable[[_WireProtocolPickledOutput], OutputCode]
    _post_compile_data: Optional[_PostCompileData] = None
    _current_progression_index: int

    def __init__(
        self,
        # Stage 0: either eager execution OutputCode or fast compile OutputCode
        stage0_output_code: OutputCode,
        # Futures for the progressive optimization stages (Stage 1+)
        progression_futures: list[Future[_WireProtocolPickledOutput]],
        # Callback to convert optimized results to OutputCode
        callback: Callable[[_WireProtocolPickledOutput], OutputCode],
    ) -> None:
        self._stage0_output_code = stage0_output_code
        self._progression_futures = list(progression_futures)
        self._callback = callback
        self._optimized_output_code = None
        self._current_progression_index = -1

    @override
    def __call__(self, *args: Any) -> Any:
        # Check if any newer progression stage is ready and switch to it
        self._check_and_switch_progression()

        if self._optimized_output_code is not None:
            # Use the optimized code from a completed progression stage
            _ProgressiveFxCompile._stat_optimized_runs += 1
            return self._optimized_output_code.__call__(*args)
        else:
            # Use Stage 0 (either eager execution or fast compile)
            _ProgressiveFxCompile._stat_fast_runs += 1
            return self._stage0_output_code.__call__(*args)

    def _check_and_switch_progression(self) -> None:
        # Check if any newer progression stage is ready (in order from latest to earliest)
        for i in range(
            len(self._progression_futures) - 1, self._current_progression_index, -1
        ):
            future = self._progression_futures[i]
            if future and future.done():
                self._switch_to_progression_stage(i)
                break

    def _switch_to_progression_stage(self, stage_index: int) -> None:
        future = self._progression_futures[stage_index]
        assert future is not None
        optimized_output_code = self._callback(future.result())

        if pcd := self._post_compile_data:
            # Only clear post_compile_data if this is the final progression stage
            if stage_index == len(self._progression_futures) - 1:
                self._post_compile_data = None
            optimized_output_code.post_compile(
                pcd.example_inputs, pcd.constants, pcd.graph_kwargs
            )

        self._optimized_output_code = optimized_output_code
        self._fast_output_code = None
        self._current_progression_index = stage_index

        # Clear earlier progression futures to free memory
        for i in range(stage_index):
            self._progression_futures[i] = None

    @override
    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        if self._optimized_output_code is not None:
            # We've already switched to optimized code
            self._optimized_output_code.post_compile(
                example_inputs, constants, graph_kwargs
            )
        else:
            # Still using Stage 0, call post_compile on it and store data for later
            self._stage0_output_code.post_compile(
                example_inputs, constants, graph_kwargs
            )
            self._post_compile_data = _PostCompileData(
                example_inputs, constants, graph_kwargs
            )


# _ProgressiveFxCompile runs progressive compilation with configurable Stage 0.
# Stage 0 can be either eager execution or fast compile, followed by optimization stages.
@final
class _ProgressiveFxCompile(FxCompile):
    _fast_compile: Optional[FxCompile]  # None means use eager execution for Stage 0
    _optimized_compile: _OutOfProcessFxCompile
    _progression_configs: list[dict[str, Any]]
    _use_eager_stage0: bool  # True for eager Stage 0, False for fast compile Stage 0

    # Debugging stats
    _stat_bg_started: int = 0
    _stat_bg_finished: int = 0
    _stat_fast_runs: int = 0
    _stat_optimized_runs: int = 0

    def __init__(
        self,
        fast_compile: Optional[FxCompile],
        optimized_compile: _OutOfProcessFxCompile,
        progression_configs: list[dict[str, Any]],
        use_eager_stage0: bool = False,
    ) -> None:
        self._fast_compile = fast_compile
        self._optimized_compile = optimized_compile
        self._progression_configs = progression_configs
        self._use_eager_stage0 = use_eager_stage0

    @classmethod
    def _reset_stats(cls) -> None:
        cls._stat_bg_started = 0
        cls._stat_bg_finished = 0
        cls._stat_fast_runs = 0
        cls._stat_optimized_runs = 0

    @override
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode:
        # Generate Stage 0 output code - either eager execution or fast compile
        if self._use_eager_stage0:
            # Use eager execution for Stage 0
            stage0_output_code = _InProcessFxCompile().codegen_and_compile(
                gm, example_inputs, inputs_to_check, graph_kwargs
            )
        else:
            # Use fast compile for Stage 0
            assert self._fast_compile is not None
            stage0_output_code = self._fast_compile.codegen_and_compile(
                gm, example_inputs, inputs_to_check, graph_kwargs
            )

        import torch._inductor.config as inductor_config

        progression_futures: list[Future[_WireProtocolPickledOutput]] = []

        # Start the progressive optimization stages (Stage 1+) in the background
        for config in self._progression_configs:
            with inductor_config.patch(config):
                _ProgressiveFxCompile._stat_bg_started += 1

                # Start the progressive compiles in the background
                serialized = self._optimized_compile.serialize_compile(
                    gm, example_inputs, inputs_to_check, graph_kwargs
                )

                if not serialized:
                    # Can't serialize - just return Stage 0
                    return stage0_output_code

                inputs, constants = serialized
                future = self._optimized_compile._send_to_child_async(inputs)
                progression_futures.append(future)

        if not progression_futures:
            # All progressive compile attempts failed - just return Stage 0
            return stage0_output_code

        # Callback to handle the optimized result
        def callback(pickled_output: _WireProtocolPickledOutput) -> OutputCode:
            _ProgressiveFxCompile._stat_bg_finished += 1
            output = pickled_output.deserialize(constants)
            self._optimized_compile._postprocess(output)
            return output.graph

        # Always use unified _ProgressiveOutputCode regardless of Stage 0 type
        return _ProgressiveOutputCode(
            stage0_output_code=stage0_output_code,
            progression_futures=progression_futures,
            callback=callback,
        )


# Factory functions for backward compatibility and cleaner API
def create_async_fx_compile(
    optimized_compile: _OutOfProcessFxCompile,
    progression_configs: Optional[list[dict[str, Any]]] = None,
) -> _ProgressiveFxCompile:
    """Create progressive FX compiler with eager execution as Stage 0."""
    if progression_configs is None:
        from .compile_fx import _get_progression_configs

        progression_configs = _get_progression_configs()

    return _ProgressiveFxCompile(
        fast_compile=None,
        optimized_compile=optimized_compile,
        progression_configs=progression_configs,
        use_eager_stage0=True,
    )


def create_progressive_fx_compile(
    fast_compile: FxCompile,
    optimized_compile: _OutOfProcessFxCompile,
    progression_configs: Optional[list[dict[str, Any]]] = None,
) -> _ProgressiveFxCompile:
    """Create progressive FX compiler with fast compile as Stage 0."""
    if progression_configs is None:
        from .compile_fx import _get_progression_configs

        progression_configs = _get_progression_configs()

    return _ProgressiveFxCompile(
        fast_compile=fast_compile,
        optimized_compile=optimized_compile,
        progression_configs=progression_configs,
        use_eager_stage0=False,
    )
