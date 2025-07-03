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


# _AsyncOutputCode handles the actual management of waiting for an
# out-of-process compile to finish and then switching over to it.
@final
class _AsyncOutputCode(OutputCode):
    _eager_forward: Optional[Callable[..., Any]]
    _output_code: Optional[OutputCode]
    _future: Optional[Future[_WireProtocolPickledOutput]]
    _callback: Callable[[_WireProtocolPickledOutput], OutputCode]
    _post_compile_data: Optional[_PostCompileData] = None
    _boxed_call: bool  # Copied from the forward/output_code

    def __init__(
        self,
        # eager_forward is run until the future is finished.
        eager_forward: Callable[..., Any],
        # this responds with the result of the out-of-process compile when it's
        # ready.
        future: Future[_WireProtocolPickledOutput],
        # this callback gets called to turn the _WireProtocolPickledOutput into an OutputCode
        callback: Callable[[_WireProtocolPickledOutput], OutputCode],
    ) -> None:
        self._eager_forward = eager_forward
        self._boxed_call = getattr(eager_forward, "_boxed_call", False)
        self._output_code = None

        self._future = future
        self._callback = callback

    @override
    def __call__(self, *args: Any) -> Any:
        if self._future is not None and self._future.done():
            args = self._switch_to_compiled_forward(args)

        if eager_forward := self._eager_forward:
            _ProgressiveFxCompile._stat_fast_runs += 1
            return eager_forward(*args)

        else:
            _ProgressiveFxCompile._stat_optimized_runs += 1
            assert self._output_code is not None
            return self._output_code.__call__(*args)

    # Takes and returns the args (converted to the "right" boxed mode)
    def _switch_to_compiled_forward(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        assert self._future is not None

        # TODO: If the future ended in an exception do we want to continue
        # running eager or hit the exception now?
        f, self._future = self._future, None
        output_code = self._callback(f.result())

        if pcd := self._post_compile_data:
            self._post_compile_data = None

            output_code.post_compile(
                pcd.example_inputs, pcd.constants, pcd.graph_kwargs
            )

        self._output_code = output_code
        self._eager_forward = None
        boxed_call = getattr(output_code, "_boxed_call", False)

        if self._boxed_call != boxed_call:
            if self._boxed_call:
                # Was boxed, now unboxed
                args = args[0] if len(args) > 0 else ()
            else:
                # Was unboxed, now boxed
                args = (args,)

        self._boxed_call = boxed_call
        return args

    @override
    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        if self._eager_forward is not None:
            self._post_compile_data = _PostCompileData(
                example_inputs, constants, graph_kwargs
            )
        else:
            assert self._output_code is not None
            self._output_code.post_compile(example_inputs, constants, graph_kwargs)


# _ProgressiveOutputCode handles running a fast compile first, then hot-swapping
# to a more optimized version when the expensive compile finishes.
@final
class _ProgressiveOutputCode(OutputCode):
    _fast_output_code: Optional[OutputCode]
    _optimized_output_code: Optional[OutputCode]
    _progression_futures: list[Optional[Future[_WireProtocolPickledOutput]]]
    _callback: Callable[[_WireProtocolPickledOutput], OutputCode]
    _post_compile_data: Optional[_PostCompileData] = None
    _current_progression_index: int
    # _boxed_call state is effectively cached (we sometimes wrap unboxed w/
    # lambdas to box them) so we can't change it mid-way. Since _boxed_call=True
    # is more common let's default to that and we'll convert if necessary.
    _boxed_call: bool = True

    def __init__(
        self,
        # Fast compile that runs faster than the progressive compiles
        fast_output_code: OutputCode,
        # Futures for the progressive optimized compiles
        progression_futures: list[Future[_WireProtocolPickledOutput]],
        # Callback to convert the optimized result to OutputCode
        callback: Callable[[_WireProtocolPickledOutput], OutputCode],
    ) -> None:
        self._fast_output_code = fast_output_code
        self._optimized_output_code = None
        self._progression_futures = list(progression_futures)
        self._callback = callback
        self._current_progression_index = -1

    @override
    def __call__(self, args: Sequence[Any]) -> Any:
        # Check if any newer progression stage is ready and switch to it
        self._check_and_switch_progression()

        if self._optimized_output_code is not None:
            _ProgressiveFxCompile._stat_optimized_runs += 1
            output_code = self._optimized_output_code
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
            self._optimized_output_code.post_compile(
                example_inputs, constants, graph_kwargs
            )
        elif self._fast_output_code is not None:
            self._fast_output_code.post_compile(example_inputs, constants, graph_kwargs)
            # Store for later when  optimized version is ready
            self._post_compile_data = _PostCompileData(
                example_inputs, constants, graph_kwargs
            )


# _ProgressiveFxCompile runs a fast compile immediately, then kicks off
# progressive compiles in the background and hot-swaps when they're ready.
# Can also be configured to use eager execution as the fast path (async mode).
@final
class _ProgressiveFxCompile(FxCompile):
    _fast_compile: Optional[FxCompile]  # None means use eager execution (async mode)
    _optimized_compile: _OutOfProcessFxCompile
    _progression_configs: list[dict[str, Any]]
    _use_eager_fast_path: bool  # True for async mode, False for progressive mode

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
        use_eager_fast_path: bool = False,
    ) -> None:
        self._fast_compile = fast_compile
        self._optimized_compile = optimized_compile
        self._progression_configs = progression_configs
        self._use_eager_fast_path = use_eager_fast_path

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
        # Handle fast path - either fast compile or eager execution
        if self._use_eager_fast_path:
            # Async mode: use eager execution as fast path
            fast_output_code = _InProcessFxCompile().codegen_and_compile(
                gm, example_inputs, inputs_to_check, graph_kwargs
            )
        else:
            # Progressive mode: use fast compile
            assert self._fast_compile is not None
            fast_output_code = self._fast_compile.codegen_and_compile(
                gm, example_inputs, inputs_to_check, graph_kwargs
            )

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
                    # Can't serialize - just return the fast version
                    return fast_output_code

                inputs, constants = serialized
                future = self._optimized_compile._send_to_child_async(inputs)
                progression_futures.append(future)

        if not progression_futures:
            # All async compile attempts failed - just return the fast version
            return fast_output_code

        # Callback to handle the optimized result
        def callback(pickled_output: _WireProtocolPickledOutput) -> OutputCode:
            _ProgressiveFxCompile._stat_bg_finished += 1
            output = pickled_output.deserialize(constants)
            self._optimized_compile._postprocess(output)
            return output.graph

        if self._use_eager_fast_path:
            # For async mode, return _AsyncOutputCode for backward compatibility
            return _AsyncOutputCode(fast_output_code, progression_futures[0], callback)
        else:
            # For progressive mode, return _ProgressiveOutputCode
            return _ProgressiveOutputCode(
                fast_output_code, progression_futures, callback
            )


# Factory functions for backward compatibility and cleaner API
def create_async_fx_compile(
    optimized_compile: _OutOfProcessFxCompile,
    progression_configs: Optional[list[dict[str, Any]]] = None,
) -> _ProgressiveFxCompile:
    """Create async FX compiler (eager fast path) using progressive infrastructure."""
    if progression_configs is None:
        from .compile_fx import _get_progression_configs

        progression_configs = _get_progression_configs()

    return _ProgressiveFxCompile(
        fast_compile=None,
        optimized_compile=optimized_compile,
        progression_configs=progression_configs,
        use_eager_fast_path=True,
    )


def create_progressive_fx_compile(
    fast_compile: FxCompile,
    optimized_compile: _OutOfProcessFxCompile,
    progression_configs: Optional[list[dict[str, Any]]] = None,
) -> _ProgressiveFxCompile:
    """Create progressive FX compiler (fast compile fast path)."""
    if progression_configs is None:
        from .compile_fx import _get_progression_configs

        progression_configs = _get_progression_configs()

    return _ProgressiveFxCompile(
        fast_compile=fast_compile,
        optimized_compile=optimized_compile,
        progression_configs=progression_configs,
        use_eager_fast_path=False,
    )
