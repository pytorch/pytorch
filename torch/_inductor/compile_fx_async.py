from __future__ import annotations

from collections import namedtuple
from typing import Any, Callable, Optional, TYPE_CHECKING
from typing_extensions import final, override

import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._inductor.output_code import CompiledFxGraphConstants, OutputCode

from .compile_fx import _CompileFxKwargs, _InProcessFxCompile, FxCompile
from .compile_fx_ext import _OutOfProcessFxCompile, _WireProtocolPickledOutput
from .output_code import complex_memory_overlap as complex_memory_overlap  # noqa: F401


if TYPE_CHECKING:
    from collections.abc import Sequence
    from concurrent.futures import Future

    from torch._inductor.utils import BoxedBool, InputType
    from torch.fx import GraphModule


_PostCompileData = namedtuple(
    "_PostCompileData", ("example_inputs", "cudagraphs", "constants")
)


@final
class _AsyncOutputCode(OutputCode):
    _eager_forward: Optional[Callable[..., Any]]
    _output_code: Optional[OutputCode]
    _future: Optional[Future[_WireProtocolPickledOutput]]
    _callback: Callable[[_WireProtocolPickledOutput], OutputCode]
    _post_compile_data: Optional[_PostCompileData] = None

    def __init__(
        self,
        eager_forward: Callable[..., Any],
        future: Future[_WireProtocolPickledOutput],
        callback: Callable[[_WireProtocolPickledOutput], OutputCode],
    ) -> None:
        self._eager_forward = eager_forward
        self._output_code = None

        self._future = future
        self._callback = callback

        # This tells callers to call us with `inputs` instead of `*inputs`.
        self._boxed_call = True

    @override
    def __call__(self, inputs: Sequence[Any]) -> Any:
        if self._future is not None and self._future.done():
            self._switch_to_compiled_forward()

        if eager_forward := self._eager_forward:
            _AsyncFxCompile._stat_eager_runs += 1
            return eager_forward(inputs)

        else:
            _AsyncFxCompile._stat_oop_runs += 1
            assert self._output_code is not None
            if getattr(self._output_code, "_boxed_call", False):
                return self._output_code.__call__(inputs)
            else:
                return self._output_code.__call__(*inputs)

    def _switch_to_compiled_forward(self) -> None:
        assert self._future is not None

        # TODO: If the future ended in an exception do we want to continue
        # running eager or hit the exception now?
        f, self._future = self._future, None
        output_code = self._callback(f.result())

        if pcd := self._post_compile_data:
            self._post_compile_data = None
            output_code.post_compile(pcd.example_inputs, pcd.cudagraphs, pcd.constants)

        self._output_code = output_code
        self._eager_forward = None

    @override
    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        cudagraphs: BoxedBool,
        constants: CompiledFxGraphConstants,
    ) -> None:
        if self._eager_forward is not None:
            self._post_compile_data = _PostCompileData(
                example_inputs, cudagraphs, constants
            )
        else:
            assert self._output_code is not None
            self._output_code.post_compile(example_inputs, cudagraphs, constants)


@final
class _AsyncFxCompile(FxCompile):
    _compile: _OutOfProcessFxCompile

    # Some debugging stats:
    # Number of times we started a background compile.
    _stat_bg_started: int = 0
    # Number of times we finished a background compile.
    _stat_bg_finished: int = 0
    # Number of times we ran "eager"
    _stat_eager_runs: int = 0
    # Number of times we ran our compiled (out-of-process) artifact
    _stat_oop_runs: int = 0

    def __init__(self, compile: FxCompile) -> None:
        assert isinstance(compile, _OutOfProcessFxCompile)
        self._compile = compile

    @classmethod
    def _reset_stats(cls) -> None:
        cls._stat_bg_started = 0
        cls._stat_bg_finished = 0
        cls._stat_eager_runs = 0
        cls._stat_oop_runs = 0

    @override
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode:
        eager_output_code = _InProcessFxCompile().codegen_and_compile(
            gm, example_inputs, inputs_to_check, graph_kwargs
        )

        # This is similar to _SerializedFxCompile.codegen_and_compile() but
        # handles the async routing.

        serialized = self._compile.serialize_compile(
            gm, example_inputs, inputs_to_check, graph_kwargs
        )
        if not serialized:
            # We can't serialize - just return the eager OutputCode
            return eager_output_code

        inputs, constants = serialized

        _AsyncFxCompile._stat_bg_started += 1
        f = self._compile._send_to_child_async(inputs)

        # This is called by _switch_to_compiled_forward() when f has a result...
        def callback(pickled_output: _WireProtocolPickledOutput) -> OutputCode:
            _AsyncFxCompile._stat_bg_finished += 1
            output = pickled_output.deserialize(constants)
            self._compile._postprocess(output)
            return output.graph

        return _AsyncOutputCode(eager_output_code, f, callback)
