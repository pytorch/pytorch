from __future__ import annotations

import os
import tempfile
import textwrap
from functools import lru_cache
from typing import Any, Optional, TYPE_CHECKING

from torch._dynamo.exc import BackendCompilerFailed, ShortenTraceback


if TYPE_CHECKING:
    import types

    from torch.cuda import _CudaDeviceProperties

if os.environ.get("TORCHINDUCTOR_WRITE_MISSING_OPS") == "1":

    @lru_cache(None)
    def _record_missing_op(target: Any) -> None:
        with open(f"{tempfile.gettempdir()}/missing_ops.txt", "a") as fd:
            fd.write(str(target) + "\n")

else:

    def _record_missing_op(target: Any) -> None:  # type: ignore[misc]
        pass


class OperatorIssue(RuntimeError):
    @staticmethod
    def operator_str(target: Any, args: list[Any], kwargs: dict[str, Any]) -> str:
        lines = [f"target: {target}"] + [
            f"args[{i}]: {arg}" for i, arg in enumerate(args)
        ]
        if kwargs:
            lines.append(f"kwargs: {kwargs}")
        return textwrap.indent("\n".join(lines), "  ")


class MissingOperatorWithoutDecomp(OperatorIssue):
    def __init__(self, target: Any, args: list[Any], kwargs: dict[str, Any]) -> None:
        _record_missing_op(target)
        super().__init__(f"missing lowering\n{self.operator_str(target, args, kwargs)}")


class MissingOperatorWithDecomp(OperatorIssue):
    def __init__(self, target: Any, args: list[Any], kwargs: dict[str, Any]) -> None:
        _record_missing_op(target)
        super().__init__(
            f"missing decomposition\n{self.operator_str(target, args, kwargs)}"
            + textwrap.dedent(
                f"""

                There is a decomposition available for {target} in
                torch._decomp.get_decompositions().  Please add this operator to the
                `decompositions` list in torch._inductor.decomposition
                """
            )
        )


class LoweringException(OperatorIssue):
    def __init__(
        self,
        exc: Exception,
        target: Any,
        args: list[Any],
        kwargs: dict[str, Any],
        stack_trace: Optional[str] = None,
    ) -> None:
        msg = f"{type(exc).__name__}: {exc}\n{self.operator_str(target, args, kwargs)}"
        if stack_trace:
            msg += f"{msg}\nFound from : \n {stack_trace}"
        super().__init__(msg)


class SubgraphLoweringException(RuntimeError):
    pass


class InvalidCxxCompiler(RuntimeError):
    def __init__(self) -> None:
        from . import config

        super().__init__(
            f"No working C++ compiler found in {config.__name__}.cpp.cxx: {config.cpp.cxx}"
        )


class CppWrapperCodegenError(RuntimeError):
    def __init__(self, msg: str) -> None:
        super().__init__(f"C++ wrapper codegen error: {msg}")


class CppCompileError(RuntimeError):
    def __init__(self, cmd: list[str], output: str) -> None:
        if isinstance(output, bytes):
            output = output.decode("utf-8")

        self.cmd = cmd
        self.output = output

        super().__init__(
            textwrap.dedent(
                """
                    C++ compile error

                    Command:
                    {cmd}

                    Output:
                    {output}
                """
            )
            .strip()
            .format(cmd=" ".join(cmd), output=output)
        )

    def __reduce__(self) -> tuple[type, tuple[list[str], str]]:
        return (self.__class__, (self.cmd, self.output))


class CUDACompileError(CppCompileError):
    pass


class TritonMissing(ShortenTraceback):
    def __init__(self, first_useful_frame: Optional[types.FrameType]) -> None:
        super().__init__(
            "Cannot find a working triton installation. "
            "Either the package is not installed or it is too old. "
            "More information on installing Triton can be found at: https://github.com/triton-lang/triton",
            first_useful_frame=first_useful_frame,
        )


class GPUTooOldForTriton(ShortenTraceback):
    def __init__(
        self,
        # pyrefly: ignore [not-a-type]
        device_props: _CudaDeviceProperties,
        first_useful_frame: Optional[types.FrameType],
    ) -> None:
        super().__init__(
            f"Found {device_props.name} which is too old to be supported by the triton GPU compiler, "
            "which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, "
            f"but your device is of CUDA capability {device_props.major}.{device_props.minor}",
            first_useful_frame=first_useful_frame,
        )


class InductorError(BackendCompilerFailed):
    backend_name = "inductor"

    def __init__(
        self,
        inner_exception: Exception,
        first_useful_frame: Optional[types.FrameType],
    ) -> None:
        self.inner_exception = inner_exception
        ShortenTraceback.__init__(
            self,
            f"{type(inner_exception).__name__}: {inner_exception}",
            first_useful_frame=first_useful_frame,
        )
