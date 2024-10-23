from __future__ import annotations

import os
import tempfile
import textwrap
from functools import lru_cache
from typing import Any, List


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
    def operator_str(target: Any, args: List[Any], kwargs: dict[str, Any]) -> str:
        lines = [f"target: {target}"] + [
            f"args[{i}]: {arg}" for i, arg in enumerate(args)
        ]
        if kwargs:
            lines.append(f"kwargs: {kwargs}")
        return textwrap.indent("\n".join(lines), "  ")


class MissingOperatorWithoutDecomp(OperatorIssue):
    def __init__(self, target: Any, args: List[Any], kwargs: dict[str, Any]) -> None:
        _record_missing_op(target)
        super().__init__(f"missing lowering\n{self.operator_str(target, args, kwargs)}")


class MissingOperatorWithDecomp(OperatorIssue):
    def __init__(self, target: Any, args: List[Any], kwargs: dict[str, Any]) -> None:
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
        self, exc: Exception, target: Any, args: List[Any], kwargs: dict[str, Any]
    ) -> None:
        super().__init__(
            f"{type(exc).__name__}: {exc}\n{self.operator_str(target, args, kwargs)}"
        )


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


class CUDACompileError(CppCompileError):
    pass
