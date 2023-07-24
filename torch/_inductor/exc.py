import os
import tempfile
import textwrap
from functools import lru_cache

if os.environ.get("TORCHINDUCTOR_WRITE_MISSING_OPS") == "1":

    @lru_cache(None)
    def _record_missing_op(target):
        with open(f"{tempfile.gettempdir()}/missing_ops.txt", "a") as fd:
            fd.write(str(target) + "\n")

else:

    def _record_missing_op(target):
        pass


class OperatorIssue(RuntimeError):
    @staticmethod
    def operator_str(target, args, kwargs):
        lines = [f"target: {target}"]
        if not getattr(target, "is_opaque", False):
            lines += [f"args[{i}]: {arg}" for i, arg in enumerate(args)]
        if kwargs:
            lines.append(f"kwargs: {kwargs}")
        return textwrap.indent("\n".join(lines), "  ")


class MissingOperatorWithoutDecomp(OperatorIssue):
    def __init__(self, target, args, kwargs):
        _record_missing_op(target)
        super().__init__(f"missing lowering\n{self.operator_str(target, args, kwargs)}")


class MissingOperatorWithDecomp(OperatorIssue):
    def __init__(self, target, args, kwargs):
        _record_missing_op(target)
        super().__init__(
            f"missing decomposition\n{self.operator_str(target, args, kwargs)}"
            + textwrap.dedent(
                f"""

                There is a decomposition available for {target} in
                torch._decomp.get_decompositions().  Please add this operator to the
                `decompositions` list in torch._inductor.decompositions
                """
            )
        )


class LoweringException(OperatorIssue):
    def __init__(self, exc, target, args, kwargs):
        super().__init__(
            f"{type(exc).__name__}: {exc}\n{self.operator_str(target, args, kwargs)}"
        )


class InvalidCxxCompiler(RuntimeError):
    def __init__(self):
        from . import config

        super().__init__(
            f"No working C++ compiler found in {config.__name__}.cpp.cxx: {config.cpp.cxx}"
        )


class CppCompileError(RuntimeError):
    def __init__(self, cmd, output):
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
