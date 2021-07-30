import abc
import ast
import functools
import inspect
import marshal
import textwrap
import timeit
import typing

from torch.utils.benchmark._impl.workers import base


class TaskBase(abc.ABC):

    @abc.abstractproperty
    def worker(self) -> base.WorkerBase:
        ...


# TODO: This needs to be factored into components that can be unit tested.
def run_in_worker(scoped: bool = True):
    """Decorator to run Task method in worker rather than the caller.

    The Worker + Task model dictates that the caller generates a string of
    Python source code. However this is not particularly ergonomic; there is
    significant tooling (syntax highlighting, type checking, etc.) which is
    lost if code must be provided as a string literal.

    Moreover, moving values from the caller to the worker can be tedious.
    Simply templating them into a string literal is brittle (because __repr__
    may not produce valid source) and may subtly alter the value (e.g. the
    string representation of a float will not produce the same number as the
    original value). `WorkerBase.store` will safely move values, but does not
    alleviate the ergonomic issues.

    Consider the following, where we want the worker to open a file, read up to
    `n` lines, and then return them to the caller. One implementation would be:

    ```
    def head(self, fpath: str, n: int) -> List[str]:
        self.worker.store("fpath", fpath)
        self.worker.store("n", n)
        self.worker.run(textwrap.dedent('''
            lines = []
            with open(fpath, "rt") as f:
                for i, l in enumerate(f):
                    if i == n:
                        break
                    lines.append(l)
        '''))
        return self.worker.load("lines")
    ```

    It works, but it's not very easy to read and leaks lots of variables
    (fpath, n, lines, f, etc.) into the worker's global namespace. This
    decorator allows the following code to be written instead:

    ```
    @run_in_worker(scoped=True)
    def head(fpath: str, n: int) -> List[str]:
        lines = []
        with open(fpath, "rt") as f:
            for i, l in enumerate(f):
                if i == n:
                    break
                lines.append(l)
        return lines
    ```

    Code in the main thread can call `head` just like any normal function, but
    it is executed in the worker. And unlike the first example, we will not
    pollute the global namespace. (This is because `scoped=True`) There are
    three aspects to `run_in_worker`:

        1) Serialize arguments and revive them in the worker.
        2) Extract the function body.
        3) Retrieve the result from the worker.

    All three are entirely mechanical; `run_in_worker` uses Python AST rather
    than raw string parsing, so it is quite robust. Because ambiguity would be
    very difficult to diagnose in this context, `run_in_worker` requires that
    a complete type annotated signature be provided and that there are no
    variadic arguments. (*args or **kwargs) Moreover, it has same restriction
    for inputs and outputs as `store` and `load`: the values must be
    serializable by the `marshal` library. (i.e. basic Python types)
    """

    def outer(f):
        signature = inspect.signature(f)
        for arg, arg_parameter in signature.parameters.items():
            if arg_parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                raise ValueError(
                    f"Variadic positional argument `*{arg}` not permitted "
                    "for `run_in_worker` function.")
            if arg_parameter.kind == inspect.Parameter.VAR_KEYWORD:
                raise ValueError(
                    f"Variadic keywork argument `**{arg}` not permitted "
                    "for `run_in_worker` function.")
            if arg_parameter.annotation == inspect.Parameter.empty:
                raise ValueError(f"Missing type annotation for parameter `{arg}`")

        if signature.return_annotation == inspect.Parameter.empty:
            raise ValueError("Missing return annotation.")

        has_return_value = (signature.return_annotation is not None)
        if has_return_value and not scoped:
            raise ValueError(
                "Unscoped (globally executed) call can not have a return value.")

        f_src = textwrap.dedent(inspect.getsource(f))
        f_ast = ast.parse(f_src)
        assert len(f_ast.body) == 1
        assert isinstance(f_ast.body[0], ast.FunctionDef)
        assert f_ast.body[0].body

        # For some reason ast one indexes lineno.
        src_lines = f_src.splitlines(keepends=False)
        raw_body_lines = src_lines[f_ast.body[0].body[0].lineno - 1:]
        col_offset = f_ast.body[0].body[0].col_offset

        body_lines: typing.List[str] = []
        for i, l in enumerate(raw_body_lines):
            prefix, suffix = l[:col_offset], l[col_offset:]

            # The first line of the body may overlap with the signature.
            #   e.g. `def f(): pass`
            # For all other lines, the prefix must only be indentation.
            assert not i or not prefix.strip()

            body_lines.append(suffix)

        @functools.wraps(f)
        def inner(self: TaskBase, *args, **kwargs) -> None:
            bound_signature = signature.bind(*args, **kwargs)
            bound_signature.apply_defaults()

            body: typing.List[str] = ["# Deserialize args", "import marshal"]
            for arg_name, arg_value in bound_signature.arguments.items():
                try:
                    arg_bytes = marshal.dumps(arg_value)
                except ValueError:
                    raise ValueError(f"unmarshallable arg {arg_name}: {arg_value}")

                body.append(f"{arg_name} = marshal.loads(bytes.fromhex({repr(arg_bytes.hex())}))  # {arg_value}")
            body.extend(["", "# Wrapped source"] + body_lines)

            src = "\n".join([
                "def _run_in_worker_f():",
                textwrap.indent("\n".join(body), " " * 4),
                "",
                "_run_in_worker_result = _run_in_worker_f()",
            ])

            # `worker.load` is not free, so for void functions we skip it.
            if has_return_value:
                self.worker.run(src)
                return self.worker.load("_run_in_worker_result")

            else:
                src = f"{src}\nassert _run_in_worker_result is None"
                self.worker.run(src)

        return inner
    return outer
