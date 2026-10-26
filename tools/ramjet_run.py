from __future__ import annotations

import ast
import importlib.util
import json
import os
import runpy
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager, ExitStack
from pathlib import Path


# Match process isolation in test/run_test.py's CUSTOM_HANDLERS.
ISOLATED_TEST_FILES = frozenset(
    {
        "test/test_cuda_primary_ctx.py",
        "test/test_cuda_nvml_based_avail.py",
        "test/test_gpu_trace.py",
        "test/distributed/test_c10d_nccl.py",
        "test/distributed/test_c10d_gloo.py",
        "test/distributed/test_c10d_ucc.py",
        "test/distributed/test_c10d_common.py",
        "test/distributed/test_c10d_spawn_gloo.py",
        "test/distributed/test_c10d_spawn_ucc.py",
        "test/distributed/test_pg_wrapper.py",
        "test/distributed/rpc/test_share_memory.py",
        "test/functorch/test_control_flow_cuda_initialization.py",
    }
)


class CpuRunnerGuard(ast.NodeTransformer):
    def __init__(self):
        self.triton = importlib.util.find_spec("triton") is not None

    def visit_Name(self, node):
        if node.id in {"HAS_GPU", "TEST_CUDA", "RUN_GPU"}:
            return ast.copy_location(ast.Constant(True), node)
        if node.id in {"HAS_GPU_AND_TRITON", "HAS_CUDA_AND_TRITON"}:
            return ast.copy_location(ast.Constant(self.triton), node)
        return node

    def visit_Call(self, node):
        name = ast.unparse(node.func)
        if name in {"torch.cuda.is_available", "is_big_gpu"}:
            return ast.copy_location(ast.Constant(True), node)
        if name == "has_triton":
            return ast.copy_location(ast.Constant(self.triton), node)
        return self.generic_visit(node)

    def visit_Compare(self, node):
        if any(
            isinstance(child, ast.Call)
            and ast.unparse(child.func) == "torch.cuda.get_device_properties"
            for child in ast.walk(node)
        ):
            return ast.copy_location(ast.Constant(True), node)
        return self.generic_visit(node)


def run_guarded_tests(script: Path, namespace: dict, status_file: str | None) -> None:
    runners = {
        "torch.testing._internal.common_utils",
        "torch._dynamo.test_case",
        "torch._inductor.test_case",
    }
    allowed_calls = {
        "torch.cuda.is_available",
        "torch.cuda.get_device_properties",
        "torch.backends.mps.is_available",
        "torch.xpu.is_available",
        "has_triton",
        "is_big_gpu",
    }

    def candidates(body, guards=(), setup=False):
        for node in body:
            if isinstance(node, ast.If):
                extra_setup = any(
                    not isinstance(child, (ast.If, ast.Import, ast.ImportFrom))
                    and not (
                        isinstance(child, ast.Expr)
                        and isinstance(child.value, ast.Call)
                        and ast.unparse(child.value.func).endswith("run_tests")
                    )
                    for child in node.body
                )
                yield from candidates(
                    node.body, (*guards, node.test), setup or extra_setup
                )
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and guards
            ):
                call = node.value
                if isinstance(call.func, (ast.Name, ast.Attribute)):
                    try:
                        runner = eval(
                            compile(ast.Expression(call.func), str(script), "eval"),
                            namespace,
                        )
                    except (NameError, AttributeError):
                        continue
                    if (
                        getattr(runner, "__module__", None) in runners
                        and getattr(runner, "__name__", None) == "run_tests"
                    ):
                        yield call, runner, guards, setup

    tree = ast.parse(script.read_text(), filename=str(script))
    blocked = None
    for main in tree.body:
        if (
            not isinstance(main, ast.If)
            or ast.unparse(main.test) != "__name__ == '__main__'"
        ):
            continue
        for call, runner, guards, setup in candidates(main.body):
            if any(
                isinstance(node, ast.Call)
                and ast.unparse(node.func) not in allowed_calls
                for guard in guards
                for node in ast.walk(guard)
            ):
                continue
            if all(
                eval(compile(ast.Expression(guard), str(script), "eval"), namespace)
                for guard in guards
            ):
                continue
            relaxed = [
                ast.fix_missing_locations(
                    CpuRunnerGuard().visit(ast.parse(ast.unparse(guard), mode="eval"))
                )
                for guard in guards
            ]
            if setup or not all(
                eval(compile(guard, str(script), "eval"), namespace)
                for guard in relaxed
            ):
                blocked = guards[0]
                continue
            module = types.ModuleType("__main__")
            module.__dict__.update(namespace)
            previous = sys.modules["__main__"]
            sys.modules["__main__"] = module
            try:
                args = [
                    eval(compile(ast.Expression(arg), str(script), "eval"), namespace)
                    for arg in call.args
                ]
                kwargs = {
                    arg.arg: eval(
                        compile(ast.Expression(arg.value), str(script), "eval"),
                        namespace,
                    )
                    for arg in call.keywords
                }
                runner(*args, **kwargs)
            finally:
                sys.modules["__main__"] = previous
            return
    if blocked is not None and status_file:
        Path(status_file).write_text(
            json.dumps(
                {
                    "skipped": f"Main runner requires {ast.unparse(blocked)}",
                    "filename": str(script),
                    "line": blocked.lineno,
                }
            )
        )


@contextmanager
def distributed_environment(script: Path):
    files = (
        "test/distributed/test_distributed_spawn.py",
        "test/distributed/algorithms/quantization/test_quantization.py",
    )
    if not any(script.as_posix().endswith(f"/{file}") for file in files):
        yield
        return

    import torch
    from torch.testing._internal import tci

    active = tci.current_platform()
    accelerator = active is not None and active.gpu is not None
    devices = torch.cuda.device_count() if accelerator else 0
    backend = os.environ.get("BACKEND", "nccl" if accelerator else "gloo")
    with ExitStack() as stack:
        directory = os.environ.get("TEMP_DIR")
        if directory is None:
            directory = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="ramjet-distributed-")
            )
        for name in ("barrier", "test_dir"):
            Path(directory, name).mkdir(parents=True, exist_ok=True)
        defaults = {
            "BACKEND": backend,
            "WORLD_SIZE": str(max(devices, 1) if accelerator else 3),
            "TEMP_DIR": directory,
            "TEST_REPORT_SOURCE_OVERRIDE": f"dist-{backend}-init-file",
        }
        previous = {name: os.environ.get(name) for name in defaults}
        for name, value in defaults.items():
            os.environ.setdefault(name, value)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m tools.ramjet_run TEST_FILE [TEST_ARGS ...]")
    script = Path(sys.argv[1]).resolve()
    sys.argv = sys.argv[1:]
    sys.path[0] = str(script.parent)
    metadata_file = os.getenv("PYTORCH_TEST_LIST_MARKERS_FILE")
    status_file = os.getenv("PYTORCH_TEST_COLLECTION_STATUS_FILE")
    collecting = "--list-markers" in sys.argv
    cpu_selection = (
        os.getenv("PYTORCH_TEST_TCI_PLATFORM") == "linux_x86_64"
        and "PYTORCH_TEST_SELECTION_FILE" in os.environ
    )
    try:
        with distributed_environment(script):
            namespace = runpy.run_path(str(script), run_name="__main__")
            if (collecting or cpu_selection) and not (
                (metadata_file and Path(metadata_file).exists())
                or (status_file and Path(status_file).exists())
            ):
                run_guarded_tests(script, namespace, status_file)
    except (SystemExit, unittest.SkipTest) as error:
        if not status_file or "--list-markers" not in sys.argv:
            raise
        if metadata_file and Path(metadata_file).exists():
            raise
        frames = []
        current = error.__traceback__
        while current is not None:
            frames.append(current)
            current = current.tb_next
        origin = frames[-1]
        if isinstance(error, SystemExit) and (
            error.code not in (None, 0) or origin.tb_frame.f_code.co_name != "<module>"
        ):
            raise
        if any(
            "_pytest" in Path(frame.tb_frame.f_code.co_filename).parts
            or (
                frame.tb_frame.f_code.co_name == "run_tests"
                and Path(frame.tb_frame.f_code.co_filename).name == "common_utils.py"
            )
            for frame in frames
        ):
            raise
        reason = (
            str(error)
            if isinstance(error, unittest.SkipTest)
            else "Module exited with status 0"
        )
        Path(status_file).write_text(
            json.dumps(
                {
                    "skipped": reason,
                    "exception": type(error).__name__,
                    "filename": origin.tb_frame.f_code.co_filename,
                    "line": origin.tb_lineno,
                }
            )
        )


if __name__ == "__main__":
    main()
