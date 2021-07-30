"""Convert code snippets into runable objects.

Note that this module generates the Python code to generate a runable object,
rather than the object itself. This is deliberate; we often want to create the
object in a worker rather than the main thread / process. As a result, a common
pattern is: `worker.run(jit.generate(...))`
"""
import atexit
import inspect
import os
import re
import shutil
import tempfile
import textwrap
import threading
import typing

from torch.utils.benchmark._impl import constants
import torch.utils.benchmark._impl.templates.template as py_template
from torch.utils.benchmark._impl.templates.stubs import CompiledTemplate


SOURCE_PATH = os.path.dirname(os.path.abspath(__file__))

class CompileEnv:
    _lock: threading.RLock = threading.RLock()
    _singleton: typing.Optional["CompileEnv"] = None

    @classmethod
    def singleton(cls) -> "CompileEnv":
        with cls._lock:
            if cls._singleton is None:
                cls._singleton = CompileEnv()

        singleton = cls._singleton
        assert singleton is not None
        return singleton

    def __init__(self) -> None:
        self._py_template: str = inspect.getsource(py_template).replace(
            "class PythonTemplate:",
            f"class {constants.COMPILED_MODULE_NAME}:",
        )

        with open(os.path.join(SOURCE_PATH, "template.cpp"), "rt") as f:
            self._cpp_template: str = f.read()

        # We create a unique build dir so that separate processes will have
        # separate build roots, but threads will share the same build root.
        # `cpp_extension` uses build root as part of the cache key, so
        # per-invocation build dirs would lead to a 0% cache hit rate and
        # spurious recompilation. Consider the following:
        #   ```
        #   setup = "auto x = torch::ones({1024, 1024});"
        #   stmt = "torch::mm(x, x);"
        #   for num_threads in [1, 2, 4, 8]:
        #     print(Timer(stmt, setup, num_threads=num_threads, language="c++").blocked_autorange())
        #   ````
        # `setup` and `stmt` do not change, so we can reuse the executable from the
        # first pass through the loop.
        self._build_dir: str = tempfile.mkdtemp()  # TODO: port common._make_temp_dir
        atexit.register(shutil.rmtree, self._build_dir)

    @staticmethod
    def replace(template: str, stmt: str, setup: str, global_setup: str) -> str:
        pattern_map = {
            "STMT_TEMPLATE_LOCATION": stmt,
            "SETUP_TEMPLATE_LOCATION": setup,
            "GLOBAL_SETUP_TEMPLATE_LOCATION": global_setup,
        }

        line_pattern = re.compile(f"^(\s*)(?:#|//) ({'|'.join(pattern_map.keys())})$")
        template_lines: typing.List[str] = template.splitlines(keepends=False)

        segments: typing.List[str] = []
        for l in template_lines:
            match = line_pattern.match(l)
            if match:
                indentation, target = match.groups()
                l = textwrap.indent(pattern_map[target], indentation)
            segments.append(l)
        return "\n".join(segments)

    def apply_template(
        self,
        stmt: str,
        setup: str,
        global_setup: str,
        language: constants.Language
    ) -> str:
        if language == constants.Language.PYTHON:
            return self.replace(
                template=self._py_template,
                stmt=stmt,
                setup=setup,
                global_setup=global_setup,
            )

        assert language == constants.Language.CPP
        source = self.replace(
            template=self._cpp_template,
            stmt=stmt,
            setup=setup,
            global_setup=global_setup,
        )

        with self._lock:
            name = f"timer_cpp_{abs(hash(source))}"
            build_dir = os.path.join(self._build_dir, name)
            source_path = os.path.join(build_dir, "template_source.cpp")
            if not os.path.exists(build_dir):
                os.makedirs(build_dir)

                with open(source_path, "wt") as f:
                    f.write(source)

        return textwrap.dedent(f"""
            def _jit_cpp_template():
                import os

                import torch
                from torch.utils import cpp_extension

                cxx_flags = torch.__config__._cxx_flags().strip().split()
                if "-g" not in cxx_flags:
                    cxx_flags.append("-g")

                extra_include_paths: List[str] = [{repr(SOURCE_PATH)}]
                conda_prefix = os.getenv("CONDA_PREFIX")
                if conda_prefix is not None:
                    # Load will automatically search /usr/include, but not conda include.
                    extra_include_paths.append(os.path.join(conda_prefix, "include"))

                return cpp_extension.load(
                    name={repr(name)},
                    sources=[{repr(source_path)}],
                    build_directory={repr(build_dir)},
                    extra_cflags=cxx_flags,
                    extra_include_paths=extra_include_paths,
                    is_python_module=True,
                    is_standalone=False,
                )

            {constants.COMPILED_MODULE_NAME} = _jit_cpp_template()
        """)


def generate(work_spec: constants.WorkSpec) -> str:
    return CompileEnv.singleton().apply_template(
        stmt=work_spec.stmt,
        setup=work_spec.setup,
        global_setup=work_spec.global_setup,
        language=work_spec.language,
    )


def get() -> CompiledTemplate:
    """Convenience method to access CompiledTimerModule in a typed manner."""

    # In Python, `global` means global to a module, not global to the program.
    # So if we simply call `globals()`, we will get the globals for jit.py, not
    # the top level globals where the result of `generate` was run. So instead
    # we grab the calling frame (which expects CompiledTimerModule to be
    # defined) and search those globals.
    calling_frame = inspect.stack()[1].frame

    compiled_module = calling_frame.f_globals.get(constants.COMPILED_MODULE_NAME, None)
    if compiled_module is None:
        raise ValueError(
            f"{constants.COMPILED_MODULE_NAME} was not defined. "
            "Did you run the result of `generate`?"
        )

    return compiled_module
