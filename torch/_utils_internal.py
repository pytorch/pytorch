# mypy: allow-untyped-defs
import functools
import logging
import os
import sys
import tempfile
import typing_extensions
from typing import Any, Callable, Optional, TypeVar
from typing_extensions import ParamSpec

import torch
from torch._strobelight.compile_time_profiler import StrobelightCompileTimeProfiler


_T = TypeVar("_T")
_P = ParamSpec("_P")

log = logging.getLogger(__name__)

if os.environ.get("TORCH_COMPILE_STROBELIGHT", False):
    import shutil

    if not shutil.which("strobeclient"):
        log.info(
            "TORCH_COMPILE_STROBELIGHT is true, but seems like you are not on a FB machine."
        )
    else:
        log.info("Strobelight profiler is enabled via environment variable")
        StrobelightCompileTimeProfiler.enable()

# this arbitrary-looking assortment of functionality is provided here
# to have a central place for overridable behavior. The motivating
# use is the FB build environment, where this source file is replaced
# by an equivalent.

if torch._running_with_deploy():
    # __file__ is meaningless in the context of frozen torch used in torch deploy.
    # setting empty torch_parent should allow below functions to operate without crashing,
    # but it's unclear if there is a valid use case for them in the context of deploy.
    torch_parent = ""
else:
    if os.path.basename(os.path.dirname(__file__)) == "shared":
        torch_parent = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    else:
        torch_parent = os.path.dirname(os.path.dirname(__file__))


def get_file_path(*path_components: str) -> str:
    return os.path.join(torch_parent, *path_components)


def get_file_path_2(*path_components: str) -> str:
    return os.path.join(*path_components)


def get_writable_path(path: str) -> str:
    if os.access(path, os.W_OK):
        return path
    return tempfile.mkdtemp(suffix=os.path.basename(path))


def prepare_multiprocessing_environment(path: str) -> None:
    pass


def resolve_library_path(path: str) -> str:
    return os.path.realpath(path)


def throw_abstract_impl_not_imported_error(opname, module, context):
    if module in sys.modules:
        raise NotImplementedError(
            f"{opname}: We could not find the fake impl for this operator. "
        )
    else:
        raise NotImplementedError(
            f"{opname}: We could not find the fake impl for this operator. "
            f"The operator specified that you may need to import the '{module}' "
            f"Python module to load the fake impl. {context}"
        )


# NB!  This treats "skip" kwarg specially!!
def compile_time_strobelight_meta(
    phase_name: str,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def compile_time_strobelight_meta_inner(
        function: Callable[_P, _T],
    ) -> Callable[_P, _T]:
        @functools.wraps(function)
        def wrapper_function(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            if "skip" in kwargs and isinstance(skip := kwargs["skip"], int):
                kwargs["skip"] = skip + 1

            # This is not needed but we have it here to avoid having profile_compile_time
            # in stack traces when profiling is not enabled.
            if not StrobelightCompileTimeProfiler.enabled:
                return function(*args, **kwargs)

            return StrobelightCompileTimeProfiler.profile_compile_time(
                function, phase_name, *args, **kwargs
            )

        return wrapper_function

    return compile_time_strobelight_meta_inner


# Meta only, see
# https://www.internalfb.com/intern/wiki/ML_Workflow_Observability/User_Guides/Adding_instrumentation_to_your_code/
#
# This will cause an event to get logged to Scuba via the signposts API.  You
# can view samples on the API at https://fburl.com/scuba/workflow_signpost/zh9wmpqs
# we log to subsystem "torch", and the category and name you provide here.
# Each of the arguments translate into a Scuba column.  We're still figuring
# out local conventions in PyTorch, but category should be something like
# "dynamo" or "inductor", and name should be a specific string describing what
# kind of event happened.
#
# Killswitch is at
# https://www.internalfb.com/intern/justknobs/?name=pytorch%2Fsignpost#event
def signpost_event(category: str, name: str, parameters: dict[str, Any]):
    log.info("%s %s: %r", category, name, parameters)


def log_compilation_event(metrics):
    log.info("%s", metrics)


def upload_graph(graph):
    pass


def set_pytorch_distributed_envs_from_justknobs():
    pass


def log_export_usage(**kwargs):
    pass


def log_trace_structured_event(*args, **kwargs) -> None:
    pass


def log_cache_bypass(*args, **kwargs) -> None:
    pass


def log_torchscript_usage(api: str, **kwargs):
    _ = api
    return


def check_if_torch_exportable():
    return False


def export_training_ir_rollout_check() -> bool:
    return True


def full_aoti_runtime_assert() -> bool:
    return True


def log_torch_jit_trace_exportability(
    api: str,
    type_of_export: str,
    export_outcome: str,
    result: str,
):
    _, _, _, _ = api, type_of_export, export_outcome, result
    return


def justknobs_check(name: str, default: bool = True) -> bool:
    """
    This function can be used to killswitch functionality in FB prod,
    where you can toggle this value to False in JK without having to
    do a code push.  In OSS, we always have everything turned on all
    the time, because downstream users can simply choose to not update
    PyTorch.  (If more fine-grained enable/disable is needed, we could
    potentially have a map we lookup name in to toggle behavior.  But
    the point is that it's all tied to source code in OSS, since there's
    no live server to query.)

    This is the bare minimum functionality I needed to do some killswitches.
    We have a more detailed plan at
    https://docs.google.com/document/d/1Ukerh9_42SeGh89J-tGtecpHBPwGlkQ043pddkKb3PU/edit
    In particular, in some circumstances it may be necessary to read in
    a knob once at process start, and then use it consistently for the
    rest of the process.  Future functionality will codify these patterns
    into a better high level API.

    WARNING: Do NOT call this function at module import time, JK is not
    fork safe and you will break anyone who forks the process and then
    hits JK again.
    """
    return default


def justknobs_getval_int(name: str) -> int:
    """
    Read warning on justknobs_check
    """
    return 0


def is_fb_unit_test() -> bool:
    return False


@functools.cache
def max_clock_rate():
    """
    unit: MHz
    """
    if not torch.version.hip:
        from triton.testing import nvsmi

        return nvsmi(["clocks.max.sm"])[0]
    else:
        # Manually set max-clock speeds on ROCm until equivalent nvmsi
        # functionality in triton.testing or via pyamdsmi enablement. Required
        # for test_snode_runtime unit tests.
        gcn_arch = str(torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0])
        if "gfx94" in gcn_arch:
            return 1700
        elif "gfx90a" in gcn_arch:
            return 1700
        elif "gfx908" in gcn_arch:
            return 1502
        elif "gfx12" in gcn_arch:
            return 1700
        elif "gfx11" in gcn_arch:
            return 1700
        elif "gfx103" in gcn_arch:
            return 1967
        elif "gfx101" in gcn_arch:
            return 1144
        elif "gfx95" in gcn_arch:
            return 1700  # TODO: placeholder, get actual value
        else:
            return 1100


def get_mast_job_name_version() -> Optional[tuple[str, int]]:
    return None


TEST_MASTER_ADDR = "127.0.0.1"
TEST_MASTER_PORT = 29500
# USE_GLOBAL_DEPS controls whether __init__.py tries to load
# libtorch_global_deps, see Note [Global dependencies]
USE_GLOBAL_DEPS = True
# USE_RTLD_GLOBAL_WITH_LIBTORCH controls whether __init__.py tries to load
# _C.so with RTLD_GLOBAL during the call to dlopen.
USE_RTLD_GLOBAL_WITH_LIBTORCH = False
# If an op was defined in C++ and extended from Python using the
# torch.library.register_fake, returns if we require that there be a
# m.set_python_module("mylib.ops") call from C++ that associates
# the C++ op with a python module.
REQUIRES_SET_PYTHON_MODULE = False


def maybe_upload_prof_stats_to_manifold(profile_path: str) -> Optional[str]:
    print("Uploading profile stats (fb-only otherwise no-op)")
    return None


def log_chromium_event_internal(
    event: dict[str, Any],
    stack: list[str],
    logger_uuid: str,
    start_time_ns: int,
):
    return None


def record_chromium_event_internal(
    event: dict[str, Any],
):
    return None


def profiler_allow_cudagraph_cupti_lazy_reinit_cuda12():
    return True


def deprecated():
    """
    When we deprecate a function that might still be in use, we make it internal
    by adding a leading underscore. This decorator is used with a private function,
    and creates a public alias without the leading underscore, but has a deprecation
    warning. This tells users "THIS FUNCTION IS DEPRECATED, please use something else"
    without breaking them, however, if they still really really want to use the
    deprecated function without the warning, they can do so by using the internal
    function name.
    """

    def decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        # Validate naming convention – single leading underscore, not dunder
        if not (func.__name__.startswith("_")):
            raise ValueError(
                "@deprecate must decorate a function whose name "
                "starts with a single leading underscore (e.g. '_foo') as the api should be considered internal for deprecation."
            )

        public_name = func.__name__[1:]  # drop exactly one leading underscore
        module = sys.modules[func.__module__]

        # Don't clobber an existing symbol accidentally.
        if hasattr(module, public_name):
            raise RuntimeError(
                f"Cannot create alias '{public_name}' -> symbol already exists in {module.__name__}. \
                 Please rename it or consult a pytorch developer on what to do"
            )

        warning_msg = f"{func.__name__[1:]} is DEPRECATED, please consider using an alternative API(s). "

        # public deprecated alias
        alias = typing_extensions.deprecated(
            warning_msg, category=UserWarning, stacklevel=1
        )(func)

        alias.__name__ = public_name

        # Adjust qualname if nested inside a class or another function
        if "." in func.__qualname__:
            alias.__qualname__ = func.__qualname__.rsplit(".", 1)[0] + "." + public_name
        else:
            alias.__qualname__ = public_name

        setattr(module, public_name, alias)

        return func

    return decorator
