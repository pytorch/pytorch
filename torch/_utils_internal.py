import functools
import logging
import os
import sys
import tempfile
from typing import Any, Dict

import torch

log = logging.getLogger(__name__)


# this arbitrary-looking assortment of functionality is provided here
# to have a central place for overrideable behavior. The motivating
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
            f"{opname}: We could not find the abstract impl for this operator. "
        )
    else:
        raise NotImplementedError(
            f"{opname}: We could not find the abstract impl for this operator. "
            f"The operator specified that you may need to import the '{module}' "
            f"Python module to load the abstract impl. {context}"
        )


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
def signpost_event(category: str, name: str, parameters: Dict[str, Any]):
    log.info("%s %s: %r", category, name, parameters)


def log_compilation_event(metrics):
    log.info("%s", metrics)


def upload_graph(graph):
    pass


def set_pytorch_distributed_envs_from_justknobs():
    pass


def log_export_usage(**kwargs):
    pass


def justknobs_check(name: str) -> bool:
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
    return True


@functools.lru_cache(None)
def max_clock_rate():
    from triton.testing import nvsmi

    return nvsmi(["clocks.max.sm"])[0]


TEST_MASTER_ADDR = "127.0.0.1"
TEST_MASTER_PORT = 29500
# USE_GLOBAL_DEPS controls whether __init__.py tries to load
# libtorch_global_deps, see Note [Global dependencies]
USE_GLOBAL_DEPS = True
# USE_RTLD_GLOBAL_WITH_LIBTORCH controls whether __init__.py tries to load
# _C.so with RTLD_GLOBAL during the call to dlopen.
USE_RTLD_GLOBAL_WITH_LIBTORCH = False
