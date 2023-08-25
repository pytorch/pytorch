import logging
import os
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


def _functionalize_sync(t):
    # This code lives in python instead of C++ since conditioning on a certain python subclass
    # is much more of a pain in C++.
    from torch._subclasses.functional_tensor import FunctionalTensor

    if isinstance(t, FunctionalTensor):
        maybe_functional_mode = torch._C._get_dispatch_mode(
            torch._C._TorchDispatchModeKey.FUNCTIONAL
        )
        # If a FunctionalTensorMode is active while syncing, we don't want it to intercept any ops that get called
        # when we sync our inner tensor.
        if maybe_functional_mode is not None:
            try:
                maybe_functional_mode.is_active = False
                torch._functionalize_sync(t.elem)
            finally:
                maybe_functional_mode.is_active = True
        else:
            torch._functionalize_sync(t.elem)
    else:
        torch._functionalize_sync(t)


def _functionalize_sync(t):
    # This code lives in python instead of C++ since conditioning on a certain python subclass
    # is much more of a pain in C++.
    from torch._subclasses.functional_tensor import FunctionalTensor

    if isinstance(t, FunctionalTensor):
        maybe_functional_mode = torch._C._get_dispatch_mode(
            torch._C._TorchDispatchModeKey.FUNCTIONAL
        )
        # If a FunctionalTensorMode is active while syncing, we don't want it to intercept any ops that get called
        # when we sync our inner tensor.
        if maybe_functional_mode is not None:
            try:
                maybe_functional_mode.is_active = False
                torch._functionalize_sync(t.elem)
            finally:
                maybe_functional_mode.is_active = True
        else:
            torch._functionalize_sync(t.elem)
    else:
        torch._functionalize_sync(t)


TEST_MASTER_ADDR = "127.0.0.1"
TEST_MASTER_PORT = 29500
# USE_GLOBAL_DEPS controls whether __init__.py tries to load
# libtorch_global_deps, see Note [Global dependencies]
USE_GLOBAL_DEPS = True
# USE_RTLD_GLOBAL_WITH_LIBTORCH controls whether __init__.py tries to load
# _C.so with RTLD_GLOBAL during the call to dlopen.
USE_RTLD_GLOBAL_WITH_LIBTORCH = False
