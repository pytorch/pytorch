import contextlib
import copy

import dataclasses
import os

import pickle

import tempfile
import unittest
from os.path import abspath, dirname
from types import FunctionType, ModuleType
from typing import Any, Dict, Optional, Set, Type
from unittest import mock

import torch

from . import external_utils


@dataclasses.dataclass
class DynamoConfig:
    # the name of a file to write the logs to
    log_file_name: Optional[str] = None

    # Verbose will print full stack traces on warnings and errors
    verbose: bool = os.environ.get("TORCHDYNAMO_VERBOSE", "0") == "1"

    # verify the correctness of optimized backend
    verify_correctness: bool = False

    # need this many ops to create an FX graph
    minimum_call_count: int = 1

    # turn on/off DCE pass
    dead_code_elimination: bool = True

    # disable (for a function) when cache reaches this size
    cache_size_limit: int = 64

    # whether or not to specialize on int inputs.  This only has an effect with
    # dynamic_shapes; when dynamic_shapes is False, we ALWAYS specialize on int
    # inputs
    specialize_int: bool = False

    # Assume these functions return constants
    constant_functions: Dict[ModuleType, bool] = dataclasses.field(
        default_factory=lambda: {
            torch.jit.is_scripting: False,
            torch.jit.is_tracing: False,
            torch._C._get_tracing_state: None,
            torch.fx._symbolic_trace.is_fx_tracing: False,
            torch.onnx.is_in_onnx_export: False,
            external_utils.is_compiling: True,
            torch._utils.is_compiling: True,
        }
    )

    # don't specialize on shapes and strides and put shape ops in graph
    dynamic_shapes: bool = os.environ.get("TORCHDYNAMO_DYNAMIC_SHAPES") == "1"

    # This is a temporarily flag, which changes the behavior of dynamic_shapes=True.
    # When assume_static_by_default is True, we only allocate symbols for shapes marked dynamic via mark_dynamic.
    # NOTE - this flag can be removed once we can run dynamic_shapes=False w/ the mark_dynamic API
    # see [Note - on the state of mark_dynamic]
    assume_static_by_default: bool = True

    # This flag changes how dynamic_shapes=True works, and is meant to be used in conjunction
    # with assume_static_by_default=True.
    # With this flag enabled, we always compile a frame as fully static for the first time, and, if we fail
    # any guards due to wobbles in shape, we recompile with *all* the wobbled shapes as being marked dynamic.
    automatic_dynamic_shapes = True

    # Set this to False to assume nn.Modules() contents are immutable (similar assumption as freezing)
    guard_nn_modules: bool = False

    # This feature doesn't really work.  We offer this flag for experimental
    # purposes / if you want to help us build out support.
    #
    # torchdynamo has very limited support for tensor subclasses that implement
    # __torch_function__.  Our current support is limited to tensor subclasses
    # that DO NOT store metadata on the tensor (in general, dynamo does not
    # support Python code that stores extra attributes on tensors at present).
    # If your tensor subclass purely changes function call behavior via
    # __torch_function__, you can allow torchdynamo to trace into it by
    # adding it to traceable_tensor_subclasses.  We don't do any safety checks,
    # so it is up to you to ensure that your subclass is well behaved.  See also
    # https://github.com/pytorch/torchdynamo/issues/1948
    #
    # We do NOT currently support __torch_dispatch__.  The implementation is
    # currently buggy, the main show stopper for nontrivial use is
    # https://github.com/pytorch/torchdynamo/issues/1952
    traceable_tensor_subclasses: Set[Type] = dataclasses.field(default_factory=set)

    # Suppress errors in torch._dynamo.optimize, instead forcing a fallback to eager.
    # This is a good way to get your model to work one way or another, but you may
    # lose optimization opportunities this way.  Devs, if your benchmark model is failing
    # this way, you should figure out why instead of suppressing it.
    suppress_errors: bool = bool(os.environ.get("TORCHDYNAMO_SUPPRESS_ERRORS", False))

    # Record and write an execution record of the current frame to a file
    # if an exception is encountered
    replay_record_enabled: bool = bool(os.environ.get("TORCH_COMPILE_DEBUG", False))

    # Rewrite assert statement in python with torch._assert
    rewrite_assert_with_torch_assert: bool = True

    # Show a warning on every graph break
    print_graph_breaks: bool = False

    # Disable dynamo
    disable = os.environ.get("TORCH_COMPILE_DISABLE", False)

    # If a PyTorch module is in this allowlist, torchdynamo will be allowed
    # to inline objects from it or its children.
    skipfiles_inline_module_allowlist: Set[ModuleType] = dataclasses.field(
        default_factory=lambda: {
            torch.nn,
            torch.distributions,
            torch.testing,
            torch.ao.nn,
            torch._refs,
            torch._prims,
            torch._decomp,
            torch.utils._contextlib,
        }
    )

    # If a string representing a PyTorch module is in this ignorelist,
    # the `allowed_functions.is_allowed` function will not consider it
    # when creating a list of PyTorch functions that will appear in
    # FX IR.
    allowed_functions_module_string_ignorelist: Set[str] = dataclasses.field(
        default_factory=lambda: {
            "torch.distributions",
            "torch.testing",
            "torch._refs",
            "torch._prims",
            "torch._decomp",
        }
    )

    # Debug Flag to try minifier at different stages. Possible values are {None, "aot", "dynamo"}
    # None - Minifier is switched off
    # dynamo - Runs minifier on the TorchDynamo produced graphs, if compilation fails
    # aot - Runs minifier on the Aot Autograd produced graphs, if compilation fails
    repro_after: bool = os.environ.get("TORCHDYNAMO_REPRO_AFTER", None)
    # Compiler compilation debug info
    # 1: Dumps the original graph out to repro.py if compilation fails
    # 2: Dumps a minifier_launcher.py if compilation fails.
    # 3: Always dumps a minifier_launcher.py. Good for segfaults.
    # 4: Dumps a minifier_launcher.py if the accuracy fails.
    repro_level: int = int(os.environ.get("TORCHDYNAMO_REPRO_LEVEL", 2))

    # By default, we try to detect accuracy failure by running both forward
    # and backward of a torchdynamo produced graph (if you are using repro_after
    # 'dynamo').  This setting forces us to only test the forward graph and
    # not the backward graph.  This can be helpful if you're trying to debug
    # an inference only problem, but the minifier seems to be choking on the
    # backwards step
    # TODO: Detect this situation automatically so the user doesn't need
    # to manually configure this
    repro_forward_only: bool = os.environ.get("TORCHDYNAMO_REPRO_FORWARD_ONLY") == "1"

    # The tolerance we should use when testing if a compiled graph
    # has diverged so that we should treat it as an accuracy failure
    repro_tolerance: float = 1e-3

    # Not all backends support scalars. Some calls on torch.Tensor (like .item()) return a scalar type.
    # When this flag is set to False, we introduce a graph break instead of capturing.
    # This requires dynamic_shapes to be True.
    capture_scalar_outputs: bool = False

    # Not all backends support operators that have dynamic output shape (e.g.,
    # nonzero, unique).  When this flag is set to False, we introduce a graph
    # break instead of capturing.  This requires dynamic_shapes to be True.
    # If you set this to True, you probably also want capture_scalar_outputs
    # (these are separated for historical reasons).
    capture_dynamic_output_shape_ops: bool = False

    # Should almost always be true in prod. This relaxes the requirement that cond's true_fn and
    # false_fn produces code with identical guards.
    enforce_cond_guards_match: bool = True

    # Automatically split model graph into pieces to match DDP bucket sizes
    # to allow DDP comm/compute overlap.  Disable to allow DDP models to
    # run without graph-breaks, but also without comm/compute overlap.
    # set torch._dynamo.config.log_level to INFO or DEBUG for more info
    # about optimize_ddp behavior.
    optimize_ddp: bool = True

    # If True, raises exception if TorchDynamo is called with a context manager
    raise_on_ctx_manager_usage: bool = True

    # If True, raise when aot autograd is unsafe to use
    raise_on_unsafe_aot_autograd: bool = False

    # Throw an error if backend changes without reset
    raise_on_backend_change: bool = False

    # If true, error with a better message if we symbolically trace over a
    # dynamo-optimized function. If false, silently suppress dynamo.
    error_on_nested_fx_trace: bool = True

    # Make dynamo skip guarding on hooks on nn modules
    # Note: unsafe: if your model actually has hooks and you remove them, or doesn't and  you add them,
    # dynamo will not notice and will execute whichever version you first compiled.
    skip_nnmodule_hook_guards = True

    # Disables graph breaking on rnn. YMMV with backends.
    allow_rnn: bool = False

    # Show a warning for every specialization
    print_specializations = False

    # If true, error if we try to compile a function that has
    # been seen before.
    error_on_recompile = False

    # Typically, if you mark_dynamic a dimension, we will error if the dimension
    # actually ended up getting specialized.  This knob changes the behavior so
    # that we don't error at all.  This is helpful for our CI where I'm using a
    # heuristic to mark batch dimensions as dynamic and the heuristic may get it
    # wrong.
    allow_ignore_mark_dynamic: bool = False

    # Print guards
    print_guards = os.environ.get("TORCHDYNAMO_PRINT_GUARDS", None) == "1"

    # If true, error if we try to compile a function that has
    # been seen before.
    error_on_recompile = False

    # root folder of the project
    base_dir: bool = dirname(dirname(dirname(abspath(__file__))))

    # If True, record autograd profiler events for dynamo cache lookups (guards)
    # TODO can we default this to True?
    # and how can we cause registration/deregestration to be sensitive to runtime change of this flag?
    profile_cache_lookup = False

    DEBUG_DIR_VAR_NAME = "TORCH_COMPILE_DEBUG_DIR"

    # this is to resolve a import problem in fbcode, we will be deleting
    # this very shortly
    DO_NOT_USE_legacy_non_fake_example_inputs = False

    # Whether to skip guarding on FSDP-managed modules
    skip_fsdp_guards: bool = True

    def is_fbcode(self):
        return not hasattr(torch.version, "git_version")

    def setup_debug_dir(self):
        if self.DEBUG_DIR_VAR_NAME in os.environ:
            self.debug_dir_root = os.path.join(
                os.environ[DEBUG_DIR_VAR_NAME], "torch_compile_debug"
            )
        elif self.is_fbcode():
            self.debug_dir_root = os.path.join(
                tempfile.gettempdir(), "torch_compile_debug"
            )
        else:
            self.debug_dir_root = os.path.join(os.getcwd(), "torch_compile_debug")
        return self

    def save_config(self):
        config = copy.copy(self)
        for key in _save_config_ignore:
            delattr(config, key)
        return pickle.dumps(config, protocol=2)

    def load_config(self, content):
        state = pickle.loads(content)
        self.__dict__.update(state.__dict__)
        return self

    def update(self, content_dict):
        self.__dict__.update(content_dict)

    def patch(self, arg1=None, arg2=None, **kwargs):
        """
        Decorator and/or context manager to make temporary changes to a config.

        As a decorator:

            @config.patch("name", val)
            @config.patch(name1=val1, name2=val2):
            @config.patch({"name1": val1, "name2", val2})
            def foo(...):
                ...

        As a context manager:

            with config.patch("name", val):
                ...
        """
        if arg1 is not None:
            if arg2 is not None:
                # patch("key", True) syntax
                changes = {arg1: arg2}
            else:
                # patch({"key": True}) syntax
                changes = arg1
            assert not kwargs
        else:
            # patch(key=True) syntax
            changes = kwargs
            assert arg2 is None
        assert isinstance(changes, dict), f"expected `dict` got {type(changes)}"
        prior = {}
        config = self

        class ConfigPatch(ContextDecorator):
            def __enter__(self):
                assert not prior
                for key in changes.keys():
                    # KeyError on invalid entry
                    prior[key] = getattr(config, key)
                config.__dict__.update(changes)

            def __exit__(self, exc_type, exc_val, exc_tb):
                config.__dict__.update(prior)
                prior.clear()

        return ConfigPatch()


_save_config_ignore = {
    "repro_after",
    "repro_level",
    # workaround: "cannot pickle PyCapsule"
    "constant_functions",
    # workaround: "cannot pickle module"
    "skipfiles_inline_module_allowlist",
}

config = DynamoConfig()
config.setup_debug_dir()

# Types saved/loaded in configs
CONFIG_TYPES = (int, float, bool, type(None), str, list, set, tuple, dict)


def install_config_module(module):
    """
    Converts a module-level config into a `ConfigModule()`
    """

    class ConfigModuleInstance(ConfigModule):
        _bypass_keys = set()

    def visit(source, dest, prefix):
        """Walk the module structure and move everything to module._config"""
        for key, value in list(source.__dict__.items()):
            if key.startswith("__") or isinstance(value, (ModuleType, FunctionType)):
                continue

            name = f"{prefix}{key}"
            if isinstance(value, property) and dest is module:
                # make @property work at the module level
                delattr(module, key)
                setattr(ConfigModuleInstance, key, value)
                ConfigModuleInstance._bypass_keys.add(key)
            elif isinstance(value, CONFIG_TYPES):
                config[name] = value
                if dest is module:
                    delattr(module, key)
            elif isinstance(value, type):
                assert value.__module__ == module.__name__
                # a subconfig with `class Blah:` syntax
                proxy = SubConfigProxy(module, f"{name}.")
                visit(value, proxy, f"{name}.")
                setattr(dest, key, proxy)
            else:
                raise AssertionError(f"Unhandled config {key}={value} ({type(value)})")

    config = dict()
    visit(module, module, "")
    module._config = config
    module._allowed_keys = set(config.keys())
    module.__class__ = ConfigModuleInstance


class ConfigModule(ModuleType):
    _config: Dict[str, Any]
    _allowed_keys: Set[str]
    _bypass_keys: Set[str]

    def __init__(self):
        raise NotImplementedError(
            f"use {__name__}.install_config_module(sys.modules[__name__])"
        )

    def __setattr__(self, name, value):
        if name in self._bypass_keys:
            super().__setattr__(name, value)
        elif name not in self._allowed_keys:
            raise AttributeError(f"{self.__name__}.{name} does not exist")
        else:
            self._config[name] = value

    def __getattr__(self, name):
        try:
            return self._config[name]
        except KeyError:
            # make hasattr() work properly
            raise AttributeError(f"{self.__name__}.{name} does not exist")

    def __delattr__(self, name):
        # must support delete because unittest.mock.patch deletes
        # then recreate things
        del self._config[name]

    def save_config(self):
        """Convert config to a pickled blob"""
        config = dict(self._config)
        for key in config.get("_save_config_ignore", ()):
            config.pop(key)
        return pickle.dumps(config, protocol=2)

    def load_config(self, data):
        """Restore from a prior call to save_config()"""
        self.to_dict().update(pickle.loads(data))

    def to_dict(self):
        return self._config

    def patch(self, arg1=None, arg2=None, **kwargs):
        """
        Decorator and/or context manager to make temporary changes to a config.

        As a decorator:

            @config.patch("name", val)
            @config.patch(name1=val1, name2=val2):
            @config.patch({"name1": val1, "name2", val2})
            def foo(...):
                ...

        As a context manager:

            with config.patch("name", val):
                ...
        """
        if arg1 is not None:
            if arg2 is not None:
                # patch("key", True) syntax
                changes = {arg1: arg2}
            else:
                # patch({"key": True}) syntax
                changes = arg1
            assert not kwargs
        else:
            # patch(key=True) syntax
            changes = kwargs
            assert arg2 is None
        assert isinstance(changes, dict), f"expected `dict` got {type(changes)}"
        prior = {}
        config = self

        class ConfigPatch(ContextDecorator):
            def __enter__(self):
                assert not prior
                for key in changes.keys():
                    # KeyError on invalid entry
                    prior[key] = config._config[key]
                config._config.update(changes)

            def __exit__(self, exc_type, exc_val, exc_tb):
                config._config.update(prior)
                prior.clear()

        return ConfigPatch()


class ContextDecorator(contextlib.ContextDecorator):
    """
    Same as contextlib.ContextDecorator, but with support for
    `unittest.TestCase`
    """

    def __call__(self, func):
        if isinstance(func, type) and issubclass(func, unittest.TestCase):

            class _TestCase(func):
                @classmethod
                def setUpClass(cls):
                    self.__enter__()
                    try:
                        super().setUpClass()
                    except Exception:
                        self.__exit__(None, None, None)
                        raise

                @classmethod
                def tearDownClass(cls):
                    try:
                        super().tearDownClass()
                    finally:
                        self.__exit__(None, None, None)

            _TestCase.__name__ = func.__name__
            return _TestCase

        return super().__call__(func)


class SubConfigProxy:
    """
    Shim to redirect to main config.
    `config.triton.cudagraphs` maps to _config["triton.cudagraphs"]
    """

    def __init__(self, config, prefix):
        # `super().__setattr__` to bypass custom `__setattr__`
        super().__setattr__("_config", config)
        super().__setattr__("_prefix", prefix)

    def __setattr__(self, name, value):
        return self._config.__setattr__(self._prefix + name, value)

    def __getattr__(self, name):
        return self._config.__getattr__(self._prefix + name)

    def __delattr__(self, name):
        return self._config.__delattr__(self._prefix + name)


def patch_object(obj, name, value):
    """
    Workaround `mock.patch.object` issue with ConfigModule
    """
    if isinstance(obj, ConfigModule):
        return obj.patch(name, value)
    return mock.patch.object(obj, name, value)
