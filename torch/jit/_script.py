"""TorchScript

This module contains functionality to support the JIT's scripting frontend, notably:
    - torch.jit.script

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""
import torch

import functools
import collections

from torch.jit._recursive import ScriptMethodStub
from torch.nn import Module
from torch.jit._state import _enabled
from torch._six import with_metaclass

if _enabled:
    Attribute = collections.namedtuple("Attribute", ["value", "type"])
else:

    def Attribute(value, type):
        return value


# For each user-defined class that subclasses ScriptModule, this meta-class:
# (1) finds all the methods annotated with @script_method in a ScriptModule and
#     removes them from the class attributes
# (2) puts a wrapper around the class's __init__ method to recusively compile
#     all of the script_methods with the module after the original __init__ has
#     run. This has to occur after the user-defined __init__ so that submodules and
#     parameters are initialized _before_ the script compiler resolve references to
#     `self.param` or `self.module`.
class ScriptMeta(type):
    def __init__(cls, name, bases, attrs):  # noqa: B902
        # Aggregate all the ScriptMethods and constants from superclasses
        cls._methods = {}
        cls._constants_set = set(getattr(cls, "__constants__", ()))
        for base in reversed(bases):
            for k, v in getattr(base, "_methods", {}).items():
                cls._methods[k] = v
            base_constants = getattr(base, "_constants_set", set())
            cls._constants_set = cls._constants_set.union(base_constants)

        # find all the script methods of the current class
        for k, v in sorted(attrs.items()):
            if isinstance(v, ScriptMethodStub):
                delattr(cls, k)
                cls._methods[v.original_method.__name__] = v

        if getattr(cls, "_disable_script_meta", False):
            # We leave built-in ScriptModule types alone, since this metaclass
            # is only for compiling user classes that inherit from
            # ScriptModule.
            return super(ScriptMeta, cls).__init__(name, bases, attrs)

        original_init = getattr(cls, "__init__", lambda self: None)

        @functools.wraps(original_init)
        def init_then_script(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if type(self) == cls:

                def make_stubs(module):
                    cls = type(module)
                    return [v for k, v in sorted(cls._methods.items())]

                self.__dict__[
                    "_actual_script_module"
                ] = torch.jit._recursive.create_script_module(self, make_stubs)

                # Delete the Python attributes that now shadow the ScriptModule
                # ones, so that __getattr__ and __setattr__ will properly find
                # the scripted versions.
                concrete_type = self._actual_script_module._concrete_type
                for name in concrete_type.get_attributes():
                    delattr(self, name)
                for name, _ in concrete_type.get_modules():
                    delattr(self, name)
                for name in ("_parameters", "_buffers", "_modules"):
                    delattr(self, name)

        cls.__init__ = init_then_script
        return super(ScriptMeta, cls).__init__(name, bases, attrs)


class _CachedForward(object):
    def __get__(self, obj, cls):
        return self.__getattr__("forward")


if _enabled:
    # this is a Python 'non-data descriptor' that causes the first access
    # to ScriptModule's forward to lookup the forward method and stash
    # it in the objects dict. Due to the standard rules for attribute lookup
    # subsequent lookups will just directly return the previously looked up method.
    # This is necessary because nn.Module defines forward as a method. If we
    # did nothing __getattr__ would not be called. Instead we'd get nn.Module.forward
    # which always throws an exception.

    class ScriptModule(with_metaclass(ScriptMeta, Module)):
        """
        ``ScriptModule``s wrap a C++ ``torch::jit::Module``. ``ScriptModule``s
        contain methods, attributes, parameters, and
        constants. These can be accessed the same as on a normal ``nn.Module``.
        """

        def __init__(self):
            super(ScriptModule, self).__init__()

        forward = _CachedForward()

        def __getattr__(self, attr):
            if "_actual_script_module" not in self.__dict__:
                return super(ScriptModule, self).__getattr__(attr)
            return getattr(self._actual_script_module, attr)

        def __setattr__(self, attr, value):
            if "_actual_script_module" not in self.__dict__:
                # Unwrap torch.jit.Attribute into a regular setattr + recording
                # the provided type in __annotations__.
                #
                # This ensures that if we use the attr again in `__init__`, it
                # will look like the actual value, not an instance of Attribute.
                if isinstance(value, Attribute):
                    # NB: Ensure that we set __annotations__ on the specific
                    # class in question, and not on a superclass (which would
                    # be wrong wrong wrong!).
                    # See also https://github.com/pytorch/pytorch/issues/39463
                    if "__annotations__" not in self.__class__.__dict__:
                        self.__class__.__annotations__ = {}
                    self.__annotations__[attr] = value.type
                    value = value.value
                return super(ScriptModule, self).__setattr__(attr, value)

            setattr(self._actual_script_module, attr, value)

        def define(self, src):
            if "_actual_script_module" in self.__dict__:
                # If we have completed initialization, just defer to the
                # backing RecursiveScriptModule to eagerly compile the provided
                # source.
                return self._actual_script_module.define(src)

            # Otherwise, we are still in the object's __init__.
            # In that case, add `src` as a stub to be compiled.
            #
            # We use frames_up=1 to get to the proper surrounding scope. The stack
            # will look like:
            # 0. createResolutionCallback
            # 1. define()
            # 2. surrounding scope.
            #
            # createResolutionCallback internally adds 1 to get us to our frame, then
            # we add 1 to get to the proper surrounding scope.
            rcb = torch._jit_internal.createResolutionCallbackFromFrame(frames_up=1)
            ast = torch._C._parse_source_def(src)
            self._methods[ast.name().name] = ScriptMethodStub(rcb, ast, None)

        def _replicate_for_data_parallel(self):
            return self._actual_script_module._replicate_for_data_parallel()


else:
    # TODO MAKE SURE THAT DISABLING WORKS
    class ScriptModule(torch.nn.Module):
        def __init__(self):
            super(ScriptModule, self).__init__()
