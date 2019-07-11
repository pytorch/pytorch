import torch
import torch.jit
import torch._jit_internal as _jit_internal

from torch._six import get_function_from_type

import contextlib
import inspect

from collections import OrderedDict


code_cache = {}


def copy_module_to_script_module(module):
    """
    Copies the parameters, buffers, constants, attributes, and submodules
    of an nn.Module into itself.
    """
    script_module = torch.jit.ScriptModule()
    if not hasattr(module, '_parameters'):
        raise RuntimeError("'{}' has not been initialized, did you forget to call 'super()'?"
                           .format(type(module).__name__))

    constants_set = set(getattr(module, "__constants__", []))
    script_module.__dict__["_constants_set"] = {}

    # Copy Parameters and Modules
    for name in dir(module):
        item = getattr(module, name)
        if item is None and name in module._parameters:
            # XXX: treat None value simply as module attributes instead of adding them to the parameter list
            # TODO: need to handle this more generally when non-tensor attributes added to module
            object.__setattr__(script_module, name, item)
        elif isinstance(item, (torch.nn.Parameter, torch.jit.Attribute)):
            if isinstance(item, (torch.nn.ModuleList, torch.nn.Sequential)):
                # These are in __constants__, so ignore them here

                if not is_recursive_script_enabled(item):
                    # For recursive script, these are constantified after
                    # they are used, so they don't need to be in constants.
                    # The `continue` here should be deleted along with
                    # [weak script refactor]
                    continue
            setattr(script_module, name, item)
        elif isinstance(item, torch.nn.Module):
            # eagerly compile modules, but not their code
            setattr(script_module, name, copy_module_to_script_module(item))

    # Copy buffers
    for name in module._buffers:
        if module._buffers[name] is None:
            object.__setattr__(script_module, name, None)
        else:
            script_module.register_buffer(name, module._buffers[name])

    # Constants annotated via `Final[T]` rather than being added to `__constants__`
    for name, ann in getattr(module, '__annotations__', {}).items():
        if torch._jit_internal.is_final(ann):
            constants_set.add(name)

    # Copy constants
    script_module.__dict__["_constants_set"] = constants_set
    for name in script_module.__dict__["_constants_set"]:
        if hasattr(module, name):
            if (name in module._parameters or name in module._buffers) and item is not None:
                # for 'None' parameters/buffers, don't actually add their values if it exists
                continue
            setattr(script_module, name, getattr(module, name))

    # Copy annotations, pull types from `__annotations__` or try to infer
    # the type if possible
    class_annotations = getattr(module, '__annotations__', {})
    for name in dir(module):
        if name in ("training", "__dict__"):
            # TODO: removing this skip should let us remove the code to add training as an
            # attribute in python_sugared_value.cpp
            continue
        if hasattr(script_module, name):
            # Don't re-copy properties
            continue
        item = getattr(module, name)
        if name in class_annotations:
            the_type = torch.jit.annotations.ann_to_type(class_annotations[name])
        else:
            the_type = torch._C._jit_try_infer_type(item)
        if the_type is not None:
            script_module._c._register_attribute(name, the_type, item)

    # Copy overloads
    script_module.__dict__["_overloads"] = dict(getattr(module, "__overloads__", {}))

    # Copy python ops
    for name in dir(module):
        if hasattr(script_module, name):
            # Skip Python class stuff and don't re-assign anything, but keep
            # functions around so they can be called
            continue
        value = getattr(module, name)
        setattr(script_module, name, value)

    return script_module

def recursive_script(mod):
    """
    Makes a ScriptModule from an nn.Module. If `_methods` is provided,
    these methods are treated as @torch.jit.script_methods. If not, it defaults to
    `('forward',)`. Methods accessed in forward are scripted on demand if
    `_enable_recursive_script()` is used.
    """
    if isinstance(mod, torch.jit.ScriptModule):
        return mod

    if isinstance(mod, (torch.nn.ModuleList, torch.nn.Sequential)):
        # Create constant versions for the iterable modules
        return create_constant_iterable_module(mod)

    methods = ()
    if hasattr(mod, 'forward'):
        if mod.forward.__func__ == torch.nn.Module.forward:
            # TODO: [enable recursive script]
            # forward was not overrided
            raise RuntimeError("No forward method was defined on {}".format(mod))
        if not _jit_internal.is_ignored_fn(mod.forward):
            methods = ('forward',)
    exported = []
    for name in dir(mod):
        item = getattr(mod, name)
        if callable(item):
            if _jit_internal.get_torchscript_modifier(item) is _jit_internal.FunctionModifiers.EXPORT:
                exported.append(name)
    methods = methods + tuple(exported)

    def make_stub(method):
        func = get_function_from_type(type(mod), method)
        return torch.jit.script_method(func, _jit_internal.createResolutionCallbackFromClosure(func))

    stubs = list(map(make_stub, methods))

    script_module = copy_module_to_script_module(mod)
    print(script_module)
    print(script_module._c)
    print(methods)
    print(stubs)
    torch.jit._create_methods_from_stubs(script_module, stubs)

    return script_module


def create_method_from_fn(module, fn):
    if _jit_internal.is_ignored_fn(fn):
        return None
    if not inspect.ismethod(fn):
        return None
    stub = torch.jit.script_method(fn, _jit_internal.createResolutionCallbackFromClosure(fn))
    with torch.jit._disable_emit_hooks():
        # We don't want to call the hooks here since the graph that is calling
        # this function is not yet complete
        torch.jit._create_methods_from_stubs(module, (stub,))
    return stub


def create_constant_iterable_module(module):
    modules = OrderedDict()

    for key, submodule in module._modules.items():
        if isinstance(submodule, (torch.nn.ModuleList, torch.nn.Sequential)):
            # Make each item in the module a constant
            modules[key] = create_constant_iterable_module(submodule)
        else:
            modules[key] = recursive_script(submodule)

    if isinstance(module, torch.nn.Sequential):
        return torch.jit._ConstSequential(torch.nn.Sequential(modules))
    elif isinstance(module, torch.nn.ModuleList):
        return torch.jit._ConstModuleList(modules)
    else:
        raise RuntimeError("Only nn.ModuleList and nn.Sequential can be made "
                           "into constant modules, found {}".format(module))


# def make_strong_submodule(field, module, parent):
#     if field not in parent._modules:
#         # It's not a submodule, don't do anything
#         return None
#
#     # Convert the module to a ScriptModule
#     new_strong_submodule = recursive_script(module)
#
#     # Install the ScriptModule on the python side
#     parent._modules._python_modules[field] = new_strong_submodule
#
#     return new_strong_submodule


# TODO: we are leaking these things because they don't have a distinct owner
# right now.
_delete_me_recursive_compile_holder = []
def try_compile_fn(fn):
    global _delete_me_recursive_compile_holder
    if _jit_internal.is_ignored_fn(fn):
        # Don't do anything for @ignore'd functions
        return None

    if isinstance(fn, torch.nn.Module):
        # Since modules are callable pybind recognizes them as functions, but
        # don't do anything for them
        return None

    if not inspect.isfunction(fn) and not inspect.ismethod(fn):
        raise RuntimeError("`{}` is not a function. Recursive scripting only supports "
                           "Python functions or methods currently.\n"
                           "Consider manually annotating `{}` with @torch.jit.script.".format(fn, fn))

    # We don't have the actual scope where the function was defined, but we can
    # extract the necessary info from the closed over variables on the function
    # object
    rcb = _jit_internal.createResolutionCallbackFromClosure(fn)
    _delete_me_recursive_compile_holder.append(torch.jit.script(fn, _rcb=rcb))
    return _delete_me_recursive_compile_holder[-1]


def try_get_dispatched_fn(fn):
    if not callable(fn):
        return None
    return _jit_internal.boolean_dispatched.get(fn)


def try_get_overloaded_fn(mod, field):
    return mod._overloads.get(field, None) if isinstance(mod, ScriptModule) else None


def is_recursive_script_enabled(value):
    # TODO: [enable recursive script]
    # when recursive script is made the default, remove this method
    enabled = torch._C._jit_recursive_script()
    module = inspect.getmodule(value)
    if module is not None and 'torch.nn' in module.__name__:
        enabled = True
    return enabled


@contextlib.contextmanager
def enable_recursive_script():
    torch._C._jit_recursive_script(True)
    yield
    torch._C._jit_recursive_script(False)
