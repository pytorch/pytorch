import inspect
import torch
import collections

import torch._jit_internal as _jit_internal
from torch.nn import Module, ModuleList, Parameter, Sequential
from torch._six import get_function_from_type


def copy_to_script_module(original, stubs):
    """
    Copies the parameters, buffers, constants, attributes, and submodules
    of an nn.Module into itself.
    """
    if not hasattr(original, '_parameters'):
        raise RuntimeError("'{}' has not been initialized, did you forget to call 'super()'?"
                           .format(type(original).__name__))

    qualified_name = torch.jit._qualified_name(type(original))
    script_module = torch.jit.ScriptModule(_qualified_name=qualified_name)

    constants_set = set(getattr(original, "__constants__", []))
    script_module.__dict__["_constants_set"] = {}

    # Copy Parameters and Modules
    for name in dir(original):
        item = getattr(original, name)
        if item is None and name in original._parameters:
            # XXX: treat None value simply as module attributes instead of adding them to the parameter list
            # TODO: need to handle this more generally when non-tensor attributes added to module
            object.__setattr__(script_module, name, item)
        elif item is script_module:
            continue
        elif isinstance(item, (Parameter, Module, torch.jit.Attribute)):
            setattr(script_module, name, item)

    # Copy buffers
    for name in original._buffers:
        if original._buffers[name] is None:
            object.__setattr__(script_module, name, None)
        else:
            script_module.register_buffer(name, original._buffers[name])

    # Constants annotated via `Final[T]` rather than being added to `__constants__`
    for name, ann in getattr(original, '__annotations__', {}).items():
        if torch._jit_internal.is_final(ann):
            constants_set.add(name)

    # Copy constants
    script_module.__dict__["_constants_set"] = constants_set
    for name in script_module.__dict__["_constants_set"]:
        if hasattr(original, name):
            if (name in original._parameters or name in original._buffers) and item is not None:
                # for 'None' parameters/buffers, don't actually add their values if it exists
                continue
            setattr(script_module, name, getattr(original, name))

    # Copy annotations, pull types from `__annotations__` or try to infer
    # the type if possible
    class_annotations = getattr(original, '__annotations__', {})
    for name in dir(original):
        if name in ("training", "__dict__"):
            # TODO: removing this skip should let us remove the code to add training as an
            # attribute in python_sugared_value.cpp
            continue
        if hasattr(script_module, name):
            # Don't re-copy properties
            continue
        item = getattr(original, name)
        if name in class_annotations:
            the_type = torch.jit.annotations.ann_to_type(class_annotations[name])
        else:
            the_type = torch._C._jit_try_infer_type(item)
        if the_type is not None:
            script_module._c._register_attribute(name, the_type, item)

    # Copy overloads
    script_module.__dict__["_overloads"] = dict(getattr(original, "__overloads__", {}))

    # Copy links to Python methods so they can be resolved when compiling
    for name in dir(original):
        item = getattr(original, name)
        if hasattr(script_module, name):
            # Skip Python builtins and all the module methods that are already
            # attached to this since it inherits from nn.Module
            continue
        if inspect.ismethod(item):
            setattr(script_module, name, item)

    torch.jit._create_methods_from_stubs(script_module, stubs)

    # Now that methods have been compiled, take methods that have been compiled
    # and have them shadow their corresponding Python functions
    for method_name in script_module._c._method_names():
        setattr(script_module, method_name, script_module._c._get_method(method_name))

    return script_module


def recursive_script(mod, exclude_methods=()):
    """
    Makes a ScriptModule from an nn.Module. If `_methods` is provided,
    these methods are treated as @script_methods. If not, it defaults to
    `('forward',)`. Methods accessed in forward are scripted on demand.
    """
    if isinstance(mod, torch.jit.ScriptModule):
        return mod

    if isinstance(mod, (torch.nn.ModuleList, torch.nn.Sequential)):
        # Create constant versions for the iterable modules
        return create_constant_iterable_module(mod)

    methods = ()
    if hasattr(mod, 'forward'):
        if mod.forward.__func__ == torch.nn.Module.forward:
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

    methods = tuple(name for name in methods if name not in exclude_methods)

    def make_stub(method):
        func = get_function_from_type(type(mod), method)
        return torch.jit.script_method(func, _jit_internal.createResolutionCallbackFromClosure(func))

    stubs = list(map(make_stub, methods))
    return copy_to_script_module(mod, stubs)


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


def make_strong_submodule(field, module, parent):
    if field not in parent._modules:
        # It's not a submodule, don't do anything
        return None

    # Convert the module to a ScriptModule
    new_strong_submodule = recursive_script(module)

    # Install the ScriptModule on the python side
    parent._modules._python_modules[field] = new_strong_submodule

    return new_strong_submodule


def try_compile_fn(fn, loc):
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
    return torch.jit.script(fn, _rcb=rcb)


def create_constant_iterable_module(module):
    modules = collections.OrderedDict()

    for key, submodule in module._modules.items():
        if isinstance(submodule, (torch.nn.ModuleList, torch.nn.Sequential)):
            # Make each item in the module a constant
            modules[key] = create_constant_iterable_module(submodule)
        else:
            modules[key] = recursive_script(submodule)

    if isinstance(module, Sequential):
        return torch.jit._ConstSequential(Sequential(modules))
    elif isinstance(module, ModuleList):
        return torch.jit._ConstModuleList(modules)
    else:
        raise RuntimeError("Only nn.ModuleList and nn.Sequential can be made "
                           "into constant modules, found {}".format(module))
