import inspect
import torch
import collections
import types
import textwrap
import functools
import warnings

import torch._jit_internal as _jit_internal
from torch.jit.frontend import get_default_args
from torch.nn import Module, ModuleList, Sequential, ModuleDict
from torch._six import get_function_from_type, bind_method


ScriptMethodStub = collections.namedtuple('ScriptMethodStub', ('resolution_callback', 'def_', 'original_method'))

# TODO: there should be a more principled way of doing this.
blacklist = [
    "__weakref__",
    "__constants__",
    "__overloads__",
    "__module__",
    "_version",
    "_parameters",
    "_buffers",
    "_modules",
    "dump_patches",
    "__doc__",
    "_type_frozen",
    "_defines",
    "_TracedModule__frozen",  # todo remove
]

# base types that can be constants
# in addition, tuples and lists of these base types are also considered constants
# If you edit this list, then you also need to edit the handlers in
# ConstantValue in jit/script/init.cpp
_constant_types = (bool, float, int, str, type(None), types.FunctionType, torch.device, torch.layout, torch.dtype)

def make_stub(func):
    rcb = _jit_internal.createResolutionCallbackFromClosure(func)
    ast = torch.jit.get_jit_def(func, self_name="RecursiveScriptModule")
    return ScriptMethodStub(rcb, ast, func)

def _get_valid_constant(attr, v):
    if isinstance(v, _constant_types):
        return v
    elif isinstance(v, tuple) or isinstance(v, list):
        return tuple(_get_valid_constant(attr, x) for x in v)
    constants = ", ".join(typ.__name__ for typ in _constant_types)
    raise TypeError(textwrap.dedent("""
        '{}' object for attribute '{}' is not a valid constant.
        Valid constants are:
          1. a nn.ModuleList
          2. a value of type {{{}}}
          3. a list or tuple of (2)
        """.format(type(v).__name__, attr, constants)))


def get_module_meta(original, level=0):
    # def printt(*args):
    #     print("\t" * level, *args)
    # Object + class -> class metadata
    # Metadata is the key for the class uniqueness.
    #   Constants as pyvalues
    #   Types of all the attributes
    # If we can't script the type, store the reason why and methods will let it work.
    # If present already, just construct that class type and install the data

    # build type metatdata
    # it's a map of attr name => ...
    #     - for constants: pyvalue
    #     - for attributes: type
    added_names = set()
    module_meta = torch._C.ModuleMetadata()
    module_meta.add_pyclass(type(original))

    for name, item in original._parameters.items():
        if item is None:
            # TODO special case: parameters can be None. The JIT assumes
            # parameters are Tensor types, so in this case just add it as a
            # attribute.
            # The "correct" fix here is to add the parameter as a NoneType
            # attribute, but NoneType refinemenet is currently wonky
            continue
        assert isinstance(item, torch.Tensor)
        attr_type = torch._C._jit_try_infer_type(item)
        module_meta.add_attribute(name, attr_type, True)
        added_names.add(name)

    for name, item in original._modules.items():
        sub_module_meta, type_ = get_type(item, level + 1)
        module_meta.add_module(name, type_, sub_module_meta)
        added_names.add(name)

    for name, item in original._buffers.items():
        if item is None:
            # TODO special case: parameters can be None. The JIT assumes
            # parameters are Tensor types, so in this case just add it as a
            # attribute
            # The "correct" fix here is to add the parameter as a NoneType
            # attribute, but NoneType refinemenet is currently wonky
            continue
        assert isinstance(item, torch.Tensor)
        attr_type = torch._C._jit_try_infer_type(item)
        module_meta.add_attribute(name, attr_type, False)
        added_names.add(name)

    # populate constants_set
    # TODO: untangle the relationship between __constants__ and _constant_set
    constants_set = getattr(original, "__constants__", set())

    # Constants annotated via `Final[T]` rather than being added to `__constants__`
    for name, ann in getattr(original, '__annotations__', {}).items():
        if torch._jit_internal.is_final(ann):
            constants_set.add(name)

    for name in constants_set:
        if name in added_names:
            # XXX: It is possible for something to be in the constants set but
            # also in the parameters/buffers. This happens in BatchNorm as a
            # hack to support optional parameters.
            continue
        if not hasattr(original, name):
            # TODO: We should really error in this case, but there are a couple
            # extant examples of this so leave it for a future PR.
            warnings.warn("'{}' was found in ScriptModule constants, "
                          "but was not actually set in __init__. "
                          "Consider removing it.".format(name))
            continue
        value = getattr(original, name)
        module_meta.add_constant(name, _get_valid_constant(name, value))
        added_names.add(name)

    # populate overloads

    overloads = getattr(original, "__overloads__", {})
    # update with any annotated overloads
    # TODO: we do this both in recursive_script and in get_type. This is
    # because overloads are considered part of the type, but also determine
    # which methods to compile.
    overloads.update(get_overload_name_mapping(get_overload_annotations(original)))
    for name, overloaded_names in overloads.items():
        module_meta.add_overload(name, overloaded_names)

    class_annotations = getattr(original, '__annotations__', {})

    # __dict__ because we only want to pick up attributes on this module
    # instance, not the class itself.
    for name, item in original.__dict__.items():
        if name in blacklist:
            # Python objects have lots of random attributes attached to them;
            # PyTorch adds a few more. Prevent these from getting compiled.
            continue

        if name in added_names:
            # Don't re-add anything we already added
            continue

        # if isinstance(getattr(type(original), name, None), property):
        #     # Avoid adding @property methods as attributes
        #     continue

        if inspect.isfunction(item) and not inspect.ismethod(item):
            cls_attr = getattr(type(original), name, None)
            if inspect.isfunction(cls_attr):
                # Skip function attributes that exist on the original class.
                #
                # We reach here for things like @staticmethod and @classmethod.
                # TODO our support for @staticmethod and @classmethod is questionable.
                continue

            # This is a Python function attribute. Try to script it.
            try:
                item = torch.jit.script(item)
            except Exception as e:
                # If we fail to script the function, it isn't a hard error.
                # Instead, we will add it to the list of attributes we failed
                # to convert, with the compilation error.
                hint = ("(This function exists as an attribute on the Python module, "
                        "but we failed to compile it to a TorchScript function. "
                        "\nThe error stack is reproduced here:\n{}").format(e)
                module_meta.add_failed_attribute(name, hint)
                pass

        if name in class_annotations:
            attr_type = torch.jit.annotations.ann_to_type(class_annotations[name])
        elif isinstance(item, torch.jit.Attribute):
            attr_type = torch.jit.annotations.ann_to_type(item.type)
        else:
            attr_type = torch._C._jit_try_infer_type(item)

        if attr_type is not None:
            module_meta.add_attribute(name, attr_type, False)
        else:
            # TODO: could add more detail here. For example, what the user should do
            # when the pytype is `list` or `NoneType`
            hint = ("(This attribute exists on the Python module, "
                    "but we failed to convert Python type: {} "
                    "to a TorchScript type.)").format(type(item).__name__)
            module_meta.add_failed_attribute(name, hint)

    return module_meta


class ModuleMetaStore(object):
    def __init__(self):
        # pytype => List[(ModuleMetadata, TypePtr)]
        self.type_meta_for_type = {}
        # TypePtr methods have been compiled
        self.methods_compiled = set()

    def find_type(self, original_type, module_meta):
        known_types = self.type_meta_for_type.get(original_type, [])
        for known_module_meta, type_ in known_types:
            if module_meta.equals(known_module_meta):
                return type_
        return None

    def create_new_type(self, original_type, module_meta):
        qualified_name = torch.jit._qualified_name(original_type)
        type_ = torch._C._get_fresh_type(qualified_name, module_meta)
        if original_type not in self.type_meta_for_type:
            self.type_meta_for_type[original_type] = []

        self.type_meta_for_type[original_type].append((module_meta, type_))
        return type_

module_meta_store = ModuleMetaStore()

def get_type(original, level=0):
    # def printt(*args):
    #     print("\t" * level, *args)
    # printt("============================")
    # printt("CALLING GET TYPE ON,", type(original).__name__)
    assert isinstance(original, Module)
    if isinstance(original, torch.jit.RecursiveScriptModule) and original._type_frozen:
        # printt("ALREADY A SCRIPT MOD")
        # printt("============================")
        return original._module_meta, original._c._type()

    module_meta = get_module_meta(original, level)
    maybe_type = module_meta_store.find_type(type(original), module_meta)
    if maybe_type is not None:
        # printt("HIT", type(original).__name__)
        type_ = maybe_type
    else:
        # printt("MISS", type(original).__name__)
        type_ = module_meta_store.create_new_type(type(original), module_meta)

    # module_meta.dump()
    # printt("======================")
    return module_meta, type_

def create_methods_from_stubs(cpp_mod, module_meta, stubs):
    defs = [m.def_ for m in stubs]
    rcbs = [m.resolution_callback for m in stubs]
    defaults = [get_default_args(m.original_method) for m in stubs]
    cpp_mod._create_methods(module_meta, defs, rcbs, defaults)

def compile_unbound_method(cpp_mod, module_meta, fn):
    if _jit_internal.is_ignored_fn(fn):
        return None
    stub = make_stub(fn)
    with torch.jit._disable_emit_hooks():
        # We don't want to call the hooks here since the graph that is calling
        # this function is not yet complete
        create_methods_from_stubs(cpp_mod, module_meta, (stub,))
    return stub

# This is called in the following ways:
# 1. From recursive script. py_module is an nn.Module
# 2. From ScriptMeta2. py_module is a ScriptModule
# 3. From tracing, in the construction of a traced module. py_module is a
#    ScriptModule with a fresh ._c already set
# 4. From tracing, if we have exports set on the traced module. Then its called
#    directly to just compile the exports.
def create_script_module(original_module, stubs, fresh_type=False):
    assert isinstance(original_module, torch.nn.Module)

    module_meta, module_type = get_type(original_module)
    if fresh_type:
        cpp_mod = torch._C.ScriptModule(torch._jit_internal._qualified_name(type(original_module)), torch.jit._python_cu, True)
        module_type = cpp_mod._type()
    else:
        cpp_mod = torch._C._create_module_with_type(module_type)
    script_module = torch.jit.RecursiveScriptModule(cpp_mod)

    # Add attributes/parameters
    for name, (attr_type, is_param) in module_meta.get_attributes().items():
        orig_value = getattr(original_module, name)

        if is_param:
            cpp_mod._register_parameter(name, orig_value, False)
        elif isinstance(orig_value, torch.jit.Attribute):
            cpp_mod._register_attribute(name, attr_type, orig_value.value)
        else:
            cpp_mod._register_attribute(name, attr_type, orig_value)

    # Add modules, recursively scripting them.
    for name in module_meta.get_module_names():
        orig_value = getattr(original_module, name)
        assert isinstance(orig_value, Module)
        scripted = recursive_script(orig_value)
        cpp_mod._register_module(name, scripted._c)

        script_module._modules[name] = scripted

    # Copy @ignored/@unused methods from the original module to the new one.
    # This ensures we can access these Python methods on the resulting ScriptModule.
    # TODO: when we split the module stuff apart, this needs to go in the outer
    # part (or does it?)
    for name in dir(original_module):
        item = getattr(original_module, name, None)
        if not inspect.ismethod(item):
            continue
        if _jit_internal.is_ignored_fn(item) or hasattr(item, "_parameter_names_fn"):
            setattr(script_module, name, item)

    script_module._module_meta = module_meta
    # finalize here so that everything is correctly available when we are compiling
    # TODO: seem finnicky
    script_module._finalize()

    # Compile methods if necessary
    if module_type not in module_meta_store.methods_compiled:
        create_methods_from_stubs(cpp_mod, module_meta, stubs)
        module_meta_store.methods_compiled.add(module_type)

    # Make the compiled methods available to the Python ScriptModule class.
    for stub in stubs:
        if stub.original_method is None:
            # define()'d methods don't have an original_method, so we don't #
            # need to do any Python stuff--we'll just look it up on the cpp_mod
            # directly
            continue

        name = stub.original_method.__name__
        if name != stub.def_.name().name:
            # TODO: Why skip this? Because @torch.jit._overload_method will
            # mangle the name of the function.
            continue
        script_method = cpp_mod._get_method(name)
        # Wrap the original to propagate docstrings and such.
        # TODO: we don't currently do this functions that are recursively
        # compiled, we should.
        script_method = functools.wraps(stub.original_method)(script_method)

        # Add the methods to the script_module directly. This ensures they will
        # be found first when `name` is looked up (as opposed to the stubs or
        # nn.Module.forward)
        script_module.__dict__[name] = script_method

    return script_module

def get_overload_annotations(mod):
    # original function => [(mangled overload name, overload function)]
    overloads = {}
    for name in dir(mod):
        item = getattr(mod, name, None)
        if not callable(item):
            continue

        # builtin functions like repr() in python 2 do not have __module__ defined
        if hasattr(item, "__module__") and item.__module__ is not None:
            method_overloads = _jit_internal._get_overloaded_methods(item, mod.__class__)
            if method_overloads is None:
                continue

            original_name = item.__name__
            names = [name + "__" + str(i) for i in range(len(method_overloads))]
            overloads[item] = list(zip(names, method_overloads))

    return overloads

def get_overload_name_mapping(overload_info):
    # Same format as __overloads__
    # original function => [overload names]
    overload_name_mappings = {}
    for orig_fn, overloads in overload_info.items():
        original_name = orig_fn.__name__
        if original_name not in overload_name_mappings:
            overload_name_mappings[original_name] = []

        for overload_name, _ in overloads:
            overload_name_mappings[original_name].append(overload_name)
    return overload_name_mappings

def make_stubs_for_overloads(overload_info):
    overload_stubs = []
    for orig_fn, overloads in overload_info.items():
        orig_ast = torch.jit.get_jit_def(orig_fn, self_name="RecursiveScriptModule")
        for overload_name, overload_fn in overloads:
            torch.jit._check_no_signature(overload_fn)
            over_ast = torch.jit.get_jit_def(overload_fn, self_name="RecursiveScriptModule")
            new_ast = torch._C._replace_overloaded_method_decl(over_ast.decl(), orig_ast, overload_name)
            _rcb = _jit_internal.createResolutionCallbackFromClosure(orig_fn)
            overload_stubs.append(ScriptMethodStub(_rcb, new_ast, overload_fn))
    return overload_stubs

def recursive_script(mod, exclude_methods=()):
    """
    Makes a ScriptModule from an nn.Module.

    Primarily responsible for determining which methods should act as
    starting points for compilation. These methods are turned into stubs and
    handed off to the actual compilation process.
    """
    if isinstance(mod, torch.jit.ScriptModule):
        return mod

    if isinstance(mod, (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ModuleDict)):
        # Create constant versions for the iterable modules
        return create_constant_iterable_module(mod)

    if not hasattr(mod, '_parameters'):
        raise RuntimeError("'{}' has not been initialized, did you forget to call 'super()'?"
                           .format(type(mod).__name__))

    methods = []
    if hasattr(mod, 'forward'):
        if getattr(mod.forward, "__func__", None) == torch.nn.Module.forward:
            # TODO, we deleted a check that forward is actually defined, instead skipping it
            pass
        elif not _jit_internal.is_ignored_fn(mod.forward):
            methods = ['forward']

    exported = []
    for name in dir(mod):
        item = getattr(mod, name, None)
        if _jit_internal.get_torchscript_modifier(item) is _jit_internal.FunctionModifiers.EXPORT:
            exported.append(name)

    methods = methods + exported
    methods = [name for name in methods if name not in exclude_methods]

    overload_name_mappings = dict(getattr(mod, "__overloads__", {}))
    overload_info = get_overload_annotations(mod)
    overload_name_mappings.update(get_overload_name_mapping(overload_info))
    overload_stubs = make_stubs_for_overloads(overload_info)

    mod.__overloads__ = overload_name_mappings

    # we shouldn't directly compile overloaded methods, just its overloads
    def ignore_overloaded(method_name):
        return method_name not in overload_name_mappings

    def make_stub_from_method(method):
        func = get_function_from_type(type(mod), method)
        if isinstance(func, ScriptMethodStub):
            return func
        return make_stub(func)

    filtered_methods = filter(ignore_overloaded, methods)

    # Unique the methods. We don't want to use a set to store the methods because it
    # introduces non-determinism to compile order.
    uniquer = set()
    uniqued_methods = []
    for name in filtered_methods:
        if name in uniquer:
            continue
        uniqued_methods.append(name)
        uniquer.add(name)

    stubs = list(map(make_stub_from_method, uniqued_methods))
    return create_script_module(mod, overload_stubs + stubs)

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
        if isinstance(submodule, (ModuleList, Sequential, ModuleDict)):
            # Make each item in the module a constant
            modules[key] = create_constant_iterable_module(submodule)
        else:
            modules[key] = recursive_script(submodule)

    if isinstance(module, Sequential):
        return torch.jit._ConstSequential(Sequential(modules))
    elif isinstance(module, ModuleList):
        return torch.jit._ConstModuleList(modules)
    elif isinstance(module, ModuleDict):
        return torch.jit._ConstModuleDict(modules)
    else:
        raise RuntimeError("Only nn.ModuleList, nn.Sequential, and nn.ModuleDict can be made "
                           "into constant modules, found {}".format(module))

def wrap_cpp_module(cpp_module):
    """
    Wrap this torch._C.ScriptModule in a Python ScriptModule, recursively for all submodules
    """
    script_module = torch.jit.RecursiveScriptModule(cpp_module)
    for name, cpp_mod in script_module._c._get_modules():
        setattr(script_module, name, wrap_cpp_module(cpp_mod))

    script_module._finalize()
    return script_module

# These exist to be called from C++, because pybind does not provide a builtin
# binding for issubclass
def is_module_dict(cls):
    return issubclass(cls, torch.jit._ConstModuleDict) or issubclass(cls, torch.nn.ModuleDict)

def is_module_list(cls):
    return issubclass(cls, torch.jit._ConstModuleList) or issubclass(cls, torch.nn.ModuleList) \
        or issubclass(cls, torch.nn.Sequential)

def bind_to_dummy_module(module_meta, unbound_method, cpp_mod):
    """
    Create a dummy ScriptModule object for `unbound_method` to bind to.
    This ScriptModule object should behave "as if" it was the actual
    ScriptModule object holding `cpp_mod`, but since we constructed it only
    from information in `module_meta` we can be sure that there is no dynamic
    shenanigans that would ruin our type sharing.
    """
    script_module = torch.jit.RecursiveScriptModule(cpp_mod)
    orig_class = module_meta.py_class

    # Copy @ignored/@unused methods from the original module to the new one.
    # This ensures they are available during compilation.
    for name in dir(orig_class):
        item = getattr(orig_class, name, None)
        if _jit_internal.is_ignored_fn(item) or hasattr(item, "_parameter_names_fn"):
            setattr(script_module, name, item)

    for name, value in module_meta.get_constants().items():
        setattr(script_module, name, value)

    script_module._finalize()
    method = bind_method(unbound_method, script_module, torch.jit.RecursiveScriptModule)
    return method
