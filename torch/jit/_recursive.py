import inspect
import torch
import collections
import textwrap
import functools
import warnings

import torch._jit_internal as _jit_internal
from torch.jit.frontend import get_default_args
from torch.jit._builtins import _find_builtin
from torch.nn import Module
from torch._six import get_function_from_type, bind_method


ScriptMethodStub = collections.namedtuple('ScriptMethodStub', ('resolution_callback', 'def_', 'original_method'))

# TODO: there should be a more principled way of doing this.
blacklist = [
    "_version",
    "_parameters",
    "_buffers",
    "_modules",
    "_initializing",
    "_backward_hooks",
    "_forward_hooks",
    "_forward_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "dump_patches",
]

def make_stub(func):
    rcb = _jit_internal.createResolutionCallbackFromClosure(func)
    ast = torch.jit.get_jit_def(func, self_name="RecursiveScriptModule")
    return ScriptMethodStub(rcb, ast, func)

def make_stub_from_method(nn_module, method):
    func = get_function_from_type(type(nn_module), method)
    if isinstance(func, ScriptMethodStub):
        return func
    return make_stub(func)

# base types that can be constants
# in addition, tuples and lists of these base types are also considered constants
# If you edit this list, then you also need to edit the handlers in
# ConstantValue in jit/script/init.cpp
_constant_types = (bool, float, int, str, type(None), torch.device, torch.layout, torch.dtype)

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


class SourceContext(torch._C._jit_tree_views.SourceRangeFactory):
    def __init__(self, source, filename, file_lineno, leading_whitespace_len):
        super(SourceContext, self).__init__(source, filename, file_lineno, leading_whitespace_len)


def infer_concrete_type_builder(nn_module):
    """
    Build a ConcreteModuleTypeBuilder from an nn.Module. This
    ConcreteModuleType doesn't have a JIT type associated with it yet, it
    must be filled in by the caller.
    """
    concrete_type_builder = torch._C.ConcreteModuleTypeBuilder(type(nn_module))
    if isinstance(nn_module, (torch.nn.ModuleDict)):
        concrete_type_builder.set_module_dict()
    if isinstance(nn_module, (torch.nn.ModuleList, torch.nn.Sequential)):
        concrete_type_builder.set_module_list()

    class_annotations = getattr(nn_module, '__annotations__', {})

    # try to infer the type from type annotation or from the object itself
    def infer_type(name, item):
        if name in class_annotations:
            attr_type = torch.jit.annotations.ann_to_type(class_annotations[name], _jit_internal.fake_range())
        elif isinstance(item, torch.jit.Attribute):
            attr_type = torch.jit.annotations.ann_to_type(item.type, _jit_internal.fake_range())
        else:
            attr_type = torch._C._jit_try_infer_type(item)
        return attr_type

    added_names = set()

    for name, item in nn_module._parameters.items():
        assert item is None or isinstance(item, torch.Tensor)
        attr_type = infer_type(name, item)
        # We currently have the invariant in various places in our code
        # that parameters must be Tensors. However, the nn.Module API also
        # allows NoneType parameters. These parameters are not returned as
        # part of `parameters()` and its variants, but are available
        # through direct attribute access.
        concrete_type_builder.add_attribute(name, attr_type, True)
        added_names.add(name)

    for name, item in nn_module._buffers.items():
        assert item is None or isinstance(item, torch.Tensor)
        attr_type = infer_type(name, item)
        concrete_type_builder.add_attribute(name, attr_type, False)
        added_names.add(name)

    for name, item in nn_module._modules.items():
        attr_type = infer_type(name, item)
        if item is None:
            # Modules can be None. We don't have direct support for optional
            # Modules, so the register it as an NoneType attribute instead.
            concrete_type_builder.add_attribute(name, attr_type, False)
            continue
        if attr_type is not None:
            assert attr_type.is_interface_type()
            # if the type can be inferred, it should be a module interface type
            sub_concrete_type = torch._C.ConcreteModuleType.from_jit_type(attr_type)
        else:
            # otherwise we get the concrete module type for item and add it to concrete_type
            sub_concrete_type = concrete_type_store.get_or_create_concrete_type(item)
        concrete_type_builder.add_module(name, sub_concrete_type)

        added_names.add(name)

    # populate constants_set
    constants_set = getattr(nn_module, "__constants__", set())

    # Constants annotated via `Final[T]` rather than being added to `__constants__`
    for name, ann in class_annotations.items():
        if torch._jit_internal.is_final(ann):
            constants_set.add(name)

    for name in constants_set:
        if name in added_names:
            # TODO: We should really error in this case, but its bc-breaking so
            # we need to warn for at least one release
            if name in nn_module._modules:
                hint = "submodule"
            elif name in nn_module._buffers:
                hint = "buffer"
            elif name in nn_module._parameters:
                hint = "parameter"
            else:
                raise AssertionError("added_names must be submodule, parameter, or buffer")

            warnings.warn("'{}' was found in ScriptModule constants, "
                          " but it is a non-constant {}. Consider removing it.".format(name, hint))
            continue
        if not hasattr(nn_module, name):
            # TODO: We should really error in this case, but its bc-breaking so
            # we need to warn for at least one release
            warnings.warn("'{}' was found in ScriptModule constants, "
                          "but was not actually set in __init__. "
                          "Consider removing it.".format(name))
            continue
        value = getattr(nn_module, name)
        concrete_type_builder.add_constant(name, _get_valid_constant(name, value))
        added_names.add(name)

    # populate overloads
    overloads = getattr(nn_module, "__overloads__", {})
    # update with any annotated overloads
    overloads.update(get_overload_name_mapping(get_overload_annotations(nn_module)))
    for name, overloaded_names in overloads.items():
        concrete_type_builder.add_overload(name, overloaded_names)

    for name, value in nn_module.__dict__.items():
        if name in blacklist or name.startswith("__"):
            # Python objects have lots of random attributes attached to them;
            # PyTorch adds a few more. Prevent these from getting compiled.
            continue

        if name in added_names:
            # Don't re-add anything we already added
            continue

        # Handle Python function attributes
        if inspect.isfunction(value):
            try:
                scripted_fn = torch.jit.script(value)
                concrete_type_builder.add_function_attribute(
                    name,
                    torch._C._jit_try_infer_type(scripted_fn),
                    value)
            except Exception as e:
                # If we fail to script the function, it isn't a hard error.
                # Instead, we will add it to the list of attributes we failed
                # to convert, with the compilation error.
                hint = ("(This function exists as an attribute on the Python module, "
                        "but we failed to compile it to a TorchScript function. "
                        "\nThe error stack is reproduced here:\n{}").format(e)
                concrete_type_builder.add_failed_attribute(name, hint)
                pass

            continue

        # Handle calls to builtin functions (either bespoke builtins from torch.jit._builtins or
        # a call to an aten function like torch.add)
        builtin_symbol_name = _find_builtin(value)
        if builtin_symbol_name:
            concrete_type_builder.add_builtin_function(name, builtin_symbol_name)
            continue

        # Handle Script function attributes
        if isinstance(value, torch.jit.ScriptFunction):
            concrete_type_builder.add_function_attribute(
                name,
                torch._C._jit_try_infer_type(value),
                value)
            continue

        # If we got here, this is a regular "data" attribute, Add it to the concrete type
        attr_type = infer_type(name, value)
        if attr_type is not None:
            concrete_type_builder.add_attribute(name, attr_type, False)
        else:
            # TODO: could add more detail here. For example, what the user should do
            # when the pytype is `list` or `NoneType`
            hint = ("(This attribute exists on the Python module, "
                    "but we failed to convert Python type: '{}' "
                    "to a TorchScript type.)").format(type(value).__name__)
            concrete_type_builder.add_failed_attribute(name, hint)

    # Add @property methods as failed attributes, to give a better error message.
    for name, value in type(nn_module).__dict__.items():
        if isinstance(value, property):
            hint = ("\n(This attribute exists on the Python module, but it's an @property "
                    "method. @property methods are not yet supported in TorchScript. "
                    "Please file a feature request on Github)")
            concrete_type_builder.add_failed_attribute(name, hint)

    return concrete_type_builder

class ConcreteTypeStore(object):
    def __init__(self):
        # Python module type => List[ConcreteModuleType)]
        self.type_store = {}
        # ConcreteTypes that have had their methods already compiled
        self.methods_compiled = set()

    def get_or_create_concrete_type(self, nn_module):
        """
        Infer a ConcreteType from this `nn.Module` instance. Underlying JIT
        types are re-used if possible.
        """
        assert isinstance(nn_module, Module)
        if isinstance(nn_module, torch.jit.ScriptModule) and \
                hasattr(nn_module, "_concrete_type"):
            return nn_module._concrete_type

        concrete_type_builder = infer_concrete_type_builder(nn_module)

        nn_module_type = type(nn_module)
        if nn_module_type not in self.type_store:
            self.type_store[nn_module_type] = []

        # Search the type store for an already-available JIT type
        known_types = self.type_store[nn_module_type]
        for known_type in known_types:
            if known_type.equals(concrete_type_builder):
                return known_type

        # We didn't find anything; generate a new JIT type from this concrete type
        concrete_type = concrete_type_builder.build()
        self.type_store[nn_module_type].append(concrete_type)
        return concrete_type

concrete_type_store = ConcreteTypeStore()

def create_methods_from_stubs(concrete_type, stubs):
    defs = [m.def_ for m in stubs]
    rcbs = [m.resolution_callback for m in stubs]
    defaults = [get_default_args(m.original_method) for m in stubs]
    concrete_type._create_methods(defs, rcbs, defaults)

def create_script_module(nn_module, stubs_fn, share_types=True):
    """
    Creates a new ScriptModule from an nn.Module

    Arguments:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.
        share_types:  Whether to share underlying JIT types between modules (if possible).
            NOTE: Only set to False this when we cannot guarantee type sharing will work
                correctly. This only happens today for traced modules, where the same
                module can produce different traced methods depending on the inputs.
    """
    assert not isinstance(nn_module, torch.jit.RecursiveScriptModule)
    check_module_initialized(nn_module)
    if share_types:
        # Look into the store of cached JIT types
        concrete_type = concrete_type_store.get_or_create_concrete_type(nn_module)
    else:
        # Get a concrete type directly, without trying to re-use an existing JIT
        # type from the type store.
        concrete_type_builder = infer_concrete_type_builder(nn_module)
        concrete_type_builder.set_poisoned()
        concrete_type = concrete_type_builder.build()

    return create_script_module_impl(nn_module, concrete_type, stubs_fn)

def create_script_module_impl(nn_module, concrete_type, stubs_fn):
    """
    Convert an nn.Module to a RecursiveScriptModule.

    Arguments:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        concrete_type:  The fully initialized ConcreteType of the module.
        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.
    """
    cpp_module = torch._C._create_module_with_type(concrete_type.jit_type)
    stubs = stubs_fn(nn_module)

    def init_fn(script_module):
        # Initialize the ScriptModule:
        # 1. Copy the attributes/parameters/buffers from the original `nn_module` to the new ScriptModule.
        for name, (attr_type, is_param) in concrete_type.get_attributes().items():
            orig_value = getattr(nn_module, name)
            orig_value = orig_value.value if isinstance(orig_value, torch.jit.Attribute) else orig_value
            cpp_module.setattr(name, orig_value)

        # 2. Copy the submodules from the original `nn_module` to the new ScriptModule,
        #    recursively scripting them.
        for name, sub_concrete_type in concrete_type.get_modules():
            orig_value = getattr(nn_module, name)
            assert isinstance(orig_value, Module), "Expected Module but got {}".format(type(orig_value))
            module_type = sub_concrete_type.jit_type
            if isinstance(module_type, torch._C.InterfaceType):
                # use the interface inference rule to compile the module
                scripted = interface_script(module_type, orig_value)
            elif isinstance(orig_value, torch.jit.ScriptModule):
                scripted = orig_value
            else:
                # use the default recursive rule to compile the module
                scripted = create_script_module_impl(orig_value, sub_concrete_type, infer_methods_to_compile)
            cpp_module.setattr(name, scripted)
            script_module._modules[name] = scripted

        # 3. Copy @ignored/@unused methods from the original `nn_module` to the new ScriptModule.
        #    This ensures we can access these Python methods on the ScriptModule.
        for name in dir(nn_module):
            item = getattr(nn_module, name, None)
            if not inspect.ismethod(item):
                continue
            if _jit_internal.is_ignored_fn(item):
                setattr(script_module, name, item)

        # For convenience, attach the concrete type to the new ScriptModule
        script_module._concrete_type = concrete_type

    # Actually create the ScriptModule, initializing it with the function we just defined
    script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)

    # Compile methods if necessary
    if concrete_type not in concrete_type_store.methods_compiled:
        create_methods_from_stubs(concrete_type, stubs)
        torch._C._run_emit_module_hook(cpp_module)
        concrete_type_store.methods_compiled.add(concrete_type)

    # Make the compiled methods available to the Python ScriptModule class.
    for stub in stubs:
        if stub.original_method is None:
            # define()'d methods don't have an Python original_method, so we
            # don't need to do any Python re-wrapping stuff
            continue

        name = stub.original_method.__name__
        if name != stub.def_.name().name:
            # TODO: Why skip this? Because @torch.jit._overload_method will
            # mangle the name of the function.
            continue
        script_method = cpp_module._get_method(name)

        # Wrap the original to propagate docstrings and such.
        # TODO: we don't currently do this functions that are recursively
        # compiled, we should.
        script_method = functools.wraps(stub.original_method)(script_method)

        # Add the methods to the script_module directly. This ensures they will
        # be found first when `name` is looked up (as opposed to the stubs or
        # nn.Module.forward)
        script_module.__dict__[name] = script_method


    # copy over python methods to script module if they aren't defined on the script module
    # this is currently an internal api used only on module containers
    for name in dir(nn_module):
        item = getattr(nn_module, name, None)
        if _jit_internal.get_torchscript_modifier(item) is _jit_internal.FunctionModifiers.COPY_TO_SCRIPT_WRAPPER:
            add_python_attr_to_scripted_model(script_module, nn_module, name)

    return script_module


# We define shims of certain attributes on the RecursiveScriptModule to support
# magic methods. To check if a script model defines an attribute we need
# to also check that the attribute is not the shim
def script_model_defines_attr(script_model, attr):
    script_attr = getattr(script_model, attr, None)
    if script_attr is None:
        return False
    default_attr = get_function_from_type(torch.jit.RecursiveScriptModule, attr)
    if default_attr is None:
        return False
    return script_attr != default_attr

def add_python_attr_to_scripted_model(script_model, orig, attr):
    if hasattr(orig, attr) and script_model_defines_attr(script_model, attr):
        setattr(script_model, attr, getattr(orig, attr))

def get_overload_annotations(mod):
    # original function => [(mangled overload name, overload function)]
    overloads = {}

    for name in dir(type(mod)):
        item = getattr(mod, name, None)
        if not callable(item):
            continue

        # builtin functions like repr() in python 2 do not have __module__ defined
        if hasattr(item, "__module__") and item.__module__ is not None:
            method_overloads = _jit_internal._get_overloaded_methods(item, mod.__class__)
            if method_overloads is None:
                continue

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

def _check_no_signature(func):
    signature = torch.jit.annotations.get_signature(func, None, None, inspect.ismethod(func))
    if signature is None:
        qual_name = torch.jit._qualified_name(func)
        raise RuntimeError("Must explicitly add type annotations to overloaded functions: {}".format(qual_name))

def make_stubs_for_overloads(overload_info):
    overload_stubs = []
    for orig_fn, overloads in overload_info.items():
        orig_ast = torch.jit.get_jit_def(orig_fn, self_name="RecursiveScriptModule")
        for overload_name, overload_fn in overloads:
            _check_no_signature(overload_fn)
            over_ast = torch.jit.get_jit_def(overload_fn, self_name="RecursiveScriptModule")
            new_ast = torch._C._replace_overloaded_method_decl(over_ast.decl(), orig_ast, overload_name)
            _rcb = _jit_internal.createResolutionCallbackFromClosure(orig_fn)
            overload_stubs.append(ScriptMethodStub(_rcb, new_ast, overload_fn))
    return overload_stubs

def check_module_initialized(mod):
    assert isinstance(mod, torch.nn.Module)
    if not hasattr(mod, '_parameters'):
        raise RuntimeError("'{}' has not been initialized, did you forget to call 'super()'?"
                           .format(type(mod).__name__))

def infer_methods_to_compile(nn_module):
    """
    Implements the default rules for which methods should act as starting
    points for compilation (TODO add a link when the rules are published).
    """
    check_module_initialized(nn_module)

    methods = []
    if hasattr(nn_module, 'forward') and not _jit_internal.is_ignored_fn(nn_module.forward):
        forward_func = getattr(nn_module.forward, "__func__", None)
        module_forward = get_function_from_type(torch.nn.Module, "forward")
        if forward_func != module_forward:
            methods = ['forward']

    exported = []
    for name in dir(nn_module):
        item = getattr(nn_module, name, None)
        if _jit_internal.get_torchscript_modifier(item) is _jit_internal.FunctionModifiers.EXPORT:
            exported.append(name)

    methods = methods + exported

    overload_name_mappings = dict(getattr(nn_module, "__overloads__", {}))
    overload_info = get_overload_annotations(nn_module)
    overload_name_mappings.update(get_overload_name_mapping(overload_info))
    overload_stubs = make_stubs_for_overloads(overload_info)

    nn_module.__overloads__ = overload_name_mappings

    # we shouldn't directly compile overloaded methods, just its overloads
    def ignore_overloaded(method_name):
        return method_name not in overload_name_mappings

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

    stubs = []
    for method in uniqued_methods:
        stubs.append(make_stub_from_method(nn_module, method))
    return overload_stubs + stubs

def interface_script(mod_interface, nn_module):
    """
    Makes a ScriptModule from an nn.Module, using the interface methods rule for
    determining which methods to compile.

    Arguments:
        mod_interface: the interface type that the module have
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
    """
    if isinstance(nn_module, torch.jit.ScriptModule):
        return nn_module

    check_module_initialized(nn_module)

    def infer_interface_methods_to_compile(nn_module):
        """
        Rule to infer the methods from the interface type to know which
        methods need to act as starting points for compilation.
        """
        stubs = []
        for method in mod_interface.getMethodNames():
            stubs.append(make_stub_from_method(nn_module, method))
        return stubs

    return create_script_module(nn_module, infer_interface_methods_to_compile)

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

def wrap_cpp_module(cpp_module):
    """
    Wrap this torch._C.ScriptModule in a Python ScriptModule, recursively for all submodules
    """
    def init_fn(script_module):
        for name, cpp_module in torch._C.ModuleDict(script_module._c).items():
            setattr(script_module, name, wrap_cpp_module(cpp_module))
        script_module._concrete_type = torch._C.ConcreteModuleType.from_jit_type(script_module._c._type())
    return torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)

def compile_unbound_method(concrete_type, fn):
    if _jit_internal.is_ignored_fn(fn):
        return None
    stub = make_stub(fn)
    with torch.jit._disable_emit_hooks():
        # We don't want to call the hooks here since the graph that is calling
        # this function is not yet complete
        create_methods_from_stubs(concrete_type, (stub,))
    return stub

def lazy_bind(concrete_type, unbound_method):
    """
    Returns a function that lazily binds `unbound_method` to a provided
    Module IValue, then invokes the method. We do this so that any Python
    shenanigans that will poison type sharing are impossible at compile
    time.
    """
    def lazy_binding_method(cpp_module, *args):
        def init_fn(script_module):
            orig_class = concrete_type.py_class

            # Copy @ignored/@unused methods from the original module to the new one.
            # This ensures they are available during execution.
            for name in dir(orig_class):
                item = getattr(orig_class, name, None)
                if _jit_internal.is_ignored_fn(item):
                    setattr(script_module, name, item)

            # Copy constants over so they are available during execution.
            for name, value in concrete_type.get_constants().items():
                setattr(script_module, name, value)

        script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
        method = bind_method(unbound_method, script_module, torch.jit.RecursiveScriptModule)
        return method(*args)

    # make the lazy binding method "look like" the original method
    lazy_binding_method.original_fn = unbound_method
    lazy_binding_method.__name__ = unbound_method.__name__
    torch._jit_internal.copy_torchscript_modifier(unbound_method, lazy_binding_method)

    return lazy_binding_method
