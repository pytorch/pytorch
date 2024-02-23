import torch
import inspect
import functools

"""
These are changes that we'll add to PyTorch
"""


class OpDef:
    @classmethod
    def build(cls):
        return register(cls)


OPERATOR_REGS = {}


def register(opdef):
    name = opdef.__name__

    frm = inspect.stack()[2]
    mod = inspect.getmodule(frm[0])
    ns = mangle_module(mod.__name__)

    check_allowed_attrs(opdef)
    qualname = f"{ns}::{name}"
    lib = get_library_allowing_overwrite(ns, name)
    lib.define(f"{name}{opdef.schema}")
    op = torch._library.utils.lookup_op(qualname)

    impl_method = getattr(opdef, "impl", None)
    if impl_method is not None:
        # TODO: Don't allow the meta to be reused for FakeTensor.
        register_backend(lib, name, impl_method, "CompositeExplicitAutograd")

    impl_cpu_method = getattr(opdef, "impl_cpu", None)
    if impl_cpu_method is not None:
        register_backend(lib, name, impl_cpu_method, "CPU")

    impl_cuda_method = getattr(opdef, "impl_cuda", None)
    if impl_cuda_method is not None:
        register_backend(lib, name, impl_cuda_method, "CUDA")

    properties = getattr(impl_method, "__properties__", {}) if impl_method is not None else {}
    if getattr(opdef, "abstract", None):
        torch.library.impl_abstract(qualname, opdef.abstract, lib=lib)
    elif impl_method and properties.get('traceable', False):
        torch.library.impl_abstract(qualname, impl_method, lib=lib)

    register_autograd(lib, name, opdef, op)

    return op


def check_allowed_attrs(op):
    return
    attrs = set(dir(op)) - set(dir(object))
    allowed_attrs = {
        "impl_cpu",
        "impl_cuda",
        "setup_backward",
        "backward",
    }
    if attrs.issubset(allowed_attrs):
        return
    raise RuntimeError("not subset")


def get_library_allowing_overwrite(ns, name):
    qualname = f"{ns}::{name}"

    if qualname in OPERATOR_REGS:
        OPERATOR_REGS[qualname]._destroy()
        del OPERATOR_REGS[qualname]

    lib = torch.library.Library(ns, "FRAGMENT")
    OPERATOR_REGS[qualname] = lib
    return lib


def traceable(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    wrapper.__properties__ = {**getattr(f, "__properties__", {})} 
    wrapper.__properties__['traceable'] = True
    return wrapper


def device_types(*devs):
    def inner(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper.__properties__ = {**getattr(f, "__properties__", {})} 
        wrapper.__properties__["device_types"] = devs
        return wrapper

    return inner


def dispatch_keyset_before(dk):
    result = torch._C._dispatch_keyset_full()
    result = result - torch._C._dispatch_keyset_full_after(dk)
    result = result.remove(dk)
    return result


def register_backend(lib, name, kernel, key):
    if kernel is None:
        def wrapped(*args, **kwargs):
            raise RuntimeError(
                "{name}: was passed {key} Tensors, but "
                "{name}.{key.lowercase}_impl was not defined")
    else:
        def wrapped(*args, **kwargs):
            before_dense = dispatch_keyset_before(torch._C.DispatchKey.Dense)
            with torch._C._ExcludeDispatchKeyGuard(before_dense):
                return kernel(*args, **kwargs)

    lib.impl(name, wrapped, key)


def mangle_module(module):
    """Mangles the module name.

    The scheme is replacing dots with some number of underscores
    (specified as mangledN where N is the number of underscores).

    Examples:
    foo.bar.baz -> mangled1_foo_bar_baz
    foo_bar.baz -> mangled2__foo_bar__baz
    foo.__baz__ -> mangled3___foo_____baz__

    Don't parse the mangled string directly; use mangle_module and demangle_module
    """
    sep = unique_underscoring(module)
    prefix = f"mangled{len(sep)}"
    splits = module.split(".")
    return sep.join([prefix, *splits])


def demangle_module(mangled_module):
    pass


def unique_underscoring(s: str):
    i = 1
    while True:
        result = "_" * i
        if result not in s:
            return result
        i += 1


def check_supported_schema(opdef):
    return
    # TODO: figure out what to do with torchgen FunctionSchema vs C++ FunctionSchema

    # We only support the following schemas, for now.
    # - functional
    # - auto_functionalizable.
    # For all others, we ask the user to go use the raw torch.library API.
    schema = opdef._schema
    if torch._library.utils.is_functional_schema(schema):
        return
    if torch._higher_order_ops.auto_functionalized.auto_functionalizable_schema(schema):
        return
    raise NotImplementedError("")


def register_autograd(lib, name, opdef, op):
    class MyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):
            # TODO: mark-dirty things
            with torch._C._AutoDispatchBelowAutograd():
                output = op(*inputs)
            if hasattr(opdef, "setup_backward"):
                opdef.setup_backward(ctx, inputs, output)
            return output

        @staticmethod
        def backward(ctx, *grads):
            return opdef.backward(ctx, *grads)

    lib.impl(name, MyFunction.apply, "Autograd")


class RegistersSubclassOnDefinition(type):
    def __new__(cls, name, bases, dct):
        result = super().__new__(cls, name, bases, dct)
        # This is the base class
        if name == "Operator":
            return result
        opoverload = register(result)
        result._opoverload = opoverload
        return result


class Operator(metaclass=RegistersSubclassOnDefinition):
    @classmethod
    def call(cls, *args, **kwargs):
        return cls._opoverload(*args, **kwargs)
