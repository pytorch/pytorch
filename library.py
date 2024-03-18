import torch
import inspect
import functools
import re
import sys
from typing import Dict, List
import torch._inductor
from torch._inductor.decomposition import register_decomposition
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing


"""
These are changes that we'll add to PyTorch
"""


API = "torch.Operator"
# We support the following impl_<device_type> methods. Feel free to add more.
SUPPORTED_DEVICE_TYPES = {"cpu", "cuda"}
# Decorators (like @traceable) set this field on methods
ANNOTATION_FIELD = "__torch_properties__"
# Maps qualname to the Library initially used to def/impl the Operator
OP_TO_LIB: Dict[str, torch.library.Library] = {}
OP_TO_TRACEABLE_IMPL = {}


def infer_namespace(stacklevel=2):
    frame = sys._getframe(stacklevel + 1)
    mod = inspect.getmodule(frame)
    if mod is None:
        raise RuntimeError("{API}: can't infer module")

    assert mod is not None
    ns = mangle_module(mod.__name__)
    return ns


def register(opdef: "Operator"):
    name = opdef.__name__

    ns = infer_namespace(2)
    qualname = f"{ns}::{name}"

    check_nameless_schema(opdef.schema)
    check_supported_schema(opdef, qualname)
    check_allowed_attrs(opdef)
    lib = get_library_allowing_overwrite(ns, name)
    lib.define(f"{name}{opdef.schema}")
    op = torch._library.utils.lookup_op(qualname)

    impl_method = register_device_impls(lib, name, opdef, qualname)
    properties = get_properties(impl_method)
    if properties.get("traceable", False):
        OP_TO_TRACEABLE_IMPL[qualname] = impl_method
        register_inductor_decomp(qualname, impl_method, lib, name)

    if getattr(opdef, "abstract", None):
        torch.library.impl_abstract(qualname, opdef.abstract, lib=lib)
    elif impl_method and properties.get('traceable', False):
        torch.library.impl_abstract(qualname, impl_method, lib=lib)

    register_autograd(lib, name, opdef, op)

    ophandle = torch._C._dispatch_find_schema_or_throw(qualname, "")
    torch._C._dispatch_set_report_error_callback(
        ophandle, functools.partial(report_error_callback, name)
    )

    return op


def get_properties(impl_method):
    properties = getattr(impl_method, ANNOTATION_FIELD, {})
    return properties


def register_inductor_decomp(qualname, decomp, lib, name):
    # overload = torch._library.utils.lookup_op(qualname)
    # register_decomposition([overload])(decomp)

    # Super sketch
    lib.impl(name, decomp, "Functionalize")


def register_device_impls(lib, name, opdef, qualname):
    check_either_single_or_split_impl(opdef)
    impl_method = getattr(opdef, "impl", None)
    if impl_method is not None:
        # TODO: Don't allow the meta to be reused for FakeTensor.
        register_backend(lib, name, impl_method, "CompositeExplicitAutograd")
        return impl_method

    for device_type in SUPPORTED_DEVICE_TYPES:
        impl_device_method = getattr(opdef, f"impl_{device_type}", None)
        if impl_device_method is not None:
            dk = torch._C._dispatch_key_for_device(device_type)
            register_backend(lib, name, impl_device_method, dk)
            properties = get_properties(impl_device_method)
            if properties.get("inlinable", False):
                OP_TO_TRACEABLE_IMPL[qualname] = impl_device_method
                register_inductor_decomp(qualname, impl_device_method, lib, name)




def check_either_single_or_split_impl(opdef):
    has_impl = getattr(opdef, "impl", None)
    device_impls: List[str] = []
    for device_type in SUPPORTED_DEVICE_TYPES:
        device_impl = getattr(opdef, "impl_{device_type}", None)
        if device_impl is not None:
            device_impls.append(device_impl)

    if has_impl and len(device_impls) > 0:
        raise ValueError(
            f"{API}: Expected there to be either a single `impl` method or "
            f"any number of `impl_<device>` methods. Found both an `impl` method "
            f"and {device_impls} methods.")


def check_nameless_schema(schema):
    """E.g. "(Tensor x) -> Tensor" instead of sin(Tensor x) -> Tensor"""
    match = re.match(r'\(.*\) -> .*$', schema)
    if match is not None:
        return
    raise ValueError(
        f"{API}: expected .schema to look like \"(<args>) -> <rets>\" "
        f"but got {schema}")


def check_allowed_attrs(op):
    return
    attrs = set(dir(op)) - set(dir(object))
    allowed_attrs = {
        "namespace",
        "schema",
        "impl_cpu",
        "impl_cuda",
        "abstract"
        "setup_backward",
        "backward",
    }
    if attrs.issubset(allowed_attrs):
        return
    raise ValueError(
        f"{API}: Subclasses are only allowed to have the following attributes: "
        f"attrs. Got unknown attribute {attrs - allowed_attrs}; please delete "
        f"them.")


def get_library_allowing_overwrite(ns, name):
    qualname = f"{ns}::{name}"

    if qualname in OP_TO_LIB:
        OP_TO_LIB[qualname]._destroy()
        del OP_TO_LIB[qualname]

    lib = torch.library.Library(ns, "FRAGMENT")
    OP_TO_LIB[qualname] = lib
    return lib


def traceable(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    setattr(wrapper, ANNOTATION_FIELD, {**getattr(f, ANNOTATION_FIELD, {})})
    getattr(wrapper, ANNOTATION_FIELD)['traceable'] = True
    return wrapper


def inlinable(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    setattr(wrapper, ANNOTATION_FIELD, {**getattr(f, ANNOTATION_FIELD, {})})
    getattr(wrapper, ANNOTATION_FIELD)['inlinable'] = True
    return wrapper


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
            # before_dense = dispatch_keyset_before(torch._C.DispatchKey.Dense)
            # with torch._C._ExcludeDispatchKeyGuard(before_dense):
            return kernel(*args, **kwargs)

    lib.impl(name, wrapped, key)


def report_error_callback(op, key: str) -> None:
    if key == "Undefined":
        raise NotImplementedError(
            f"{op}: There were no Tensor inputs to this operator "
            f"(e.g. you passed an empty list of Tensors). If your operator is a "
            f"factory function (that is, it takes no Tensors and constructs "
            f"a new one), then please file an issue on GitHub."
        )
    if key == "Meta":
        raise NotImplementedError(
            f"{op}: when running with device='Meta' tensors: there is no "
            f"abstract impl registered for this op. Please register one by "
            f"defining the {op}.abstract staticmethod."
        )
    if key in ("CPU", "CUDA"):
        device = key.lower()
        raise NotImplementedError(
            f"{op}: when running with device='{device}' tensors: there is no "
            f"{device} impl registered for this {API}. Please register one by "
            f"defining the {op}.impl_{device} staticmethod."
        )
    raise NotImplementedError(
        f"{op}: No implementation for dispatch key {key}. It is likely "
        f"that we have not added this functionality yet, please either open an "
        f"issue or use the low-level torch.library APIs."
    )


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


def check_supported_schema(opdef, qualname):
    # We only support the following schemas, for now.
    # - functional
    # - auto_functionalizable.
    # For all others, we ask the user to go use the raw torch.library API.
    schema = qualname + opdef.schema
    import torch
    if torch._library.utils.is_functional_schema(schema):
        return
    # TODO(rzou):a put in final version
    import torch._higher_order_ops.auto_functionalize
    if torch._higher_order_ops.auto_functionalize.auto_functionalizable_schema(torch._C.parse_schema(schema)):
        return
    raise NotImplementedError(
        f"{API}: Tried to create an operator with unsupported schema "
        f"'{str(schema)}'. We support functional ops and mutable ops "
        f"where the outputs do not alias the inputs.")


def impl_autograd(lib, qualname, setup_context, backward, list_out=False):
    op = torch._library.utils.lookup_op(qualname)
    ns, name = qualname.split("::")

    # TODO: (1) autograd not found, (2) only support autograd for functional ops
    class MyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            # TODO: mark-dirty things
            with torch._C._AutoDispatchBelowAutograd():
                output = op(*args)
            if setup_context is not None:
                setup_context(ctx, args, output)
            if list_out:
                return tuple(output)
            return output

        @staticmethod
        def backward(ctx, *grads):
            assert backward is not None
            grad_inputs = backward(ctx, *grads)
            return grad_inputs

    lib.impl(name, MyFunction.apply, "Autograd")


def register_autograd(lib, name, opdef, op):
    impl_autograd(lib, op.__qualname__, getattr(opdef, "setup_backward", None), getattr(opdef, "backward", None))

class RegistersSubclassOnDefinition(type):
    def __new__(cls, name, bases, dct):
        result = super().__new__(cls, name, bases, dct)
        # This is the base class
        if name == "Operator":
            return result
        opoverload = register(result)
        result.opoverload = opoverload
        return result


class Operator(metaclass=RegistersSubclassOnDefinition):
    @classmethod
    def apply(cls, *args, **kwargs):
        return cls.opoverload(*args, **kwargs)

from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_call, triton_wrapper
import torch.utils._pytree as pytree


BackendImpl = Operator

# Must be a Pure Function
class Function:
    cache = {}

    @staticmethod
    def forward(*args):
        pass

    # @staticmethod
    # def post_forward(ctx, args, result):
    #     pass

    # @staticmethod
    # def backward(ctx, *grads):
    #     pass

    @classmethod
    def apply(cls, *args):
        if cls not in cls.cache:
            cls.cache[cls] = []
        flat_args, spec = pytree.tree_flatten(*args)
        for cached_spec, cached_run in cls.cache[cls]:
            if cached_spec == spec:
                return cached_run(*args)

        run = construct_run(cls, args)
        cls.cache[cls].append((spec, run))
        return run(*args)


def construct_run(cls, args):
    flat_args, spec = pytree.tree_flatten(*args)
    if any([a is None for a in flat_args]):
        raise NotImplementedError("NYI: Nones")
    schema_args = []
    for i, a in enumerate(flat_args):
        assert type(a) in SCHEMA_TYPE
        schema_args.append(f"{SCHEMA_TYPE[type(a)]} _{i}")
    # Assume only Tensors can be outputs
    schema = f'({", ".join(schema_args)}) -> Tensor[]'
    ns = mangle_module(cls.__module__)
    name = cls.__name__ + str(len(cls.cache))
    qualname = f"{ns}::{name}"
    lib = get_library_allowing_overwrite(ns, name)
    lib.define(f"{name}{schema}")

    input_spec = None
    output_spec = None
    single_return = False

    def forward(*args):
        nonlocal input_spec
        args = pytree.tree_unflatten(args, input_spec)
        result = cls.forward(*args)
        nonlocal single_return
        single_return = not isinstance(result, tuple)
        nonlocal output_spec
        flat_result, output_spec = pytree.tree_flatten(result)
        return flat_result

    lib.impl(name, forward, "CompositeExplicitAutograd")

    if hasattr(cls, "post_forward"):
        def post_forward(ctx, args, output):
            args = pytree.tree_unflatten(args, input_spec)
            output = pytree.tree_unflatten(output, output_spec)
            cls.post_forward(ctx, args, output)
    else:
        post_forward = None

    if hasattr(cls, "backward"):
        def backward(ctx, *grad_outputs):
            grad_outputs = pytree.tree_unflatten(grad_outputs, output_spec)
            if single_return:
                result = cls.backward(ctx, grad_outputs)
            else:
                result = cls.backward(ctx, *grad_outputs)
            flat_result, _ = pytree.tree_flatten(result)
            return tuple(flat_result)
    else:
        backward = None

    impl_autograd(lib, qualname, post_forward, backward, list_out=True)
    op = torch._library.utils.lookup_op(qualname)
    print(op._schema)

    def run(*args):
        nonlocal input_spec
        flat_args, input_spec = pytree.tree_flatten(args)
        flat_outs = op(*flat_args)
        result = pytree.tree_unflatten(flat_outs, output_spec)
        return result

    return run


def flatten_outputs(f):
    output_spec = None
    input_spec = None

    def inner(*args):
        if input_spec is not None:
            args = pytree.tree_unflatten(args, input_spec)
        result = f(*args)
        nonlocal output_spec
        flat_result, output_spec = pytree.tree_flatten(result)
        return flat_result

    def get_output_spec():
        return output_spec

    def set_input_spec(spec):
        nonlocal input_spec
        input_spec = spec

    return get_output_spec, inner, set_input_spec


SCHEMA_TYPE = {
    torch.Tensor: "Tensor",
    int: "SymInt",
    float: "Scalar",
    bool: "Scalar",
    str: "str",
}

from torch._custom_op.impl import infer_schema

class BlackBoxDef:
    def __init__(self, mutable_args, device_type, namespace, schema, fn):
        self._mutable_args = mutable_args
        self._namespace = namespace
        self._schema = schema
        self._impls = {}
        self._impls[device_type] = fn
        self._need_rebuild = True
        self._name = fn.__name__
        self._qualname = f"{self._namespace}::{self._name}"
        self._opoverload = None
        self._first_fn = fn
        self._needs_rebuild = True

    def impl(self, device_type):
        def inner(fn):
            self._impls[device_type] = fn
            self._needs_rebuild = True
        return inner

    def impl_abstract(self, fn):
        self._impls["abstract"] = fn
        self._needs_rebuild = True

    def impl_autograd(self, setup_context, backward):
        self._impls["autograd"] = (setup_context, backward)
        self._needs_rebuild = True

    def _rebuild(self):
        schema = infer_schema(self._first_fn)
        lib = get_library_allowing_overwrite(self._namespace, self._name)
        lib.define(f"{self._name}{schema}")
        for typ, fn in self._impls.items():
            if typ == "abstract":
                torch.library.impl_abstract(self._qualname, lib=lib)(fn)
            elif typ == "autograd":
                impl_autograd(lib, self._qualname, fn[0], fn[1])
            elif typ is None:
                torch.library.impl(self._qualname, "CompositeExplicitAutograd", lib=lib)(fn)
            else:
                torch.library.impl(self._qualname, typ, lib=lib)(fn)
        self._opoverload = torch._library.utils.lookup_op(self._qualname)
        self._needs_rebuild = False

    def __call__(self, *args, **kwargs):
        if self._needs_rebuild:
            self._rebuild()
        return self._opoverload(*args, **kwargs)


def def_blackbox(*, mutable_args, device_type=None, namespace=None, schema=None):
    assert namespace is None
    assert schema is None
    namespace = infer_namespace(stacklevel=1)

    def inner(fn):
        return BlackBoxDef(mutable_args, device_type, namespace, schema, fn)
    return inner


KEEP_ALIVE = []

def def_traceable(*, mutable_args, device_type=None, namespace=None, schema=None):
    assert namespace is None
    assert schema is None
    namespace = infer_namespace(stacklevel=1)

    def inner(fn):
        if inspect.isclass(fn) and issubclass(fn, torch.autograd.Function):
            schema = infer_schema(fn.forward)
            name = fn.__name__
            qualname = f"{namespace}::{name}"
            lib = torch.library.Library(namespace, "FRAGMENT")

            torch.library.define(qualname, schema, lib=lib)
            torch.library.impl(qualname, "CompositeExplicitAutograd", fn.forward, lib=lib)
            impl_autograd(lib, qualname, fn.setup_context, fn.backward)
            KEEP_ALIVE.append(lib)
            op = torch._library.utils.lookup_op(qualname)

            class Whatever:
                @staticmethod
                def apply(*args, **kwargs):
                    return op(*args, **kwargs)

            return Whatever
        else:
            schema = infer_schema(fn)
            name = fn.__name__
            qualname = f"{namespace}::{name}"
            lib = torch.library.Library(namespace, "FRAGMENT")

            torch.library.define(qualname, schema, lib=lib)
            torch.library.impl(qualname, "CompositeExplicitAutograd", fn, lib=lib)
            op = torch._library.utils.lookup_op(qualname)
            KEEP_ALIVE.append(lib)
            return op


    return inner
