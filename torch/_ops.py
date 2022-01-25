import torch._C

import contextlib
import ctypes
import sys
import types

import torch.jit
import torch._utils_internal

# Query `hasattr` only once.
_SET_GLOBAL_FLAGS = hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags')


@contextlib.contextmanager
def dl_open_guard():
    """
    Context manager to set the RTLD_GLOBAL dynamic linker flag while we open a
    shared library to load custom operators.
    """
    if _SET_GLOBAL_FLAGS:
        old_flags = sys.getdlopenflags()
        sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
    yield
    if _SET_GLOBAL_FLAGS:
        sys.setdlopenflags(old_flags)

# Each OpOverload object contains pointer to a a specific operator overload, a pointer to the parent `OpOverloadPacket` object.
# You can obtain an OpOverload object through attribute query on OpOverloadPacket.
class OpOverload:
    def __init__(self, overloadpacket, op, schema):
        self._op = op
        self._schema = schema
        self._overloadpacket = overloadpacket

    # it's a no-op since OpOverload object is immutable and must be unique for a given op overload.
    def __deepcopy__(self, memo=None):
        return self

    def __str__(self):
        return "OpOverload(op='{}.{}', overload='{}')".format(*self._schema.name.split("::"), self.overload_name)

    def __call__(self, *args, **kwargs):
        return self._op(*args, **kwargs or {})

    def __getattr__(self, key):
        return getattr(self._op, key)

    # `my_namespace::my_op`
    @property
    def name(self):
        return "{}.{}".format(*self._schema.name.split("::"))

    @property
    def overload_name(self):
        return self._schema.overload_name

    @property
    def overload_packet(self):
        return self._overloadpacket

    @property
    def op(self):
        return self._op

    # TODO: add more methods to expose information about input and output arguments

# OpOverloadPacket class contains pointer to a base unresolved operator that doesn't correspond to a specific operator
# You can obtain an OpOverload object through attribute query.
class OpOverloadPacket:
    def __init__(self, qualified_op_name, op_name, op):
        # These attributes are accessible on the object through the properties
        # defined below but are immutable
        self._qualified_op_name = qualified_op_name
        self._op_name = op_name
        self._op = op

    # it's a no-op since OpOverloadPacket object is immutable and must be unique for a given op.
    def __deepcopy__(self, memo=None):
        return self

    def __str__(self):
        return "OpOverloadPacket(op='{}.{}')".format(*self._qualified_op_name.split("::"))

    @property
    def qualified_op_name(self):
        return "{}.{}".format(*self._qualified_op_name.split("::"))

    @property
    def op_name(self):
        return self._op_name

    @property
    def op(self):
        return self._op

    def __getattr__(self, key):
        # It is not a valid op_name when __file__ is passed in
        if key == '__file__':
            return 'torch.ops'

        try:
            use_key = '' if key == 'default' else key
            # TODO: disallow access to overloads registered by JIT
            op_ = torch._C._get_operation_overload(self._qualified_op_name, use_key)
            schema = torch._C._get_schema(self._qualified_op_name, use_key)
            overload = OpOverload(self, op_, schema)
            # cache the overload object
            setattr(self, key, overload)
            return overload
        except RuntimeError:
            try:
                # This is added to maintain bc in case the user queries an attribute that exists on `self._op`
                # which used to be returned before instead of the OpOverloadPacket
                out = getattr(self._op, key)
                return out
            except AttributeError:
                raise AttributeError("'{}' object has no attribute '{}'".format(str(self), key)) from None

    def __call__(self, *args, **kwargs):
        # overloading __call__ to ensure torch.ops.foo.bar() is still callable from JIT
        # We save the function ptr as the `op` attribute on OpOverloadPacket to access it here.
        return self._op(*args, **kwargs or {})

# Resolution of torch.fn is different from torch.ops.aten.fn
# torch.fn uses the Python argparser, matches with the appropriate schema, and calls into the unboxed version of the method
# torch.ops.aten.fn resolution is done via the mechanism defined in JIT. JIT creates a stack of all the overloads and
# then tries to match the correct one at runtime and always calls into the boxed version of the method
# Autograd codegen creates VariableType, TracerType, inplace or view type and python bindings
# Aten codegen generates tensor methods for the the tensor class

# _OpNamespace is a subclass of ModuleType because the torch script
# allows attribute lookups on modules only. Since we want torch.ops.foo.bar()
# to work from script, we need to ensure ops and foo are modules
class _OpNamespace(types.ModuleType):
    """
    An op namespace to dynamically bind Operators into Python.

    Say a user has created a custom Operator called "my_namespace::my_op". To
    call this op, the user will write torch.ops.my_namespace.my_op(...).
    At startup, this operation will not yet be bound into Python. Instead, the
    following sequence of magic tricks will occur:
    1. `torch.ops.my_namespace` will invoke the `__getattr__` magic method
       on the `torch.ops` object, which will create a new `_OpNamespace`
       object called `my_namespace` and set it as an attribute on the `ops`
       object.
    2. `torch.ops.my_namespace.my_op` will then invoke `__getattr__` on
       the `my_namespace` object, which will retrieve the operation via
       `torch.get_operation`, a function bound from C++, and then in a similar
       fashion bind this new object onto the `my_namespace` object.
    3. `torch.ops.my_namespace.my_op(...)` then calls this new operation
        and subsequent accesses will incur no further lookup (the namespace and
        operation will already exist).
    """
    def __init__(self, name):
        super(_OpNamespace, self).__init__('torch.ops.' + name)
        self.name = name

    def __getattr__(self, op_name):
        # It is not a valid op_name when __file__ is passed in
        if op_name == '__file__':
            return 'torch.ops'
        # Get the op `my_namespace::my_op` if available. This will also check
        # for overloads and raise an exception if there are more than one.
        namespace_name = self.name
        qualified_op_name = '{}::{}'.format(namespace_name, op_name)
        op = torch._C._jit_get_operation(qualified_op_name)

        # let the script frontend know that op is identical to the builtin op
        # with qualified_op_name
        torch.jit._builtins._register_builtin(op, qualified_op_name)
        op.__module__ = self.__module__ + "." + namespace_name
        # opoverloadpacket = OpOverloadPacket(qualified_op_name, op_name, op)
        # opoverloadpacket.__module__ = self.__module__ + "." + namespace_name
        # cache the opoverloadpacket to ensure that each op corresponds to
        # a unique OpOverloadPacket object
        # setattr(self, op_name, opoverloadpacket)
        setattr(self, op_name, op)
        return op

class _Ops(types.ModuleType):
    __file__ = '_ops.py'

    def __init__(self):
        super(_Ops, self).__init__('torch.ops')
        self.loaded_libraries = set()

    def __getattr__(self, name):
        # Here we are creating `torch.ops.my_namespace`
        namespace = _OpNamespace(name)
        setattr(self, name, namespace)
        return namespace

    def load_library(self, path):
        """
        Loads a shared library from the given path into the current process.

        The library being loaded may run global initialization code to register
        custom operators with the PyTorch JIT runtime. This allows dynamically
        loading custom operators. For this, you should compile your operator
        and the static registration code into a shared library object, and then
        call ``torch.ops.load_library('path/to/libcustom.so')`` to load the
        shared object.

        After the library is loaded, it is added to the
        ``torch.ops.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.

        Args:
            path (str): A path to a shared library to load.
        """
        if sys.executable == "torch_deploy":
            return

        path = torch._utils_internal.resolve_library_path(path)
        with dl_open_guard():
            # Import the shared library into the process, thus running its
            # static (global) initialization code in order to register custom
            # operators with the JIT.
            ctypes.CDLL(path)
        self.loaded_libraries.add(path)

# The ops "namespace"
ops = _Ops()
