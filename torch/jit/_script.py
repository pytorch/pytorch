"""TorchScript

This module contains functionality to support the JIT's scripting frontend, notably:
    - torch.jit.script

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""
import collections
import copy
import enum
import functools
import inspect
import pickle
import warnings
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import torch
import torch._jit_internal as _jit_internal
from torch._classes import classes
from torch._jit_internal import _qualified_name
from torch.jit._builtins import _register_builtin
from torch.jit._fuser import _graph_for, _script_method_graph_for

from torch.jit._monkeytype_config import (
    JitTypeTraceConfig,
    JitTypeTraceStore,
    monkeytype_trace,
)
from torch.jit._recursive import (
    _compile_and_register_class,
    infer_methods_to_compile,
    ScriptMethodStub,
    wrap_cpp_module,
)
from torch.jit._state import (
    _enabled,
    _set_jit_function_cache,
    _set_jit_overload_cache,
    _try_get_jit_cached_function,
    _try_get_jit_cached_overloads,
)
from torch.jit.frontend import get_default_args, get_jit_class_def, get_jit_def
from torch.nn import Module
from torch.overrides import (
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)
from torch.package import PackageExporter, PackageImporter
from torch.utils import set_module
from ._serialization import validate_map_location

type_trace_db = JitTypeTraceStore()  # DB to hold all call traces from MonkeyType

torch._C.ScriptMethod.graph_for = _script_method_graph_for  # type: ignore[attr-defined]
torch._C.ScriptFunction.graph_for = _graph_for  # type: ignore[attr-defined]
ScriptFunction = torch._C.ScriptFunction
ScriptFunction.__doc__ = """
Functionally equivalent to a :class:`ScriptModule`, but represents a single
function and does not have any attributes or Parameters.
"""
set_module(ScriptFunction, "torch.jit")


# Throws an error if a jit function is pickled.
# Helps to avoid Python crashes for Python versions 3.9.5 + when protocol 0 or 1 is given as an argument.
def _reduce(cls):
    raise pickle.PickleError("ScriptFunction cannot be pickled")


ScriptFunction.__reduce__ = _reduce  # type: ignore[assignment]


if _enabled:
    Attribute = collections.namedtuple("Attribute", ["value", "type"])
else:

    def Attribute(value, type):  # type: ignore[no-redef]
        return value


Attribute.__doc__ = """
    This method is a pass-through function that returns `value`, mostly
    used to indicate to the TorchScript compiler that the left-hand side
    expression is a class instance attribute with type of `type`. Note that
    `torch.jit.Attribute` should only be used in `__init__` method of `jit.ScriptModule`
    subclasses.

    Though TorchScript can infer correct type for most Python expressions, there are some cases where
    type inference can be wrong, including:

    - Empty containers like `[]` and `{}`, which TorchScript assumes to be container of `Tensor`
    - Optional types like `Optional[T]` but assigned a valid value of type `T`, TorchScript would assume
      it is type `T` rather than `Optional[T]`

    In eager mode, it is simply a pass-through function that returns `value`
    without other implications.

    Example:

    .. testcode::

        import torch
        from typing import Dict

        class AttributeModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.foo = torch.jit.Attribute(0.1, float)

                # we should be able to use self.foo as a float here
                assert 0.0 < self.foo

                self.names_ages = torch.jit.Attribute({}, Dict[str, int])
                self.names_ages["someone"] = 20
                assert isinstance(self.names_ages["someone"], int)

        m = AttributeModule()
        # m will contain two attributes
        # 1. foo of type float
        # 2. names_ages of type Dict[str, int]

    .. testcleanup::

        del AttributeModule
        del m

    Note: it's now preferred to instead use type annotations instead of `torch.jit.Attribute`:

    .. testcode::

        import torch
        from typing import Dict

        class AttributeModule(torch.nn.Module):
            names: Dict[str, int]

            def __init__(self):
                super().__init__()
                self.names = {}

        m = AttributeModule()

    .. testcleanup::

        del AttributeModule
        del m

    Args:
        value: An initial value to be assigned to attribute.
        type: A Python type

    Returns:
        Returns `value`
"""


def _get_type_trace_db():
    # This is a private API. Use of this for external purposes is discouraged.
    return type_trace_db


# Gets a function from the name of a method on a type
def _get_function_from_type(cls, name):
    return getattr(cls, name, None)


# ScriptClasses must be new-style classes because we construct them using their
# __new__ method.
def _is_new_style_class(cls):
    if hasattr(cls, "__class__"):
        return "__dict__" in dir(cls) or hasattr(cls, "__slots__")


# These OrderedDictWrapper classes replace the actual OrderedDicts in
# module with versions that get/set properties inside of Module.
# This allows us to reuse most of nn.Module while still storing the
# data in C++.
# Each OrderedDict needs to support:
#  x not in view
#  x in view
#  view[name] = ...
#  view.values()
#  del view[name]
#  view.items()
#  view.keys()
#  len(view)


class OrderedDictWrapper:
    def __init__(self, _c):
        self._c = _c

    def keys(self):
        return [k for k, v in self.items()]

    def values(self):
        return [v for k, v in self.items()]

    def __len__(self):
        return len(self.values())

    def __delitem__(self, k):
        raise RuntimeError("cannot delete methods or parameters of a script module")

    def items(self):
        return self._c.items()

    def __setitem__(self, k, v):
        if k not in self:
            raise RuntimeError(
                f"Can't add a new parameter after ScriptModule construction. Tried to add '{k}"
            )
        self._c.setattr(k, v)

    def __contains__(self, k):
        return self._c.contains(k)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        return self._c.getattr(k)


class OrderedModuleDict(OrderedDictWrapper):
    def __init__(self, module, python_dict):
        super().__init__(torch._C.ModuleDict(module))
        # contains _both_ script modules and non-script python-only modules

        # because script modules are subclassed in python and the
        # C++ Module class will not hold references to them,
        # to ensure that you always get the same python value here
        # we store it in the python dict as well
        self._python_modules = python_dict

    def items(self):
        r = self._python_modules.items()
        return r

    def __contains__(self, k):
        return k in self._python_modules

    def __setitem__(self, k, v):
        # Cases where sub-module can be re-assigned after ScriptModule construction
        # 1. If the attr is an module interface type, it's guaranteed that the module is
        #    not inlined in the graph, so it's safe to swap a new ScriptModule in.
        # 2. if the new value if a ScriptModule with the same JIT type, IR won't change
        #    and it's legit to swap a new module in.
        # In these two cases we allow swapping a new scripted module and update the
        # corresponding python module dict to keep sync.
        # Note: the value to be swapped in has to be ScriptModule instead of nn.Module,
        # otherwise it's illegal and we throw error.
        if isinstance(v, ScriptModule):
            self._c.setattr(k, v)
            self._python_modules[k] = v
        else:
            raise RuntimeError(
                "Cannot re-assign modules in a ScriptModule with non-scripted "
                "module, tried to replace existing module '{}': {}".format(k, v)
            )

    def __getitem__(self, k):
        return self._python_modules[k]


# For each user-defined class that subclasses ScriptModule, this meta-class:
# (1) finds all the methods annotated with @script_method in a ScriptModule and
#     removes them from the class attributes
# (2) puts a wrapper around the class's __init__ method to recursively compile
#     all of the script_methods with the module after the original __init__ has
#     run. This has to occur after the user-defined __init__ so that submodules and
#     parameters are initialized _before_ the script compiler resolve references to
#     `self.param` or `self.module`.
class ScriptMeta(type):
    def __init__(cls, name, bases, attrs):  # noqa: B902
        # Aggregate all the ScriptMethods and constants from superclasses
        cls._methods: Dict[str, Any] = {}
        cls._constants_set = set(getattr(cls, "__constants__", ()))
        for base in reversed(bases):
            for k, v in getattr(base, "_methods", {}).items():
                cls._methods[k] = v
            base_constants: Set = getattr(base, "_constants_set", set())
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
            return super().__init__(name, bases, attrs)

        original_init = getattr(cls, "__init__", lambda self: None)

        @functools.wraps(original_init)
        def init_then_script(self, *args, **kwargs):
            num_methods = len(cls._methods)
            original_init(self, *args, **kwargs)
            added_methods_in_init = len(cls._methods) > num_methods

            if type(self) == cls:

                def make_stubs(module):
                    cls = type(module)
                    if hasattr(cls, "_methods"):
                        return [v for k, v in sorted(cls._methods.items())]
                    else:
                        return infer_methods_to_compile(module)

                self.__dict__[
                    "_actual_script_module"
                ] = torch.jit._recursive.create_script_module(
                    self, make_stubs, share_types=not added_methods_in_init
                )

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

        cls.__init__ = init_then_script  # type: ignore[misc]
        super().__init__(name, bases, attrs)


class _CachedForward:
    def __get__(self, obj, cls):
        return self.__getattr__("forward")  # type: ignore[attr-defined]


class ScriptWarning(Warning):
    pass


def script_method(fn):
    if not _enabled:
        return fn
    # NOTE: we need to traverse two frames here because the meta-class frame
    # for ScriptModule will be present, as opposed to invoking @script on a
    # a function or invoking define() on a CompilationUnit.
    # The stack will look like:
    #
    # 0. createResolutionCallback()
    # 1. script_method()
    # 2. ScriptModule metaclass frame
    # 3. Surrounding scope
    #
    # createResolutionCallback internally adds 1 to get us to the scope of this
    # function (the calling function). Adding 2 gets us to the proper surrounding scope.
    _rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=2)
    ast = get_jit_def(fn, fn.__name__, self_name="ScriptModule")
    return ScriptMethodStub(_rcb, ast, fn)


class ConstMap:
    def __init__(self, const_mapping):
        self.const_mapping = const_mapping

    def __getattr__(self, attr):
        return self.const_mapping[attr]


def unpackage_script_module(
    importer: PackageImporter, script_module_id: str
) -> torch.nn.Module:
    """
    Called by ``torch.package.PackageImporter``'s Pickler's ``persistent_load`` function.
    Performs work of loading and returning a ScriptModule from a ``torch.package`` archive.
    """
    if not isinstance(importer.zip_reader, torch._C.PyTorchFileReader):
        raise RuntimeError(
            "Loading ScriptObjects from a PackageImporter created from a "
            "directory is not supported. Use a package archive file instead."
        )
    cu = torch._C.CompilationUnit()
    cpp_module = torch._C._import_ir_module_from_package(
        cu,
        importer.zip_reader,
        importer.storage_context,
        validate_map_location(importer.last_map_location),
        script_module_id,
    )
    return wrap_cpp_module(cpp_module)


if _enabled:
    _magic_methods = [
        "__iter__",
        "__len__",
        "__neg__",
        "__mul__",
        "__contains__",
        "__add__",
        "__sub__",
        "__pow__",
        "__truediv__",
        "__mod__",
        "__ne__",
        "__eq__",
        "__lt__",
        "__gt__",
        "__le__",
        "__ge__",
        "__and__",
        "__or__",
        "__xor__",
        "__getitem__",
        "__setitem__",
        "__call__",
        "__int__",
        "__float__",
        "__bool__",
        "__str__",
        "__enter__",
        "__exit__",
    ]

    class RecursiveScriptClass:
        """
        An analogue of RecursiveScriptModule for regular objects that are not modules.
        This class is a wrapper around a torch._C.ScriptObject that represents an instance
        of a TorchScript class and allows it to be used in Python.

        Attributes:
            _c [torch._C.ScriptObject]: The C++ object to which attribute lookups and method
                calls are forwarded.
            _props [Dict[str, property]]: A dictionary of properties fetched from self._c and
                exposed on this wrppaer.
        """

        def __init__(self, cpp_class):
            super().__init__()
            self.__dict__["_initializing"] = True
            self._c = cpp_class

            # Add wrapped object's properties to this class instance.
            self._props = {
                prop.name: property(prop.getter, prop.setter)
                for prop in self._c._properties()
            }

            self.__dict__["_initializing"] = False

        def __getattr__(self, attr):
            if "_initializing" in self.__dict__ and self.__dict__["_initializing"]:
                return super().__getattr__(attr)  # type: ignore[misc]

            if attr in self._props:
                return self._props[attr].fget()  # type: ignore[call-arg, misc]

            return getattr(self._c, attr)

        def __setattr__(self, attr, value):
            if "_initializing" in self.__dict__ and self.__dict__["_initializing"]:
                return super().__setattr__(attr, value)

            if attr in self._props:
                return self._props[attr].fset(value)  # type: ignore[call-arg, misc]

            setattr(self._c, attr, value)

        # Delegate calls to magic methods like __len__ to the C++ module backing the
        # RecursiveScriptClass.
        def forward_magic_method(self, method_name, *args, **kwargs):
            if not self._c._has_method(method_name):
                raise TypeError()

            self_method = self.__getattr__(method_name)
            return self_method(*args, **kwargs)

        def __getstate__(self):
            raise pickle.PickleError("ScriptClasses cannot be pickled")

        def __iadd__(self, other):
            if self._c._has_method("__iadd__"):
                return self.forward_magic_method("__iadd__", other)
            else:
                return self.forward_magic_method("__add__", other)

    for method_name in _magic_methods:

        def method_template(self, *args, **kwargs):
            return self.forward_magic_method(method_name, *args, **kwargs)

        setattr(RecursiveScriptClass, method_name, method_template)

    # this is a Python 'non-data descriptor' that causes the first access
    # to ScriptModule's forward to look up the forward method and stash
    # it in the objects dict. Due to the standard rules for attribute lookup,
    # subsequent lookups will just directly return the previously looked up method.
    # This is necessary because nn.Module defines forward as a method. If we
    # did nothing, __getattr__ would not be called. Instead we'd get nn.Module.forward
    # which always throws an exception.

    class ScriptModule(Module, metaclass=ScriptMeta):
        r"""
        A wrapper around C++ ``torch::jit::Module``. ``ScriptModule``\s
        contain methods, attributes, parameters, and
        constants. These can be accessed the same way as on a normal ``nn.Module``.
        """
        __jit_unused_properties__ = [
            "code",
            "code_with_constants",
            "graph",
            "inlined_graph",
            "original_name",
        ]

        def __init__(self):
            super().__init__()

        forward: Callable[..., Any] = _CachedForward()  # type: ignore[assignment]

        def __getattr__(self, attr):
            if "_actual_script_module" not in self.__dict__:
                return super().__getattr__(attr)
            return getattr(self._actual_script_module, attr)

        def __setattr__(self, attr, value):
            if "_actual_script_module" not in self.__dict__:
                # Unwrap torch.jit.Attribute into a regular setattr + record
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
                return super().__setattr__(attr, value)

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
            rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=1)
            ast = torch._C._parse_source_def(src)
            self._methods[ast.name().name] = ScriptMethodStub(rcb, ast, None)

        def _replicate_for_data_parallel(self):
            return self._actual_script_module._replicate_for_data_parallel()

        def __reduce_package__(self, exporter: PackageExporter):
            """
            Called by ``torch.package.PackageExporter``'s Pickler's ``persistent_id`` when
            saving TorchScript objects. Performs act of saving a ScriptModule inside of
            a ``torch.package`` archive.

            Returns method to load the ScriptModule from a ``torch.package.PackageImporter``'s
            Pickler's ``persistent_load`` function.
            """
            script_module_id = exporter.get_unique_id()
            exporter.script_module_serializer.serialize(self._c, int(script_module_id))
            return (unpackage_script_module, (script_module_id,))

    class RecursiveScriptModule(ScriptModule):
        # XXX: RecursiveScriptModule inherits from ScriptModule for the sole
        # reason that it retains the existing isinstance(ScriptModule)
        # behavior.
        r"""
        The core data structure in TorchScript is the ``ScriptModule``. It is an
        analogue of torch's ``nn.Module`` and represents an entire model as a tree of
        submodules. Like normal modules, each individual module in a ``ScriptModule`` can
        have submodules, parameters, and methods. In ``nn.Module``\s methods are implemented
        as Python functions, but in ``ScriptModule``\s methods are implemented as
        TorchScript functions, a statically-typed subset of Python that contains all
        of PyTorch's built-in Tensor operations. This difference allows your
        ``ScriptModule``\s code to run without the need for a Python interpreter.

        ``ScriptModule``\s should not be created manually, instead use
        either :func:`tracing <torch.jit.trace>` or :func:`scripting <torch.jit.script>`.
        Tracing and scripting can be applied incrementally and :ref:`composed as necessary <Types>`.

        * Tracing records the tensor operations as executed with a set of example inputs and uses these
          operations to construct a computation graph. You can use the full dynamic behavior of Python with tracing,
          but values other than Tensors and control flow aren't captured in the graph.

        * Scripting inspects the Python code of the model
          and compiles it to TorchScript. Scripting allows the use of many `types`_ of values and supports dynamic control flow.
          Many, but not all features of Python are supported by the compiler, so changes to the source code may be necessary.
        """
        _disable_script_meta = True

        def __init__(self, cpp_module):
            self.__dict__["_initializing"] = True
            self._c = cpp_module
            super().__init__()
            # Delete the 'training' attribute set up by `Module.__init__`. It
            # will get set on the underlying cpp module, so we delete it here
            # to avoid this version shadowing the cpp module version.
            delattr(self, "training")

        @staticmethod
        def _construct(cpp_module, init_fn):
            """
            Construct a RecursiveScriptModule that's ready for use. PyTorch
            code should use this to construct a RecursiveScriptModule instead
            of instead of calling `__init__` directly, as it makes sure the
            object is properly finalized (and in the future, we may take
            control of how the RecursiveScriptModule instance is created).

            Args:
                cpp_module:  The C++ Module that will hold the actual state of
                             this RecursiveScriptModule instance.
                init_fn:  Lambda that initializes the RecursiveScriptModule passed to it.
            """
            script_module = RecursiveScriptModule(cpp_module)
            init_fn(script_module)

            # Finalize the ScriptModule: replace the nn.Module state with our
            # custom implementations and flip the _initializing bit.
            RecursiveScriptModule._finalize_scriptmodule(script_module)
            return script_module

        @staticmethod
        def _finalize_scriptmodule(script_module):
            script_module._parameters = OrderedDictWrapper(
                torch._C.ParameterDict(script_module._c)
            )
            script_module._buffers = OrderedDictWrapper(
                torch._C.BufferDict(script_module._c)
            )
            script_module._modules = OrderedModuleDict(
                script_module._c, script_module._modules
            )
            script_module._initializing = False

        def _reconstruct(self, cpp_module):
            """
            Re-construct an instance of RecursiveScriptModule using an instance of a C++ module.

            Args:
                cpp_module: The C++ module that this RecursiveScriptModule will be rebuilt around.
            """
            self.__init__(cpp_module)  # type: ignore[misc]

            # Copy the concrete type from the C++ module to this ScriptModule.
            self._concrete_type = torch._C.ConcreteModuleType.from_jit_type(
                self._c._type()
            )

            # Copy submodules from the C++ module to this ScriptModule.
            modules = {}
            for name, cpp_module in torch._C.ModuleDict(self._c).items():
                modules[name] = wrap_cpp_module(cpp_module)
            self._modules = OrderedModuleDict(self._c, modules)  # type: ignore[assignment]

            # Copy parameters and buffers.
            self._parameters = OrderedDictWrapper(torch._C.ParameterDict(self._c))  # type: ignore[assignment]
            self._buffers = OrderedDictWrapper(torch._C.BufferDict(self._c))  # type: ignore[assignment]

            # Get rid of the functions from the old C++ module.
            self.__dict__ = {
                k: v
                for k, v in self.__dict__.items()
                if not isinstance(v, torch._C.ScriptMethod)
            }
            self.__dict__["_initializing"] = False

        @property
        def graph(self):
            r"""
            Returns a string representation of the internal graph for the
            ``forward`` method. See :ref:`interpreting-graphs` for details.
            """
            return self._c._get_method("forward").graph

        @property
        def inlined_graph(self):
            r"""
            Returns a string representation of the internal graph for the
            ``forward`` method. This graph will be preprocessed to inline all function and method calls.
            See :ref:`interpreting-graphs` for details.
            """
            return self.forward.inlined_graph  # type: ignore[attr-defined]

        @property
        def code(self):
            r"""
            Returns a pretty-printed representation (as valid Python syntax) of
            the internal graph for the ``forward`` method. See
            :ref:`inspecting-code` for details.
            """
            return self.forward.code  # type: ignore[attr-defined]

        @property
        def code_with_constants(self):
            r"""
            Returns a tuple of:

            [0] a pretty-printed representation (as valid Python syntax) of
            the internal graph for the ``forward`` method. See `code`.
            [1] a ConstMap following the CONSTANT.cN format of the output in [0].
            The indices in the [0] output are keys to the underlying constant's values.

            See :ref:`inspecting-code` for details.
            """
            r = self.forward.code_with_constants  # type: ignore[attr-defined]
            return (r[0], ConstMap(r[1]))

        def save(self, f, **kwargs):
            r"""
            save(f, _extra_files={})

            See :func:`torch.jit.save <torch.jit.save>` for details.
            """
            return self._c.save(str(f), **kwargs)

        def _save_for_lite_interpreter(self, *args, **kwargs):
            r"""
            _save_for_lite_interpreter(f)

            Add (or update) the bytecode session to the script model. The updated model is used
            in lite interpreter for mobile applications.

            Args:
                f: a string containing a file name.
                _extra_files: Map from filename to contents which will be stored as part of 'f'.

            """
            return self._c._save_for_mobile(*args, **kwargs)

        def _save_to_buffer_for_lite_interpreter(self, *args, **kwargs):
            return self._c._save_to_buffer_for_mobile(*args, **kwargs)

        def save_to_buffer(self, *args, **kwargs):
            return self._c.save_to_buffer(*args, **kwargs)

        def get_debug_state(self, *args, **kwargs):
            return self._c.get_debug_state()

        def extra_repr(self):
            return f"original_name={self.original_name}"

        def graph_for(self, *args, **kwargs):
            return self.forward.graph_for(self, *args, **kwargs)  # type: ignore[attr-defined]

        @property
        def original_name(self):
            if type(self) == str(self._c._type().name()):
                return ""
            return str(self._c._type().name())

        def define(self, src):
            # We use frames_up=1 to get to the proper surrounding scope. The stack
            # will look like:
            # 0. createResolutionCallback
            # 1. define()
            # 2. surrounding scope.
            #
            # createResolutionCallback internally adds 1 to get us to our frame, then
            # we add 1 to get to the proper surrounding scope.
            rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=1)
            self._c._define(self._concrete_type, src, rcb)

        def __getattr__(self, attr):
            if "_initializing" not in self.__dict__:
                raise RuntimeError(
                    "ScriptModule has not been initialized, did you forget to call super's init?"
                )

            if self._initializing:
                return super().__getattr__(attr)

            # _modules check is before hasattr since modules are included as attributes in _c,
            # but we want to get the python wrapper from _modules instead of the raw _c object.
            if attr in self._modules:
                return self._modules[attr]
            elif self._c.hasattr(attr):
                return self._c.getattr(attr)
            elif self._c._has_method(attr):
                script_method = self._c._get_method(attr)
                # cache method so future calls do not go through __getattr__
                # to improve invocation performance
                self.__dict__[attr] = script_method
                return script_method

            return super().__getattr__(attr)

        def __setattr__(self, attr, value):
            if self._initializing:
                return super().__setattr__(attr, value)

            if attr in self._modules:
                self._modules[attr] = value
            elif self._c.hasattr(attr):
                self._c.setattr(attr, value)
            elif (
                hasattr(self, "_concrete_type")
                and attr in self._concrete_type.get_constants().keys()
            ):
                # TODO: we don't have _concrete_type set after load(), and in general we lose constant information.
                # We should encode constants as class type attributes (or something) so it persists across save/load.
                raise AttributeError(
                    f"Cannot mutate TorchScript constant value: '{attr}'. Value: '{value}'"
                )
            else:
                # We allow setting Python attributes on the ScriptModule, for
                # when people want to stash some convenience info on it.
                # TODO: it's possible that the following is confusing:
                #   s = torch.jit.script(...)
                #   s.python_attr = ...
                #   s.save()   <--- this doesn't have `python_attr`
                # It's fairly trivial to save enough info to warn in this case.
                return super().__setattr__(attr, value)

        def __copy__(self):
            return torch.jit._recursive.wrap_cpp_module(copy.copy(self._c))

        def __deepcopy__(self, memo):
            return torch.jit._recursive.wrap_cpp_module(copy.deepcopy(self._c, memo))

        # Python magic methods do method lookups on an object's class type, instead of looking up
        # the method defines on the class instance. In order to continue to expose the magic methods
        # of builtin-containers (ModuleList, Sequential, ModuleDict) to Python, we
        # define magic methods here as a shim to the correct attribute.
        def forward_magic_method(self, method_name, *args, **kwargs):
            self_method = getattr(self, method_name)
            if getattr(self_method, "__func__", None) == getattr(
                RecursiveScriptModule, method_name
            ):
                raise NotImplementedError()
            return self_method(*args, **kwargs)

        def __iter__(self):
            return self.forward_magic_method("__iter__")

        def __getitem__(self, idx):
            return self.forward_magic_method("__getitem__", idx)

        def __len__(self):
            return self.forward_magic_method("__len__")

        def __contains__(self, key):
            return self.forward_magic_method("__contains__", key)

        # dir is defined by the base nn.Module, so instead of throwing if
        # it is not overridden, we call into the nn.Module __dir__ method
        def __dir__(self):
            self_method = self.__dir__
            if (
                self_method.__func__  # type: ignore[attr-defined]
                == _get_function_from_type(RecursiveScriptModule, "__dir__")
            ):
                return super().__dir__()
            return self_method()

        # to resolve bool(value), Python looks if __bool__ is defined then __iter__
        # is defined then returns true for classes. Since __iter__() on this
        # class throws if it isn't overridden, we define __bool__ to preserve default behavior
        def __bool__(self):
            self_method = self.__bool__
            if (
                self_method.__func__  # type: ignore[attr-defined]
                == _get_function_from_type(RecursiveScriptModule, "__bool__")
            ):
                return True
            return self_method()

        def _replicate_for_data_parallel(self):
            # we have to initialize ScriptModule properly so that
            # it works with pybind11
            def init_fn(script_module):
                # Don't do anything here, we'll initialize the ScriptModule below
                return

            return RecursiveScriptModule._construct(
                self._c._replicate_for_data_parallel(), init_fn
            )

    # Need to copy all RecursiveScriptModule methods to ScriptModule.
    #
    # This is because `super().foo()` does not use
    # `__getattr__` to look up `foo`. So we need to make each method available on
    # the ScriptModule manually.
    for name, item in RecursiveScriptModule.__dict__.items():
        if not callable(item) and not isinstance(item, property):
            continue
        if name.startswith("__") or hasattr(ScriptModule, name):
            continue
        # We can copy over the implementation wholesale because besides the
        # `super()` thing above, ScriptModule behaves exactly like
        # RecursiveScriptModule
        setattr(ScriptModule, name, item)

    def _get_methods(cls):
        import inspect

        # In Python 3 unbound methods are functions, but in Python 2 they are methods
        return inspect.getmembers(
            cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x)
        )

    _compiled_methods_allowlist = {
        "forward",
        "register_buffer",
        "register_parameter",
        "register_module",
        "add_module",
        "_apply",
        "apply",
        "cuda",
        "cpu",
        "to",
        "type",
        "float",
        "double",
        "half",
        "state_dict",
        "_save_to_state_dict",
        "load_state_dict",
        "_load_from_state_dict",
        "_named_members",
        "parameters",
        "named_parameters",
        "buffers",
        "named_buffers",
        "children",
        "named_children",
        "modules",
        "named_modules",
        "zero_grad",
        "share_memory",
        "_get_name",
        "extra_repr",
        "_slow_forward",
        "_tracing_name",
        "eval",
        "train",
        "get_extra_state",
        "set_extra_state",
    }

    def _make_fail(name):
        def fail(self, *args, **kwargs):
            raise RuntimeError(name + " is not supported on ScriptModules")

        return fail

    for name, method in _get_methods(torch.nn.Module):
        if name.startswith("__") or name.endswith("_call_impl"):
            continue
        if (
            name not in RecursiveScriptModule.__dict__
            and name not in _compiled_methods_allowlist
        ):
            setattr(RecursiveScriptModule, method.__name__, _make_fail(name))


else:
    # TODO MAKE SURE THAT DISABLING WORKS
    class RecursiveScriptClass:  # type: ignore[no-redef]
        pass

    class ScriptModule(torch.nn.Module):  # type: ignore[no-redef]
        def __init__(self, arg=None):
            super().__init__()

    class RecursiveScriptModule(ScriptModule):  # type: ignore[no-redef]
        def __init__(self, arg=None):
            super().__init__()


def call_prepare_scriptable_func_impl(obj, memo):
    if not isinstance(obj, torch.nn.Module):
        return obj

    obj_id = id(obj)

    # If obj_id is in memo, obj has already been prepared or is being
    # prepared in another call up the stack.
    if obj_id in memo:
        return memo[id(obj)]

    obj = obj.__prepare_scriptable__() if hasattr(obj, "__prepare_scriptable__") else obj  # type: ignore[operator]
    # Record obj in memo to avoid infinite recursion in the case of cycles in the module
    # hierarchy when recursing below.
    memo[obj_id] = obj

    new_obj_dict = {}

    for name, sub_module in obj.__dict__.items():
        if name == "_modules":
            for k, v in sub_module.items():
                sub_module[k] = call_prepare_scriptable_func_impl(v, memo)
            new_obj_dict[name] = sub_module
        elif isinstance(sub_module, torch.nn.Module) and not isinstance(
            sub_module, ScriptModule
        ):
            new_obj_dict[name] = call_prepare_scriptable_func_impl(sub_module, memo)
        else:
            new_obj_dict[name] = sub_module

    for k, v in new_obj_dict.items():
        obj.__dict__[name] = v

    return obj


def call_prepare_scriptable_func(obj):
    memo: Dict[int, torch.nn.Module] = {}
    return call_prepare_scriptable_func_impl(obj, memo)


def create_script_dict(obj):
    """
    Create a ``torch._C.ScriptDict`` instance with the data from ``obj``.

    Args:
        obj (dict): The Python dictionary that is used to initialize the ``ScriptDict``
                    returned by this function.

    Returns:
        An instance of ``torch._C.ScriptDict`` that has the same data as ``obj``
        and can be passed between Python and TorchScript with reference semantics and
        zero copy overhead.
    """
    return torch._C.ScriptDict(obj)  # type: ignore[attr-defined]


def create_script_list(obj, type_hint=None):
    """
    Create a ``torch._C.ScriptList`` instance with the data from ``obj``.
    Args:
        obj (dict): The Python list that is used to initialize the ``ScriptList``
                    returned by this function.
    Returns:
        An instance of ``torch._C.ScriptList`` that has the same data as ``obj``
        and can be passed between Python and TorchScript with reference semantics and
        zero copy overhead.
    """
    return torch._C.ScriptList(obj)  # type: ignore[attr-defined]


def script(
    obj,
    optimize=None,
    _frames_up=0,
    _rcb=None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
):
    r"""
    Scripting a function or ``nn.Module`` will inspect the source code, compile
    it as TorchScript code using the TorchScript compiler, and return a :class:`ScriptModule` or
    :class:`ScriptFunction`. TorchScript itself is a subset of the Python language, so not all
    features in Python work, but we provide enough functionality to compute on
    tensors and do control-dependent operations. For a complete guide, see the
    :ref:`language-reference`.

    Scripting a dictionary or list copies the data inside it into a TorchScript instance than can be
    subsequently passed by reference between Python and TorchScript with zero copy overhead.

    ``torch.jit.script`` can be used as a function for modules, functions, dictionaries and lists
     and as a decorator ``@torch.jit.script`` for :ref:`torchscript-classes` and functions.

    Args:
        obj (Callable, class, or nn.Module):  The ``nn.Module``, function, class type,
                                                  dictionary, or list to compile.
        example_inputs (Union[List[Tuple], Dict[Callable, List[Tuple]], None]): Provide example inputs
            to annotate the arguments for a function or ``nn.Module``.

    Returns:
        If ``obj`` is ``nn.Module``, ``script`` returns
        a :class:`ScriptModule` object. The returned :class:`ScriptModule` will
        have the same set of sub-modules and parameters as the
        original ``nn.Module``. If ``obj`` is a standalone function,
        a :class:`ScriptFunction` will be returned. If ``obj`` is a ``dict``, then
        ``script`` returns an instance of `torch._C.ScriptDict`. If ``obj`` is a ``list``,
        then ``script`` returns an instance of `torch._C.ScriptList`.

    **Scripting a function**
        The ``@torch.jit.script`` decorator will construct a :class:`ScriptFunction`
        by compiling the body of the function.

        Example (scripting a function):

        .. testcode::

            import torch

            @torch.jit.script
            def foo(x, y):
                if x.max() > y.max():
                    r = x
                else:
                    r = y
                return r

            print(type(foo))  # torch.jit.ScriptFunction

            # See the compiled graph as Python code
            print(foo.code)

            # Call the function using the TorchScript interpreter
            foo(torch.ones(2, 2), torch.ones(2, 2))

        .. testoutput::
            :hide:

            ...

    ****Scripting a function using example_inputs**
        Example inputs can be used to annotate a function arguments.

        Example (annotating a function before scripting):

        .. testcode::

            import torch

            def test_sum(a, b):
                return a + b

            # Annotate the arguments to be int
            scripted_fn = torch.jit.script(test_sum, example_inputs=[(3, 4)])

            print(type(scripted_fn))  # torch.jit.ScriptFunction

            # See the compiled graph as Python code
            print(scripted_fn.code)

            # Call the function using the TorchScript interpreter
            scripted_fn(20, 100)

        .. testoutput::
            :hide:

            ...

    **Scripting an nn.Module**
        Scripting an ``nn.Module`` by default will compile the ``forward`` method and recursively
        compile any methods, submodules, and functions called by ``forward``. If a ``nn.Module`` only uses
        features supported in TorchScript, no changes to the original module code should be necessary. ``script``
        will construct :class:`ScriptModule` that has copies of the attributes, parameters, and methods of
        the original module.

        Example (scripting a simple module with a Parameter):

        .. testcode::

            import torch

            class MyModule(torch.nn.Module):
                def __init__(self, N, M):
                    super().__init__()
                    # This parameter will be copied to the new ScriptModule
                    self.weight = torch.nn.Parameter(torch.rand(N, M))

                    # When this submodule is used, it will be compiled
                    self.linear = torch.nn.Linear(N, M)

                def forward(self, input):
                    output = self.weight.mv(input)

                    # This calls the `forward` method of the `nn.Linear` module, which will
                    # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
                    output = self.linear(output)
                    return output

            scripted_module = torch.jit.script(MyModule(2, 3))

        Example (scripting a module with traced submodules):

        .. testcode::

            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class MyModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    # torch.jit.trace produces a ScriptModule's conv1 and conv2
                    self.conv1 = torch.jit.trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
                    self.conv2 = torch.jit.trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))

                def forward(self, input):
                    input = F.relu(self.conv1(input))
                    input = F.relu(self.conv2(input))
                    return input

            scripted_module = torch.jit.script(MyModule())

        To compile a method other than ``forward`` (and recursively compile anything it calls), add
        the :func:`@torch.jit.export <torch.jit.export>` decorator to the method. To opt out of compilation
        use :func:`@torch.jit.ignore <torch.jit.ignore>` or :func:`@torch.jit.unused <torch.jit.unused>`.

        Example (an exported and ignored method in a module)::

            import torch
            import torch.nn as nn

            class MyModule(nn.Module):
                def __init__(self):
                    super().__init__()

                @torch.jit.export
                def some_entry_point(self, input):
                    return input + 10

                @torch.jit.ignore
                def python_only_fn(self, input):
                    # This function won't be compiled, so any
                    # Python APIs can be used
                    import pdb
                    pdb.set_trace()

                def forward(self, input):
                    if self.training:
                        self.python_only_fn(input)
                    return input * 99

            scripted_module = torch.jit.script(MyModule())
            print(scripted_module.some_entry_point(torch.randn(2, 2)))
            print(scripted_module(torch.randn(2, 2)))

        Example ( Annotating forward of nn.Module using example_inputs)::

            import torch
            import torch.nn as nn
            from typing import NamedTuple

            class MyModule(NamedTuple):
            result: List[int]

            class TestNNModule(torch.nn.Module):
                def forward(self, a) -> MyModule:
                    result = MyModule(result=a)
                    return result

            pdt_model = TestNNModule()

            # Runs the pdt_model in eager model with the inputs provided and annotates the arguments of forward
            scripted_model = torch.jit.script(pdt_model, example_inputs={pdt_model: [([10, 20, ], ), ], })

            # Run the scripted_model with actual inputs
            print(scripted_model([20]))
    """
    global type_trace_db
    if not _enabled:
        return obj

    if optimize is not None:
        warnings.warn(
            "`optimize` is deprecated and has no effect. Use `with torch.jit.optimized_execution() instead"
        )

    # No-op for modules, functions, class instances that are already scripted
    if isinstance(obj, RecursiveScriptClass):
        return obj
    if isinstance(obj, ScriptModule):
        return obj
    if isinstance(obj, ScriptFunction):
        return obj

    if example_inputs:
        # If MonkeyType is installed, enable profile directed type annotation
        # Check if example_inputs are defined and generate call traces
        # for the method by running eager mode version of the method with
        # the provide example inputs. This logs all the traces in type_trace_db
        type_trace_db = JitTypeTraceStore()
        if monkeytype_trace:
            monkeytype_config = JitTypeTraceConfig(type_trace_db)
            with monkeytype_trace(monkeytype_config):
                if isinstance(example_inputs, Dict):
                    # If the obj is an nn.Module or a class, then each method is
                    # executed with the arguments provided in the example inputs.
                    # example inputs here will be of type Dict(class.method, (arguments))
                    # This is used to infer type annotations for those methods
                    # which are not called directly under the hood of monkeytype.
                    for module, example_input in example_inputs.items():
                        for example in example_input:
                            module(*example)
                elif isinstance(example_inputs, List):
                    for examples in example_inputs:
                        obj(*examples)
                else:
                    raise ValueError(
                        "Error: Unable to infer types. Please format the inputs to type `List[Tuple]`"
                        " or `Dict[Callable, List[Tuple]]` to be run with MonkeyType."
                    )
        else:
            warnings.warn(
                "Warning: monkeytype is not installed. Please install https://github.com/Instagram/MonkeyType "
                "to enable Profile-Directed Typing in TorchScript. Refer to "
                "https://github.com/Instagram/MonkeyType/blob/master/README.rst to install MonkeyType. "
            )

    if isinstance(obj, torch.nn.Module):
        obj = call_prepare_scriptable_func(obj)
        return torch.jit._recursive.create_script_module(
            obj, torch.jit._recursive.infer_methods_to_compile
        )

    if isinstance(obj, dict):
        return create_script_dict(obj)
    if isinstance(obj, list):
        return create_script_list(obj)

    if inspect.isclass(obj):
        qualified_name = _qualified_name(obj)
        # If this type is a `nn.Module` subclass, they probably meant to pass
        # an instance instead of a Module
        if issubclass(obj, torch.nn.Module):
            raise RuntimeError(
                f"Type '{obj}' cannot be compiled since it inherits from nn.Module, pass an instance instead"
            )

        # Enums are automatically usable in TorchScript, explicitly scripting
        # is not necessary, but not harmful either.
        if issubclass(obj, enum.Enum):
            return obj

        if not _is_new_style_class(obj):
            raise RuntimeError(
                "TorchScript classes must be new-style classes. "
                "Please inherit from 'object'."
            )
        if len(obj.mro()) > 2:
            raise RuntimeError(
                "TorchScript classes does not support inheritance yet. "
                "Please directly inherit from 'object'."
            )
        if _rcb is None:
            _rcb = _jit_internal.createResolutionCallbackFromFrame(_frames_up + 1)
        _compile_and_register_class(obj, _rcb, qualified_name)
        return obj
    elif inspect.isfunction(obj) or inspect.ismethod(obj):
        qualified_name = _qualified_name(obj)
        # this is a decorated fn, and we need to the underlying fn and its rcb
        if hasattr(obj, "__script_if_tracing_wrapper"):
            obj = obj.__original_fn  # type: ignore[union-attr]
            _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)

        # some functions are explicitly marked as not supported in script mode
        if hasattr(obj, "__script_unsupported"):
            raise RuntimeError("TorchScript error: " + obj.__script_unsupported)

        _check_directly_compile_overloaded(obj)
        maybe_already_compiled_fn = _try_get_jit_cached_function(obj)
        if maybe_already_compiled_fn:
            return maybe_already_compiled_fn
        ast = get_jit_def(obj, obj.__name__)
        if _rcb is None:
            _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)
        fn = torch._C._jit_script_compile(
            qualified_name, ast, _rcb, get_default_args(obj)
        )
        # Forward docstrings
        fn.__doc__ = obj.__doc__
        # Allow torch.compile() to inline
        fn._torchdynamo_inline = obj  # type: ignore[attr-defined]
        _set_jit_function_cache(obj, fn)
        return fn
    else:
        return torch.jit._recursive.create_script_class(obj)


# overloads are registered in _jit_internal and compiled here so that _overload
# can be used in nn/functional.py without an import cycle


def _check_overload_defaults(impl_defaults, overload_defaults, loc):
    for name, overload_value in overload_defaults.items():
        if name not in impl_defaults or impl_defaults[name] != overload_value:
            raise torch.jit.frontend.FrontendError(
                loc,
                "Default parameters on overloads do not affect the runtime so they "
                "must equal to the default parameter on the implementation function. Found on "
                "parameter {name}".format(name=name),
            )


def _compile_function_with_overload(overload_fn, qual_name, impl_fn):
    overload_decl = get_jit_def(overload_fn, overload_fn.__name__).decl()
    overload_signature = torch.jit.annotations.get_signature(
        overload_fn, None, None, inspect.ismethod(overload_fn)
    )
    impl_ast = get_jit_def(impl_fn, impl_fn.__name__)
    overload_defaults = get_default_args(overload_fn)
    implementation_defaults = get_default_args(impl_fn)
    _rcb = _jit_internal.createResolutionCallbackFromClosure(impl_fn)
    _check_overload_defaults(
        implementation_defaults, overload_defaults, overload_decl.range()
    )
    fn = torch._C._jit_script_compile_overload(
        qual_name,
        overload_decl,
        impl_ast,
        _rcb,
        implementation_defaults,
        overload_signature,
    )
    return fn


def _get_overloads(obj):
    # check for cached compiled fns
    existing_compiled_fns = _try_get_jit_cached_overloads(obj)
    qual_name = _qualified_name(obj)
    uncompiled_overloads = _jit_internal._get_fn_overloads(qual_name)
    if uncompiled_overloads is None:
        return existing_compiled_fns

    if obj in uncompiled_overloads:
        raise RuntimeError(
            _jit_internal.get_overload_no_implementation_error_message("function", obj)
        )

    compiled_fns = []
    for overload_fn in uncompiled_overloads:
        compiled_fns.append(
            _compile_function_with_overload(overload_fn, qual_name, obj)
        )

    if existing_compiled_fns:
        compiled_fns = existing_compiled_fns + compiled_fns

    # cache compilation, remove information stored to do compilation
    _set_jit_overload_cache(obj, compiled_fns)
    _jit_internal._clear_fn_overloads(qual_name)
    return compiled_fns


def _check_directly_compile_overloaded(obj):
    qual_name = _qualified_name(obj)
    if _jit_internal._get_fn_overloads(qual_name) or _try_get_jit_cached_overloads(obj):
        raise RuntimeError(
            "Function {} cannot be directly compiled because it"
            " is overloaded. It must be used in a context of a function"
            " where its inputs can determine which overload to call.".format(qual_name)
        )


def interface(obj):
    if not inspect.isclass(obj):
        raise RuntimeError("interface must be applied to a class")
    if not _is_new_style_class(obj):
        raise RuntimeError("TorchScript interfaces must inherit from 'object'")

    # Expected MRO is:
    #   User module
    #   torch.nn.modules.module.Module
    #   object
    is_module_interface = issubclass(obj, torch.nn.Module) and len(obj.mro()) == 3

    if not is_module_interface and len(obj.mro()) > 2:
        raise RuntimeError(
            "TorchScript interface does not support inheritance yet. "
            "Please directly inherit from 'object' or 'nn.Module'."
        )

    qualified_name = _qualified_name(obj)
    rcb = _jit_internal.createResolutionCallbackFromFrame(1)
    # if this type is a `nn.Module` subclass, generate a module interface type
    # instead of a class interface type; a module interface type only compiles
    # the user provided methods as part of the interface
    ast = get_jit_class_def(obj, obj.__name__)
    mangled_classname = torch._C._jit_script_interface_compile(
        qualified_name, ast, rcb, is_module_interface
    )
    obj.__torch_script_interface__ = mangled_classname
    return obj


def _recursive_compile_class(obj, loc):
    _qual_name = _qualified_name(obj)
    # We're starting a new compilation, so update the error call stack in
    # case it fails
    error_stack = torch._C.CallStack(_qual_name, loc)
    rcb = _jit_internal.createResolutionCallbackForClassMethods(obj)
    return _compile_and_register_class(obj, rcb, _qual_name)


CompilationUnit = torch._C.CompilationUnit
set_module(CompilationUnit, "torch.jit")


def pad(s: str, padding: int, offset: int = 0, char: str = " "):
    if padding >= len(s):
        padding -= len(s)
    return "".join([char for _ in range(padding + offset)]) + s


class _ScriptProfileColumn:
    def __init__(self, header: str, alignment: int = 4, offset: int = 0):
        self.header = header
        self.alignment = alignment
        self.offset = offset
        self.rows: Dict[int, Any] = {}

    def add_row(self, lineno: int, value: Any):
        self.rows[lineno] = value

    def materialize(self):
        max_length = len(self.header)
        rows: List[Tuple[int, str]] = []
        for key, value in self.rows.items():
            cell = str(value)
            rows.append((key, cell))
            max_length = max(len(cell), max_length)

        if self.alignment > 0:
            padding = max_length + self.alignment
            padding -= padding % self.alignment
        else:
            padding = 0

        rows = [(key, pad(cell, padding, self.offset)) for key, cell in rows]
        return pad(self.header, padding, self.offset), rows


class _ScriptProfileTable:
    def __init__(self, cols: List[_ScriptProfileColumn], source_range: List[int]):
        self.cols = cols
        self.source_range = source_range

    def dump_string(self):
        outputs: List[str] = []
        cells: List[Tuple[str, Dict[int, str]]] = []
        header_buffer = ""
        for col in self.cols:
            header, rows = col.materialize()
            header_buffer += header
            cells.append((header, dict(rows)))

        outputs.append(header_buffer)
        outputs.append(pad("", len(header_buffer), 0, "="))
        for line in self.source_range:
            row_buffer = ""
            for header, rows in cells:
                cell = rows.get(line)
                if cell is None:
                    row_buffer += pad("", len(header))
                else:
                    row_buffer += cell
            outputs.append(row_buffer)
        return "\n".join(outputs)


class _ScriptProfile:
    def __init__(self):
        self.profile = classes.profiling._ScriptProfile()

    def enable(self):
        self.profile.enable()

    def disable(self):
        self.profile.disable()

    def dump_string(self) -> str:
        outputs: List[str] = []
        for source_stats in self.profile._dump_stats():
            source_ref = source_stats.source()
            source_lines = source_ref.text().splitlines()
            dedent = min([len(line) - len(line.lstrip(" ")) for line in source_lines])
            source_lines = [line[dedent:] for line in source_lines]

            start_line = source_ref.starting_lineno()
            end_line = start_line + len(source_lines)
            source_range = range(start_line, end_line)
            lineno = _ScriptProfileColumn("Line #")
            hits = _ScriptProfileColumn("Hits")
            time_ns = _ScriptProfileColumn("Time (ns)")
            line_contents = _ScriptProfileColumn("Line Contents", 0, 1)
            stats = source_stats.line_map()
            for line in source_range:
                lineno.add_row(line, line)
                line_contents.add_row(line, source_lines[line - start_line])
                stat = stats.get(line)
                if stat is not None:
                    hits.add_row(line, stat.count())
                    time_ns.add_row(line, stat.duration_ns())

            table = _ScriptProfileTable(
                [lineno, hits, time_ns, line_contents], list(source_range)
            )
            outputs.append(table.dump_string())
        return "\n\n".join(outputs)

    def dump(self):
        print(self.dump_string())


def _unwrap_optional(x):
    assert x is not None, "Unwrapping null optional"
    return x


_register_builtin(_unwrap_optional, "aten::_unwrap_optional")
_register_builtin(_jit_internal.is_scripting, "aten::is_scripting")
_register_builtin(has_torch_function, "aten::has_torch_function")
_register_builtin(has_torch_function_unary, "aten::has_torch_function")
_register_builtin(has_torch_function_variadic, "aten::has_torch_function")
