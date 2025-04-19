# mypy: allow-untyped-defs
import builtins
import collections
import contextlib
import copy
import functools
import inspect
import math
import os
import warnings
from itertools import chain
from types import CodeType, FunctionType, ModuleType
from typing import Any, Callable, NamedTuple, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch._C import ScriptObject  # type: ignore[attr-defined]
from torch._library.fake_class_registry import FakeScriptObject

from ._compatibility import compatibility
from ._lazy_graph_module import _make_graph_module
from .graph import _PyTreeCodeGen, _PyTreeInfo, Graph
from .graph_module import GraphModule
from .node import Argument, base_types, map_aggregate
from .proxy import ParameterProxy, Proxy, Scope, ScopeContextManager, TracerBase


HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS

# These need to run in global scope to handle nested calls correctly
_orig_module_call: Callable = torch.nn.Module.__call__
_orig_module_getattr: Callable = torch.nn.Module.__getattr__

_proxyable_classes: dict[type, None] = {}

_is_fx_tracing_flag = False


def is_fx_tracing():
    return _is_fx_tracing_flag


@compatibility(is_backward_compatible=True)
class ProxyableClassMeta(type):
    """
    ProxyableClassMeta allows you to make construction of a given Python class
    symbolically traceable. For example::

        import torch
        import torch.fx


        class TensorPair(metaclass=torch.fx.ProxyableClassMeta):
            def __init__(self, left, right):
                self.left, self.right = left, right

            def add(self, other):
                l = self.left + other.left
                r = self.right + other.right
                return TensorPair(l, r)

            def mul(self, other):
                l = self.left * other.left
                r = self.right * other.right
                return TensorPair(l, r)


        def use_tensor_pair_ctor(x: TensorPair, y: torch.Tensor):
            s = x.add(TensorPair(y, y))
            return s.mul(x)


        x = TensorPair(torch.randn(5, 3), torch.randn(5, 3))
        y = torch.randn(5, 3)
        ref_out = use_tensor_pair_ctor(x, y)

        traced = torch.fx.symbolic_trace(use_tensor_pair_ctor)
        print(traced.code)
        '''
        def forward(self, x : __main___TensorPair, y : torch.Tensor):
            tensor_pair = __main___TensorPair(y, y);  y = None
            add = x.add(tensor_pair);  tensor_pair = None
            mul = add.mul(x);  add = x = None
            return mul
        '''

    From this example, we can see that construction of a class (``TensorPair``)
    defined with ``ProxyableClassMeta`` as metaclass can be recorded in symbolic
    tracing.
    """

    def __init__(cls, name, bases, attrs):
        _proxyable_classes.setdefault(cls)
        super().__init__(name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls)  # type: ignore[call-overload]

        if not is_fx_tracing():
            cls.__init__(instance, *args, **kwargs)  # type: ignore[misc]
            return instance

        found_proxies = []

        def check_proxy(a):
            if isinstance(a, Proxy):
                found_proxies.append(a)

        map_aggregate(args, check_proxy)
        map_aggregate(kwargs, check_proxy)

        if len(found_proxies) != 0:
            tracer = found_proxies[0].tracer
            return tracer.create_proxy("call_function", cls, args, kwargs)
        else:
            cls.__init__(instance, *args, **kwargs)  # type: ignore[misc]
            return instance


def _patch_function(fn: FunctionType, nargs: int) -> FunctionType:
    co = fn.__code__
    co_flags = co.co_flags & ~HAS_VARSTUFF
    co_args: tuple
    if hasattr(co, "co_qualname"):
        # Python-3.11+ code signature
        co_args = (
            nargs,
            0,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_qualname,  # type: ignore[attr-defined]
            co.co_firstlineno,
            co.co_lnotab,
            co.co_exceptiontable,  # type: ignore[attr-defined]
            co.co_freevars,
            co.co_cellvars,
        )
    elif hasattr(co, "co_posonlyargcount"):
        co_args = (
            nargs,
            0,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_freevars,
            co.co_cellvars,
        )
    else:
        co_args = (
            nargs,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_freevars,
            co.co_cellvars,
        )
    new_code = CodeType(*co_args)  # type: ignore[arg-type]
    return FunctionType(
        new_code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )

    # we need to insert placeholder nodes for *args and **kwargs
    # we can't call this function normally, otherwise it would try to unpack them
    # instead, let's make python think that args and kwargs are normal variables


@compatibility(is_backward_compatible=False)
class PHBase:
    """
    Object representing an input placeholder to `concrete_args`
    """

    def __repr__(self):
        return "PH"


PH = PHBase()


@compatibility(is_backward_compatible=False)
class PHWithMeta(PHBase):
    """
    Object representing an input placeholder to `concrete_args`
    """

    def __init__(self, ph_key: Optional[str] = None):
        super().__init__()

        # Provide a hey for user to identify placeholder node during analysis
        self.ph_key = ph_key


def _transfer_attrs(fr, to):
    for attr_name in dir(fr):
        attr_val = getattr(fr, attr_name)
        if (
            not callable(attr_val)
            and not attr_name.startswith("__")
            and not hasattr(to, attr_name)
        ):
            setattr(to, attr_name, attr_val)


@compatibility(is_backward_compatible=True)
class Tracer(TracerBase):
    # Reference: https://github.com/pytorch/pytorch/issues/54354
    # The first line of this docstring overrides the one Sphinx generates for the
    # documentation. We need it so that Sphinx doesn't leak `math`s path from the
    # build environment (e.g. `<module 'math' from '/leaked/path').

    """Tracer(autowrap_modules=(math,), autowrap_functions=())

    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.

    Tracer can be subclassed to override various behaviors of the tracing
    process. The different behaviors that can be overridden are described
    in the docstrings of the methods on this class.
    """

    # Not checking BC on this API because the default value for `autowrap_modules`
    # includes the local filepath to the `math` module, which would jitter
    # across machines.
    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        autowrap_modules: tuple[ModuleType] = (math,),
        autowrap_functions: tuple[Callable, ...] = (),
        param_shapes_constant: bool = False,
    ) -> None:
        # This method's signature is overridden by the first line of this class'
        # docstring. If this method's signature is modified, the signature that
        # overrides it also should be modified accordingly.

        """
        Construct a Tracer object.

        Args:

            autowrap_modules (Tuple[ModuleType]): defaults to `(math, )`,
                Python modules whose functions should be wrapped automatically
                without needing to use fx.wrap(). Backward-compatibility for
                this parameter is guaranteed.

            autowrap_functions (Tuple[Callable, ...]): defaults to `()`,
                Python functions that should be wrapped automatically without
                needing to use fx.wrap(). Backward compatibility for this
                parameter is guaranteed.

            param_shapes_constant (bool): When this flag is set,  calls to shape,
                size and a few other shape like attributes of a module's parameter
                will be evaluated directly, rather than returning a new Proxy value
                for an attribute access. Backward compatibility for this parameter
                is guaranteed.
        """

        super().__init__()

        # Functions we will eagerly wrap when we see them while tracing
        # this captures both `math.sqrt()` and `from math import sqrt` automatically
        self._autowrap_function_ids: set[int] = {
            id(value)
            for name, value in chain.from_iterable(
                m.__dict__.items() for m in autowrap_modules
            )
            if not name.startswith("_") and callable(value)
        }
        self._autowrap_function_ids.update({id(f) for f in autowrap_functions})

        # Python modules to apply autowrap to at the start, in addition to
        # modules we see while tracing
        self._autowrap_search: list[ModuleType] = list(autowrap_modules)
        self.param_shapes_constant = param_shapes_constant

        self.submodule_paths: Optional[dict[torch.nn.Module, str]] = None
        self.root_module_name: str = ""
        # Maps the containing module's name to the operator name
        self.scope = Scope("", None)
        # Records the module call stack
        self.module_stack = collections.OrderedDict()
        self.num_calls: dict[str, int] = {}
        # Mapping of node name to module scope
        self.node_name_to_scope: dict[str, tuple[str, type]] = {}

    _qualname_counter: dict[str, int] = collections.defaultdict(int)

    @compatibility(is_backward_compatible=True)
    def get_fresh_qualname(self, prefix: str) -> str:
        """
        Gets a fresh name for a prefix and returns it. This function ensures
        that it will not clash with an existing attribute on the graph.
        """
        # The idea here is that if the module doesn't have this prefix at all we
        # should reset the counter to start from the beginning
        # It's a ... little bit hacky (doesn't cover all cases) but the precise
        # naming of the prefixes isn't a correctness issue, just a niceness
        # issue
        qualname = f"{prefix}0"
        if not hasattr(self.root, qualname):
            self._qualname_counter[prefix] = 0
            return qualname

        i = self._qualname_counter[prefix]
        while True:
            qualname = f"{prefix}{i}"
            i += 1
            if not hasattr(self.root, qualname):
                break
        self._qualname_counter[prefix] = i

        return qualname

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> "Argument":
        """
        A method to specify the behavior of tracing when preparing values to
        be used as arguments to nodes in the ``Graph``.

        By default, the behavior includes:

        #. Iterate through collection types (e.g. tuple, list, dict) and recursively
           call ``create_args`` on the elements.
        #. Given a Proxy object, return a reference to the underlying IR ``Node``
        #. Given a non-Proxy Tensor object, emit IR for various cases:

            * For a Parameter, emit a ``get_attr`` node referring to that Parameter
            * For a non-Parameter Tensor, store the Tensor away in a special
              attribute referring to that attribute.

        This method can be overridden to support more types.

        Args:

            a (Any): The value to be emitted as an ``Argument`` in the ``Graph``.


        Returns:

            The value ``a`` converted into the appropriate ``Argument``
        """
        # The base tracer is used to construct Graphs when there is no associated
        # module hierarchy, so it can never create parameter references.
        # The default tracer adds the ability to refer to parameters when
        # tracing modules.
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node("get_attr", n, (), {})
            raise NameError("parameter is not a member of this module")
        elif isinstance(a, torch.Tensor):
            for n_, p_ in self.root.named_buffers():
                if a is p_:
                    return self.create_node("get_attr", n_, (), {})
        elif isinstance(a, torch.nn.Module):
            for n_, p_ in self.root.named_modules():
                if a is p_:
                    return self.create_node("get_attr", n_, (), {})
        # For NamedTuple instances that appear literally as args, we emit
        # a node to construct the NamedTuple and use that Node as the argument.
        if isinstance(a, tuple) and hasattr(a, "_fields"):
            args = tuple(self.create_arg(elem) for elem in a)
            return self.create_node("call_function", a.__class__, args, {})

        # Tensors do not have a reliable string repr() from which they can be
        # constructed (and we probably don't want to rely on that, either), so
        # for any constant Tensor values we encounter, first search for if they
        # are an attribute of some module in the module hierarchy. If so, emit
        # a get_attr to retrieve that tensor. Otherwise, we'll store away the
        # tensor value into a special attribute on the Module s.t. we can
        # retrieve it with a get_attr.
        if isinstance(a, (torch.Tensor, ScriptObject, FakeScriptObject)):
            qualname: Optional[str] = self.tensor_attrs.get(a)

            # Tensor was not found in the Module hierarchy, stow it away in a
            # special attribute and set the qualname to refer to that
            if not qualname:
                base_name = (
                    "_tensor_constant"
                    if isinstance(a, torch.Tensor)
                    else "_torchbind_obj"
                )
                qualname = self.get_fresh_qualname(base_name)
                assert isinstance(qualname, str)
                self.tensor_attrs[a] = qualname
                setattr(self.root, qualname, a)

            return self.create_node("get_attr", qualname, (), {})

        if type(a) in _proxyable_classes:
            # This is an instance of a proxyable class for which we did not
            # witness its construction. Intern this as a constant attribute

            # TODO: binary search
            qualname = self.get_fresh_qualname(f"_{a.__class__.__name__}_constant_")
            assert isinstance(qualname, str)
            setattr(self.root, qualname, a)

            return self.create_node("get_attr", qualname, (), {})

        return super().create_arg(a)

    @compatibility(is_backward_compatible=True)
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.

        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.

        Args:

            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        return (
            m.__module__.startswith("torch.nn")
            or m.__module__.startswith("torch.ao.nn")
        ) and not isinstance(m, torch.nn.Sequential)

    @compatibility(is_backward_compatible=True)
    def path_of_module(self, mod: torch.nn.Module) -> str:
        """
        Helper method to find the qualified name of ``mod`` in the Module hierarchy
        of ``root``. For example, if ``root`` has a submodule named ``foo``, which has
        a submodule named ``bar``, passing ``bar`` into this function will return
        the string "foo.bar".

        Args:

            mod (str): The ``Module`` to retrieve the qualified name for.
        """
        # Prefer the O(1) algorithm
        if self.submodule_paths:
            path = self.submodule_paths.get(mod)
            if path is None:
                raise NameError("module is not installed as a submodule")
            assert isinstance(path, str)
            return path
        # O(N^2) fallback in the case that we didn't store the submodule
        # paths.
        else:
            for n, p in self.root.named_modules():
                if mod is p:
                    return n
            raise NameError("module is not installed as a submodule")

    @compatibility(is_backward_compatible=True)
    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Method that specifies the behavior of this ``Tracer`` when it encounters
        a call to an ``nn.Module`` instance.

        By default, the behavior is to check if the called module is a leaf module
        via ``is_leaf_module``. If it is, emit a ``call_module`` node referring to
        ``m`` in the ``Graph``. Otherwise, call the ``Module`` normally, tracing through
        the operations in its ``forward`` function.

        This method can be overridden to--for example--create nested traced
        GraphModules, or any other behavior you would want while tracing across
        ``Module`` boundaries.

        Args:

            m (Module): The module for which a call is being emitted
            forward (Callable): The forward() method of the ``Module`` to be invoked
            args (Tuple): args of the module callsite
            kwargs (Dict): kwargs of the module callsite

        Return:

            The return value from the Module call. In the case that a ``call_module``
            node was emitted, this is a ``Proxy`` value. Otherwise, it is whatever
            value was returned from the ``Module`` invocation.
        """
        module_qualified_name = self.path_of_module(m)
        with ScopeContextManager(
            self.scope, Scope(module_qualified_name, type(m))
        ) as _scope:
            # module_stack is an ordered dict so writing then deleting the
            # entry is equivalent to push/pop on a list
            num_calls = self.num_calls.get(module_qualified_name, 0)
            module_key = (
                f"{_scope.module_path}@{num_calls}"
                if num_calls > 0
                else _scope.module_path
            )
            self.module_stack[module_key] = (module_qualified_name, _scope.module_type)
            self.num_calls[module_qualified_name] = num_calls + 1
            if not self.is_leaf_module(m, module_qualified_name):
                ret_val = forward(*args, **kwargs)
            else:
                ret_val = self.create_proxy(
                    "call_module", module_qualified_name, args, kwargs
                )
            key, _ = self.module_stack.popitem(last=True)
            assert key == module_key, f" Unexpected key {key}"

        return ret_val

    @compatibility(is_backward_compatible=False)
    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: dict[str, Any]):
        """
        Method that specifies the behavior of this ``Tracer`` when we call getattr
        on a call to an ``nn.Module`` instance.

        By default, the behavior is to return a proxy value for the attribute. It
        also stores the proxy value in the ``parameter_proxy_cache``, so that future
        calls will reuse the proxy rather than creating a new one.

        This method can be overridden to --for example-- not return proxies when
        querying parameters.

        Args:

            attr (str): The name of the attribute being queried
            attr_val (Any): The value of the attribute
            parameter_proxy_cache (Dict[str, Any]): A cache of attr names to proxies

        Return:

            The return value from the getattr call.
        """

        def maybe_get_proxy_for_attr(
            attr_val, collection_to_search, parameter_proxy_cache
        ):
            for n, p in collection_to_search:
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        kwargs = {}
                        if (
                            "proxy_factory_fn"
                            in inspect.signature(self.create_proxy).parameters
                        ):
                            kwargs["proxy_factory_fn"] = (
                                None
                                if not self.param_shapes_constant
                                else lambda node: ParameterProxy(
                                    self, node, n, attr_val
                                )
                            )
                        val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)  # type: ignore[arg-type]
                        parameter_proxy_cache[n] = val_proxy
                    return parameter_proxy_cache[n]
            return None

        if isinstance(attr_val, torch.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_parameters(), parameter_proxy_cache
            )
            if maybe_parameter_proxy is not None:
                return maybe_parameter_proxy

        if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
            maybe_buffer_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_buffers(), parameter_proxy_cache
            )
            if maybe_buffer_proxy is not None:
                return maybe_buffer_proxy

        return attr_val

    # This method will be refactored
    @compatibility(is_backward_compatible=False)
    def create_args_for_root(self, root_fn, is_module, concrete_args=None):
        """
        Create ``placeholder`` nodes corresponding to the signature of the ``root``
        Module. This method introspects root's signature and emits those
        nodes accordingly, also supporting ``*args`` and ``**kwargs``.
        """
        # In some cases, a function or method has been decorated with a wrapper
        # defined via ``functools.wraps``. In this case, the outer code object
        # will likely not contain the actual parameters we care about, so unwrap
        # the function to get to the innermost callable.
        fn_for_analysis = inspect.unwrap(root_fn)
        co = fn_for_analysis.__code__
        total_args = co.co_argcount + co.co_kwonlyargcount
        orig_args = list(co.co_varnames)
        names_iter = iter(co.co_varnames)
        args: list[Any] = []
        skip_arg_idx = 0
        if is_module:
            if total_args == 0:
                raise RuntimeError(
                    "``self`` argument cannot be part of *args expansion!"
                )
            skip_arg_idx = 1
            next(names_iter)  # skip self
            args.append(self.root)

        sig = inspect.signature(fn_for_analysis)

        # This covers the very specific case where we are passing in flat
        # concrete_args as a tuple, but our traced fn takes (*args, **kwargs).
        # In this case, just take the concrete_args and pass them through.
        name_idx = 0
        if (
            isinstance(concrete_args, tuple)
            and len(concrete_args) > 0
            and (co.co_flags & HAS_VARSTUFF)
            and total_args == 1
        ):
            for concrete_arg in concrete_args:
                out = self.create_proxy("placeholder", f"input_{name_idx}", (), {})
                if isinstance(concrete_arg, PHBase):
                    if concrete_arg != PH:
                        # Transfer attrs in the case where you're using a placeholder other
                        # than the singleton PH (PH has no attributes to transfer).
                        # Proxies were created out of the placeholders.
                        # Transfer any metadata (put on the placeholders in the form of
                        # attributes set by the user) from the placeholder to the
                        # underlying nodes (the proxy is unwrapped by the user, but
                        # the metadata should hold).
                        _transfer_attrs(fr=concrete_arg, to=out.node)
                args.append(out)
                name_idx += 1
            return root_fn, args

        arg_names = [next(names_iter) for idx in range(skip_arg_idx, total_args)]
        if isinstance(concrete_args, tuple):
            if len(arg_names) != len(concrete_args):
                raise RuntimeError(
                    f"Tracing expected {len(arg_names)} arguments but got {len(concrete_args)} concrete arguments"
                )
            concrete_args = dict(zip(arg_names, concrete_args))

        def proxy_placeholder(name):
            return self._proxy_placeholder(name, concrete_args, sig, fn_for_analysis)

        args.extend(proxy_placeholder(names) for names in arg_names)

        if co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF:
            # TODO: type annotations for *args and **kwargs
            if co.co_flags & inspect.CO_VARARGS:
                args.append(proxy_placeholder("*" + next(names_iter)))
            if co.co_flags & inspect.CO_VARKEYWORDS:
                args.append(proxy_placeholder("**" + next(names_iter)))
            root_fn = _patch_function(root_fn, len(args))

        flat_args, in_spec = pytree.tree_flatten(tuple(args))
        if not all(child.is_leaf() for child in in_spec.children_specs):
            # In the case that we have pytree-flattened inputs in
            # `concrete_args`, generate a flattening wrapper around the
            # original root function and return that.
            self.graph._codegen = _PyTreeCodeGen(
                _PyTreeInfo(orig_args[:total_args], in_spec, None)
            )

            def flatten_fn(*args):
                tree_args = pytree.tree_unflatten(list(args), in_spec)
                tree_out = root_fn(*tree_args)
                out_args, out_spec = pytree.tree_flatten(tree_out)
                assert isinstance(self.graph._codegen, _PyTreeCodeGen)
                self.graph._codegen.pytree_info = (
                    self.graph._codegen.pytree_info._replace(out_spec=out_spec)
                )
                return out_args

            return flatten_fn, flat_args
        return root_fn, args

    @compatibility(is_backward_compatible=True)
    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[dict[str, Any]] = None,
    ) -> Graph:
        """
        Trace ``root`` and return the corresponding FX ``Graph`` representation. ``root``
        can either be an ``nn.Module`` instance or a Python callable.

        Note that after this call, ``self.root`` may be different from the ``root`` passed
        in here. For example, when a free function is passed to ``trace()``, we will
        create an ``nn.Module`` instance to use as the root and add embedded constants
        to.


        Args:

            root (Union[Module, Callable]): Either a ``Module`` or a function to be
                traced through. Backwards-compatibility for this parameter is
                guaranteed.
            concrete_args (Optional[Dict[str, any]]): Concrete arguments that should
                not be treated as Proxies. This parameter is experimental and
                its backwards-compatibility is *NOT* guaranteed.

        Returns:

            A ``Graph`` representing the semantics of the passed-in ``root``.
        """
        global _is_fx_tracing_flag
        old_is_fx_tracing_flag = _is_fx_tracing_flag
        _is_fx_tracing_flag = True
        try:
            if isinstance(root, torch.nn.Module):
                # do real recompilation for _LazyGraphModule before retracing since the trace
                # method can not trace the _lazy_forward method. Got error:
                #   https://gist.github.com/shunting314/75549c2e82ae07ac1139c94a3583d259
                # without this.
                from torch.fx._lazy_graph_module import _LazyGraphModule

                _LazyGraphModule.force_recompile(root)

                self.root = root

                assert hasattr(
                    type(root), self.traced_func_name
                ), f"traced_func_name={self.traced_func_name} doesn't exist in {type(root).__name__}"

                fn = getattr(type(root), self.traced_func_name)
                self.root_module_name = root._get_name()
                self.submodule_paths = {mod: name for name, mod in root.named_modules()}
            else:
                self.root = torch.nn.Module()
                fn = root

            tracer_cls: Optional[type[Tracer]] = getattr(self, "__class__", None)
            self.graph = Graph(tracer_cls=tracer_cls)
            if hasattr(fn, "__code__"):
                code = fn.__code__
                self.graph._co_fields = {
                    "co_name": code.co_name,
                    "co_filename": code.co_filename,
                    "co_firstlineno": code.co_firstlineno,
                }

            # When we encounter a Tensor value that's not a parameter, we look if it
            # is some other attribute on the model. Construct a dict mapping Tensor
            # values to the qualified name here for efficiency. This is used downstream
            # in create_arg
            self.tensor_attrs: dict[
                Union[torch.Tensor, ScriptObject, FakeScriptObject], str
            ] = {}

            def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: list[str]):
                for k, v in m.__dict__.items():
                    if isinstance(v, (torch.Tensor, ScriptObject, FakeScriptObject)):
                        self.tensor_attrs[v] = ".".join(prefix_atoms + [k])
                for k, v in m.named_children():
                    collect_tensor_attrs(v, prefix_atoms + [k])

            collect_tensor_attrs(self.root, [])

            assert isinstance(fn, FunctionType)

            fn_globals = fn.__globals__  # run before it gets patched
            fn, args = self.create_args_for_root(
                fn, isinstance(root, torch.nn.Module), concrete_args
            )

            parameter_proxy_cache: dict[
                str, Proxy
            ] = {}  # Reduce number of get_attr calls

            # Method dispatch on parameters is not recorded unless it's directly used.
            # Thus, we need to insert a proxy when __getattr__ requests a parameter.
            @functools.wraps(_orig_module_getattr)
            def module_getattr_wrapper(mod, attr):
                attr_val = _orig_module_getattr(mod, attr)
                return self.getattr(attr, attr_val, parameter_proxy_cache)

            @functools.wraps(_orig_module_call)
            def module_call_wrapper(mod, *args, **kwargs):
                def forward(*args, **kwargs):
                    return _orig_module_call(mod, *args, **kwargs)

                _autowrap_check(
                    patcher,  # type: ignore[has-type]
                    getattr(getattr(mod, "forward", mod), "__globals__", {}),
                    self._autowrap_function_ids,
                )
                return self.call_module(mod, forward, args, kwargs)

            with _new_patcher() as patcher:
                # allow duplicate patches to support the case of nested calls
                patcher.patch_method(
                    torch.nn.Module,
                    "__getattr__",
                    module_getattr_wrapper,
                    deduplicate=False,
                )
                patcher.patch_method(
                    torch.nn.Module,
                    "__call__",
                    module_call_wrapper,
                    deduplicate=False,
                )
                _patch_wrapped_functions(patcher)
                _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
                for module in self._autowrap_search:
                    _autowrap_check(
                        patcher, module.__dict__, self._autowrap_function_ids
                    )
                self.create_node(
                    "output",
                    "output",
                    (self.create_arg(fn(*args)),),
                    {},
                    type_expr=fn.__annotations__.get("return", None),
                )

            self.submodule_paths = None
        except RuntimeError as e:
            if isinstance(e.args[0], str) and "data-dependent" in e.args[0]:
                partial_fx_graph = self.graph.python_code(
                    root_module="self",
                    verbose=True,
                ).src
                e.partial_fx_graph = partial_fx_graph  # type: ignore[attr-defined]
                raise

            raise
        finally:
            _is_fx_tracing_flag = old_is_fx_tracing_flag
        return self.graph

    def __deepcopy__(self, memo):
        # _autowrap_search contains modules, which cannot be deepcopied.
        new_tracer = Tracer.__new__(Tracer)

        for k, v in self.__dict__.items():
            if k in {"_autowrap_search"}:
                new_obj = copy.copy(v)
            else:
                new_obj = copy.deepcopy(v, memo)

            new_tracer.__dict__[k] = new_obj

        return new_tracer

    def _proxy_placeholder(self, name, concrete_args, sig, fn_for_analysis):
        if concrete_args is not None and name in concrete_args:
            cnt = 0

            def replace_ph(x):
                nonlocal cnt
                cnt += 1
                param = sig.parameters[name]
                default: tuple[Any, ...] = (
                    () if param.default is inspect.Parameter.empty else (param.default,)
                )
                out = self.create_proxy(
                    "placeholder", f"{name}_{str(cnt)}", default, {}
                )
                if isinstance(x, PHBase):
                    if x != PH:
                        # Transfer attrs in the case where you're using a placeholder other
                        # than the singleton PH (PH has no attributes to transfer).
                        # Proxies were created out of the placeholders.
                        # Transfer any metadata (put on the placeholders in the form of
                        # attributes set by the user) from the placeholder to the
                        # underlying nodes (the proxy is unwrapped by the user, but
                        # the metadata should hold).
                        _transfer_attrs(fr=x, to=out.node)

                    return out
                # Union[int, bool] == bool in Python <= 3.6
                if type(x) == bool or type(x) in base_types and type(x) != torch.Tensor:
                    torch._assert(
                        out == x,
                        f"{name} has been specialized to have value {x} but got another value",
                    )
                elif x is None:
                    args = (
                        out,
                        f"{name} has been specialized to have value None but got another value",
                    )
                    self.create_proxy("call_function", _assert_is_none, args, {})
                else:
                    warnings.warn(
                        f"Was not able to add assertion to guarantee correct input {name} to "
                        f"specialized function. It is up to the user to make sure that your inputs match the "
                        f"inputs you specialized the function with."
                    )

                return x

            return pytree.tree_map(replace_ph, concrete_args[name])
        if name[0] == "*":
            default: tuple[Any, ...] = ()
        else:
            param = sig.parameters[name]
            default = (  # type: ignore[assignment]
                () if param.default is inspect.Parameter.empty else (param.default,)
            )
        return self.create_proxy(
            "placeholder",
            name,
            default,
            {},
            type_expr=fn_for_analysis.__annotations__.get(name, None),
        )


# Dictionary of (id(globals dict), function name) => globals_dict to patch for
# the purposes of the wrap() API.
# We key by the globals dict id and function name to ensure we're wrapping a given
# function only once.
_wrapped_fns_to_patch: dict[tuple[int, str], dict] = {}

# List of methods on classes to wrap (class type, function name)
# this currently only works for Tensor.* methods that aren't traced properly
_wrapped_methods_to_patch: list[tuple[type, str]] = []

if os.environ.get("FX_PATCH_GETITEM") == "1":
    # This change is needed to trace models like PositionalEmbedding from BERT:
    # https://github.com/pytorch/benchmark/blob/master/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/position.py
    # but causes issues in quantization documented here:
    # https://github.com/pytorch/pytorch/issues/50710
    # once that is fixed we can make this the default behavior.
    _wrapped_methods_to_patch.append((torch.Tensor, "__getitem__"))


def _find_proxy(*objects_to_search):
    """
    Recursively search a data structure for a Proxy() and return it,
    return None if not found.
    """
    proxy = None

    def find_proxy(x):
        nonlocal proxy
        if isinstance(x, Proxy):
            proxy = x

    map_aggregate(objects_to_search, find_proxy)
    return proxy


def _create_wrapped_func(orig_fn):
    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        """
        Given an closed-over ``orig_function`` to invoke, search the args and kwargs for
        a Proxy object. If there is one, emit a ``call_function`` node to preserve the
        call to this leaf function directly. Otherwise, just return the results of
        this function call, as this function is not being traced.
        """
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return_proxy = proxy.tracer.create_proxy(
                "call_function", orig_fn, args, kwargs
            )
            return_proxy.node.meta["is_wrapped"] = True
            return return_proxy
        return orig_fn(*args, **kwargs)

    return wrapped


def _create_wrapped_method(cls, name):
    orig_fn = getattr(cls, name)

    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        """
        Search the args and kwargs for a Proxy object. If there is one,
        emit a ``call_method`` node to preserve the call to this method
        directly. Otherwise, just return the results of this function
        call, as this function is not being traced.
        """
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return proxy.tracer.create_proxy("call_method", name, args, kwargs)
        return orig_fn(*args, **kwargs)

    return wrapped


class _PatchedFn(NamedTuple):
    frame_dict: Any
    fn_name: str
    orig_fn: Any
    new_fn: Any

    def revert(self):
        raise NotImplementedError

    def patch(self):
        raise NotImplementedError


class _PatchedFnSetItem(_PatchedFn):
    def revert(self):
        self.frame_dict[self.fn_name] = self.orig_fn

    def patch(self):
        self.frame_dict[self.fn_name] = self.new_fn


class _PatchedFnDel(_PatchedFn):
    def revert(self):
        del self.frame_dict[self.fn_name]

    def patch(self):
        self.frame_dict[self.fn_name] = self.new_fn


class _PatchedFnSetAttr(_PatchedFn):
    def revert(self):
        setattr(self.frame_dict, self.fn_name, self.orig_fn)

    def patch(self):
        setattr(self.frame_dict, self.fn_name, self.new_fn)


class _Patcher:
    def __init__(self) -> None:
        super().__init__()
        self.patches_made: list[_PatchedFn] = []
        self.visited: set[int] = set()

    def patch(
        self,
        frame_dict: dict[str, Any],
        name: str,
        new_fn: Callable,
        deduplicate: bool = True,
    ):
        """
        Replace frame_dict[name] with new_fn until we exit the context manager.
        """
        new_fn.__fx_already_patched = deduplicate  # type: ignore[attr-defined]
        if name not in frame_dict and hasattr(builtins, name):
            self.patches_made.append(_PatchedFnDel(frame_dict, name, None, new_fn))
            self.patches_made[-1].patch()
        elif getattr(frame_dict[name], "__fx_already_patched", False):
            return  # already patched, no need to do it again
        else:
            self.patches_made.append(
                _PatchedFnSetItem(frame_dict, name, frame_dict[name], new_fn)
            )
            self.patches_made[-1].patch()

    def patch_method(
        self, cls: type, name: str, new_fn: Callable, deduplicate: bool = True
    ):
        """
        Replace object_or_dict.name with new_fn until we exit the context manager.
        """
        new_fn.__fx_already_patched = deduplicate  # type: ignore[attr-defined]
        orig_fn = getattr(cls, name)
        if getattr(orig_fn, "__fx_already_patched", False):
            return  # already patched, no need to do it again
        self.patches_made.append(_PatchedFnSetAttr(cls, name, orig_fn, new_fn))
        self.patches_made[-1].patch()

    def visit_once(self, thing: Any):
        """Return True on the first call to with thing, otherwise false"""
        idx = id(thing)
        if idx in self.visited:
            return False
        self.visited.add(idx)
        return True

    def revert_all_patches(self):
        """
        Remove all the stored patcheds. It doesn't modify patches_made.
        """
        for patch in self.patches_made:
            patch.revert()
        return self.patches_made

    def reapply_all_patches(self):
        """
        Patch all the stored patcheds. It doesn't modify patches_made.
        """
        for patch in self.patches_made:
            patch.patch()
        return self.patches_made

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Undo all the changes made via self.patch() and self.patch_method()
        """
        while self.patches_made:
            # unpatch in reverse order to handle duplicates correctly
            self.patches_made.pop().revert()
        self.visited.clear()


CURRENT_PATCHER: Optional[_Patcher] = None


@contextlib.contextmanager
def _new_patcher():
    global CURRENT_PATCHER
    prior_patcher = CURRENT_PATCHER
    try:
        CURRENT_PATCHER = _Patcher()
        yield CURRENT_PATCHER
    finally:
        # Clear all the patches made by when using current patcher.
        assert CURRENT_PATCHER is not None
        CURRENT_PATCHER.revert_all_patches()
        CURRENT_PATCHER = prior_patcher


@contextlib.contextmanager
def _maybe_revert_all_patches():
    current_patcher = CURRENT_PATCHER
    patches_made = None
    patches_removed = None
    try:
        if current_patcher is not None:
            patches_removed = current_patcher.revert_all_patches()
        yield
    finally:
        if current_patcher is not None:
            patches_made = current_patcher.reapply_all_patches()
        assert (
            patches_made == patches_removed
        ), "CURRENT_PATCHER was changed during a revert_all_patches"


def _patch_wrapped_functions(patcher: _Patcher):
    """
    Go through ``_wrapped_fn_patch_table`` and, for each frame object, wrap
    the listed global functions in the `_create_wrapped_func` wrapper.
    """
    for (_, name), frame_dict in _wrapped_fns_to_patch.copy().items():
        if name not in frame_dict and hasattr(builtins, name):
            orig_fn = getattr(builtins, name)
        else:
            orig_fn = frame_dict[name]
        patcher.patch(frame_dict, name, _create_wrapped_func(orig_fn))

    for cls, name in _wrapped_methods_to_patch:
        patcher.patch_method(cls, name, _create_wrapped_method(cls, name))


def _autowrap_check(
    patcher: _Patcher, frame_dict: dict[str, Any], function_ids: set[int]
):
    """
    Some methods, like `math.sqrt` are common enough we want to automatically wrap them as we see them.
    This method searches a scope for them and patches them if found.
    """
    if patcher.visit_once(frame_dict):
        for name, value in frame_dict.items():
            if (
                not name.startswith("_")
                and callable(value)
                and id(value) in function_ids
            ):
                patcher.patch(frame_dict, name, _create_wrapped_func(value))


@compatibility(is_backward_compatible=True)
def wrap(fn_or_name: Union[str, Callable]):
    """
    This function can be called at module-level scope to register fn_or_name as a "leaf function".
    A "leaf function" will be preserved as a CallFunction node in the FX trace instead of being
    traced through::

        # foo/bar/baz.py
        def my_custom_function(x, y):
            return x * x + y * y


        torch.fx.wrap("my_custom_function")


        def fn_to_be_traced(x, y):
            # When symbolic tracing, the below call to my_custom_function will be inserted into
            # the graph rather than tracing it.
            return my_custom_function(x, y)

    This function can also equivalently be used as a decorator::

        # foo/bar/baz.py
        @torch.fx.wrap
        def my_custom_function(x, y):
            return x * x + y * y

    A wrapped function can be thought of a "leaf function", analogous to the concept of
    "leaf modules", that is, they are functions that are left as calls in the FX trace
    rather than traced through.

    Args:

        fn_or_name (Union[str, Callable]): The function or name of the global function to insert into the
            graph when it's called
    """
    if not callable(fn_or_name) and not isinstance(fn_or_name, str):
        raise RuntimeError(
            "Unsupported type for global function! Must be either a callable or "
            "string name"
        )

    if callable(fn_or_name):
        assert not isinstance(fn_or_name, str)  # to make mypy happy
        fn_name = fn_or_name.__name__
    else:
        assert isinstance(
            fn_or_name, str
        ), "fn_or_name must be a global function or string name"
        fn_name = fn_or_name

    currentframe = inspect.currentframe()
    assert currentframe is not None
    f = currentframe.f_back
    assert f is not None
    if f.f_code.co_name != "<module>":
        raise NotImplementedError("wrap must be called at the top level of a module")

    # consider implementing Callable version of this via _autowrap_function_ids / _autowrap_search
    # semantics would be slightly different, but would add support `from x import wrapped_function`
    _wrapped_fns_to_patch[(id(f.f_globals), fn_name)] = f.f_globals
    return fn_or_name


@compatibility(is_backward_compatible=True)
def symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[dict[str, Any]] = None,
) -> GraphModule:
    """
    Symbolic tracing API

    Given an ``nn.Module`` or function instance ``root``, this function will return a ``GraphModule``
    constructed by recording operations seen while tracing through ``root``.

    ``concrete_args`` allows you to partially specialize your function, whether it's to remove control flow or data structures.

    For example::

        def f(a, b):
            if b == True:
                return a
            else:
                return a * 2

    FX can typically not trace through this due to the presence of control
    flow. However, we can use `concrete_args` to specialize on the value of
    `b` to trace through this::

        f = fx.symbolic_trace(f, concrete_args={"b": False})
        assert f(3, False) == 6

    Note that although you can still pass in different values of `b`, they will be ignored.

    We can also use `concrete_args` to eliminate data-structure handling from
    our function. This will use pytrees to flatten your input. To avoid
    overspecializing, pass in `fx.PH` for values that shouldn't be
    specialized. For example::

        def f(x):
            out = 0
            for v in x.values():
                out += v
            return out


        f = fx.symbolic_trace(f, concrete_args={"x": {"a": fx.PH, "b": fx.PH, "c": fx.PH}})
        assert f({"a": 1, "b": 2, "c": 4}) == 7


    Args:
        root (Union[torch.nn.Module, Callable]): Module or function to be traced and converted
            into a Graph representation.
        concrete_args (Optional[Dict[str, any]]): Inputs to be partially specialized

    Returns:
        GraphModule: a Module created from the recorded operations from ``root``.
    """
    tracer = Tracer()
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return _make_graph_module(tracer.root, graph, name)


@wrap
def _assert_is_none(value, msg):
    assert value is None, msg
