import builtins
import functools
import inspect
import math
import os
from types import CodeType, FunctionType, ModuleType
from typing import Any, Dict, NamedTuple, Optional, Set, Tuple, List, Callable, Union
from itertools import chain
import torch
from torch._C import ScriptObject  # type: ignore

from .node import Argument, map_aggregate
from .graph import Graph
from .graph_module import GraphModule
from .proxy import TracerBase, Proxy

HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS

# These need to run in global scope to handle nested calls correctly
_orig_module_call : Callable = torch.nn.Module.__call__
_orig_module_getattr : Callable = torch.nn.Module.__getattr__


def _patch_function(fn: FunctionType, nargs: int) -> FunctionType:
    co = fn.__code__
    co_flags = co.co_flags & ~HAS_VARSTUFF
    co_args : tuple
    if hasattr(co, "co_posonlyargcount"):
        co_args = (
            nargs, 0,
            0, co.co_nlocals, co.co_stacksize,
            co_flags, co.co_code, co.co_consts, co.co_names,
            co.co_varnames, co.co_filename, co.co_name,
            co.co_firstlineno, co.co_lnotab, co.co_freevars,
            co.co_cellvars
        )
    else:
        co_args = (
            nargs, 0, co.co_nlocals,
            co.co_stacksize, co_flags, co.co_code, co.co_consts,
            co.co_names, co.co_varnames, co.co_filename,
            co.co_name, co.co_firstlineno, co.co_lnotab,
            co.co_freevars, co.co_cellvars)
    new_code = CodeType(*co_args)  # type: ignore
    return FunctionType(new_code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)

    # we need to insert placeholder nodes for *args and **kwargs
    # we can't call this function normally, otherwise it would try to unpack them
    # instead, let's make python think that args and kwargs are normal variables

class Tracer(TracerBase):
    """
    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.

    Tracer can be subclassed to override various behaviors of the tracing
    process. The different behaviors that can be overridden are described
    in the docstrings of the methods on this class.
    """
    def __init__(self, autowrap_modules : Tuple[ModuleType] = (math, )):
        """
        Construct a Tracer object.

        Args:

            autowrap_modules (List[ModuleType]): defaults to `[math]`,
                Python modules whose functions should be wrapped automatically
                without needing to use fx.wrap().
        """

        super().__init__()

        # Functions we will eagerly wrap when we see them while tracing
        # this captures both `math.sqrt()` and `from math import sqrt` automatically
        self._autowrap_function_ids: Set[int] = {
            id(value) for name, value in chain(*[m.__dict__.items() for m in autowrap_modules])
            if not name.startswith("_") and callable(value)}

        # Python modules to apply autowrap to at the start, in addition to
        # modules we see while tracing
        self._autowrap_search: List[ModuleType] = list(autowrap_modules)


    def create_arg(self, a: Any) -> 'Argument':
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
                    return self.create_node('get_attr', n, (), {})
            raise NameError('parameter is not a member of this module')
        elif isinstance(a, torch.Tensor):
            for n_, p_ in self.root.named_buffers():
                if a is p_:
                    return self.create_node('get_attr', n_, (), {})

        # For NamedTuple instances that appear literally as args, we emit
        # a node to construct the NamedTuple and use that Node as the argument.
        if isinstance(a, tuple) and hasattr(a, '_fields'):
            args = tuple(self.create_arg(elem) for elem in a)
            return self.create_node('call_function', a.__class__, args, {})

        # Tensors do not have a reliable string repr() from which they can be
        # constructed (and we probably don't want to rely on that, either), so
        # for any constant Tensor values we encounter, first search for if they
        # are an attribute of some module in the module hierarchy. If so, emit
        # a get_attr to retrieve that tensor. Otherwise, we'll store away the
        # tensor value into a special attribute on the Module s.t. we can
        # retrieve it with a get_attr.
        if isinstance(a, (torch.Tensor, ScriptObject)):
            qualname : Optional[str] = self.tensor_attrs.get(a)

            # Tensor was not found in the Module hierarchy, stow it away in a
            # special attribute and set the qualname to refer to that
            if not qualname:
                i = 0
                while True:
                    qualname = f'_tensor_constant{i}'
                    if not hasattr(self.root, qualname):
                        break
                    i += 1
                setattr(self.root, qualname, a)

            return self.create_node('get_attr', qualname, (), {})
        return super().create_arg(a)

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
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
        return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)

    def path_of_module(self, mod : torch.nn.Module) -> str:
        """
        Helper method to find the qualified name of ``mod`` in the Module hierarchy
        of ``root``. For example, if ``root`` has a submodule named ``foo``, which has
        a submodule named ``bar``, passing ``bar`` into this function will return
        the string "foo.bar".

        Args:

            mod (str): The ``Module`` to retrieve the qualified name for.
        """
        for n, p in self.root.named_modules():
            if mod is p:
                return n
        raise NameError('module is not installed as a submodule')

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any], args : Tuple[Any, ...], kwargs : Dict[str, Any]) -> Any:
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
        if not self.is_leaf_module(m, module_qualified_name):
            return forward(*args, **kwargs)
        return self.create_proxy('call_module', module_qualified_name, args, kwargs)

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
        names_iter = iter(co.co_varnames)
        args : List[Any] = []
        skip_arg_idx = 0
        if is_module:
            if total_args == 0:
                raise RuntimeError('``self`` argument cannot be part of *args expansion!')
            skip_arg_idx = 1
            next(names_iter)  # skip self
            args.append(self.root)

        sig = inspect.signature(fn_for_analysis)

        def proxy_placeholder(name: str):
            if concrete_args is not None and name in concrete_args:
                return concrete_args[name]
            if name[0] == '*':
                default = ()    # type: ignore
            else:
                param = sig.parameters[name]
                default = () if param.default is inspect.Parameter.empty else (param.default,)  # type: ignore
            return self.create_proxy('placeholder', name, default, {},
                                     type_expr=fn_for_analysis.__annotations__.get(name, None))

        args.extend(proxy_placeholder(next(names_iter)) for _ in range(skip_arg_idx, total_args))

        if co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF:
            # TODO: type annotations for *args and **kwargs
            if co.co_flags & inspect.CO_VARARGS:
                args.append(proxy_placeholder('*' + next(names_iter)))
            if co.co_flags & inspect.CO_VARKEYWORDS:
                args.append(proxy_placeholder('**' + next(names_iter)))
            root_fn = _patch_function(root_fn, len(args))

        return root_fn, args

    def trace(self, root: Union[torch.nn.Module, Callable], concrete_args: Optional[Dict[str, Any]] = None) -> Graph:
        """
        Trace ``root`` and return the corresponding FX ``Graph`` representation. ``root``
        can either be an ``nn.Module`` instance or a Python callable.

        Note that after this call, ``self.root`` may be different from the ``root`` passed
        in here. For example, when a free function is passed to ``trace()``, we will
        create an ``nn.Module`` instance to use as the root and add embedded constants
        to.


        Args:

            root (Union[Module, Callable]): Either a ``Module`` or a function to be
                traced through.

        Returns:

            A ``Graph`` representing the semantics of the passed-in ``root``.
        """
        if isinstance(root, torch.nn.Module):
            self.root = root
            fn = type(root).forward
        else:
            self.root = torch.nn.Module()
            fn = root
        self.graph = Graph()

        # When we encounter a Tensor value that's not a parameter, we look if it
        # is some other attribute on the model. Construct a dict mapping Tensor
        # values to the qualified name here for efficiency. This is used downstream
        # in create_arg
        self.tensor_attrs : Dict[torch.Tensor, str] = {}

        def collect_tensor_attrs(m : torch.nn.Module, prefix_atoms : List[str]):
            for k, v in m.__dict__.items():
                if isinstance(v, (torch.Tensor, ScriptObject)):
                    self.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
            for k, v in m.named_children():
                collect_tensor_attrs(v, prefix_atoms + [k])

        collect_tensor_attrs(self.root, [])

        assert isinstance(fn, FunctionType)

        fn_globals = fn.__globals__  # run before it gets patched
        fn, args = self.create_args_for_root(fn, isinstance(root, torch.nn.Module), concrete_args)

        parameter_proxy_cache : Dict[str, Proxy] = {}  # Reduce number of get_attr calls

        # Method dispatch on parameters is not recorded unless it's directly used.
        # Thus, we need to insert a proxy when __getattr__ requests a parameter.
        @functools.wraps(_orig_module_getattr)
        def module_getattr_wrapper(mod, attr):
            attr_val = _orig_module_getattr(mod, attr)
            if isinstance(attr_val, torch.nn.Parameter):
                for n, p in self.root.named_parameters():
                    if attr_val is p:
                        if n not in parameter_proxy_cache:
                            parameter_proxy_cache[n] = self.create_proxy('get_attr', n, (), {})
                        return parameter_proxy_cache[n]
            return attr_val

        @functools.wraps(_orig_module_call)
        def module_call_wrapper(mod, *args, **kwargs):
            def forward(*args, **kwargs):
                return _orig_module_call(mod, *args, **kwargs)

            _autowrap_check(patcher, getattr(getattr(mod, "forward", mod), "__globals__", {}),
                            self._autowrap_function_ids)
            return self.call_module(mod, forward, args, kwargs)

        with _Patcher() as patcher:
            # allow duplicate patches to support the case of nested calls
            patcher.patch_method(torch.nn.Module, "__getattr__", module_getattr_wrapper, deduplicate=False)
            patcher.patch_method(torch.nn.Module, "__call__", module_call_wrapper, deduplicate=False)
            _patch_wrapped_functions(patcher)
            _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
            for module in self._autowrap_search:
                _autowrap_check(patcher, module.__dict__, self._autowrap_function_ids)

            self.create_node('output', 'output', (self.create_arg(fn(*args)),), {},
                             type_expr=fn.__annotations__.get('return', None))

        return self.graph


# List of pairs of (global dict, function name) functions
# to patch for the purposes of the wrap() API.
_wrapped_fns_to_patch : List[Tuple[dict, str]] = []

# List of methods on classes to wrap (class type, function name)
# this currently only works for Tensor.* methods that aren't traced properly
_wrapped_methods_to_patch : List[Tuple[type, str]] = []

if os.environ.get("FX_PATCH_GETITEM") == "1":
    # This change is needed to trace models like PositionalEmbedding from BERT:
    # https://github.com/pytorch/benchmark/blob/master/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/position.py  # noqa
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
            return proxy.tracer.create_proxy('call_function', orig_fn, args, kwargs)
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
            return proxy.tracer.create_proxy('call_method', name, args, kwargs)
        return orig_fn(*args, **kwargs)

    return wrapped


class _PatchedFn(NamedTuple):
    frame_dict : Any
    fn_name : str
    orig_fn : Any

    def revert(self):
        raise NotImplementedError()


class _PatchedFnSetItem(_PatchedFn):
    def revert(self):
        self.frame_dict[self.fn_name] = self.orig_fn


class _PatchedFnDel(_PatchedFn):
    def revert(self):
        del self.frame_dict[self.fn_name]


class _PatchedFnSetAttr(_PatchedFn):
    def revert(self):
        setattr(self.frame_dict, self.fn_name, self.orig_fn)


class _Patcher(object):
    def __init__(self):
        super(_Patcher, self).__init__()
        self.patches_made : List[_PatchedFn] = []
        self.visited : Set[int] = set()

    def patch(self, frame_dict : Dict[str, Any], name : str, new_fn : Callable,
              deduplicate : bool = True):
        """
        Replace frame_dict[name] with new_fn until we exit the context manager.
        """
        new_fn.__fx_already_patched = deduplicate  # type: ignore
        if name not in frame_dict and hasattr(builtins, name):
            self.patches_made.append(_PatchedFnDel(frame_dict, name, None))
        elif getattr(frame_dict[name], "__fx_already_patched", False):
            return  # already patched, no need to do it again
        else:
            self.patches_made.append(_PatchedFnSetItem(frame_dict, name, frame_dict[name]))
        frame_dict[name] = new_fn

    def patch_method(self, cls: type, name : str, new_fn : Callable,
                     deduplicate : bool = True):
        """
        Replace object_or_dict.name with new_fn until we exit the context manager.
        """
        new_fn.__fx_already_patched = deduplicate  # type: ignore
        orig_fn = getattr(cls, name)
        if getattr(orig_fn, "__fx_already_patched", False):
            return  # already patched, no need to do it again
        self.patches_made.append(_PatchedFnSetAttr(cls, name, orig_fn))
        setattr(cls, name, new_fn)

    def visit_once(self, thing: Any):
        """ Return True on the first call to with thing, otherwise false """
        idx = id(thing)
        if idx in self.visited:
            return False
        self.visited.add(idx)
        return True

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


def _patch_wrapped_functions(patcher : _Patcher):
    """
    Go through ``_wrapped_fn_patch_table`` and, for each frame object, wrap
    the listed global functions in the `_create_wrapped_func` wrapper.
    """
    for frame_dict, name in _wrapped_fns_to_patch:
        if name not in frame_dict and hasattr(builtins, name):
            orig_fn = getattr(builtins, name)
        else:
            orig_fn = frame_dict[name]
        patcher.patch(frame_dict, name, _create_wrapped_func(orig_fn))

    for cls, name in _wrapped_methods_to_patch:
        patcher.patch_method(cls, name, _create_wrapped_method(cls, name))


def _autowrap_check(patcher : _Patcher, frame_dict : Dict[str, Any], function_ids : Set[int]):
    """
    Some methods, like `math.sqrt` are common enough we want to automatically wrap them as we see them.
    This method searches a scope for them and patches them if found.
    """
    if patcher.visit_once(frame_dict):
        for name, value in frame_dict.items():
            if not name.startswith("_") and callable(value) and id(value) in function_ids:
                patcher.patch(frame_dict, name, _create_wrapped_func(value))


def wrap(fn_or_name : Union[str, Callable]):
    """
    This function can be called at module-level scope to register fn_or_name as a "leaf function".
    A "leaf function" will be preserved as a CallFunction node in the FX trace instead of being
    traced through::

        # foo/bar/baz.py
        def my_custom_function(x, y):
            return x * x + y * y

        torch.fx.wrap('my_custom_function')

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
        raise RuntimeError('Unsupported type for global function! Must be either a callable or '
                           'string name')

    if hasattr(fn_or_name, '__code__'):
        assert not isinstance(fn_or_name, str)  # to make mypy happy
        fn_name = fn_or_name.__code__.co_name
    else:
        assert isinstance(fn_or_name, str), "fn_or_name must be a global function or string name"
        fn_name = fn_or_name

    currentframe = inspect.currentframe()
    assert currentframe is not None
    f = currentframe.f_back
    assert f is not None
    if f.f_code.co_name != '<module>':
        raise NotImplementedError('wrap must be called at the top level of a module')

    # consider implementing Callable version of this via _autowrap_function_ids / _autowrap_search
    # semantics would be slightly different, but would add support `from x import wrapped_function`
    _wrapped_fns_to_patch.append((f.f_globals, fn_name))
    return fn_or_name

def symbolic_trace(root : Union[torch.nn.Module, Callable], concrete_args: Optional[Dict[str, Any]] = None) -> GraphModule:
    """Symbolic tracing API

    Given an ``nn.Module`` or function instance ``root``, this function will return a ``GraphModule``
    constructed by recording operations seen while tracing through ``root``.

    Args:
        root (Union[torch.nn.Module, Callable]): Module or function to be traced and converted
            into a Graph representation.
        concrete_args (Optional[Dict[str, any]]): Concrete arguments that should not be treated as Proxies.

    Returns:
        GraphModule: a Module created from the recorded operations from ``root``.

    """
    tracer = Tracer()
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return GraphModule(tracer.root, graph, name)
