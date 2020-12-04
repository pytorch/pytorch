import torch
import torch.overrides
import linecache
from typing import Type, Dict, List, Any, Union
from .graph import Graph
import copy
import sys
import traceback
import math

# normal exec loses the source code, however we can patch
# the linecache module to still recover it.
# using exec_with_source will add it to our local cache
# and then tools like TorchScript will be able to get source info.
_next_id = 0
def exec_with_source(src: str, globals: Dict[str, Any]):
    global _next_id
    key = f'<eval_with_key_{_next_id}>'
    _next_id += 1
    _eval_cache[key] = [line + '\n' for line in src.splitlines()]
    exec(compile(src, key, 'exec'), globals)

# patch linecache so that any code we exec using exec_with_source
# works with inspect
_eval_cache : Dict[str, List[str]] = {}
_orig_getlines = linecache.getlines
def patched_getline(*args, **kwargs):
    if args[0] in _eval_cache:
        return _eval_cache[args[0]]
    return _orig_getlines(*args, **kwargs)
linecache.getlines = patched_getline

def _forward_from_src(src : str):
    gbls: Dict[str, Any] = {'inf': math.inf, 'nan': math.nan}
    exec_with_source(src, gbls)
    return gbls['forward']


def deserialize_graphmodule(body : dict) -> torch.nn.Module:
    """
    Deserialize a GraphModule given the dictionary of the original module,
    using the code to reconstruct the graph. We delete the actual graph before
    saving the dictionary so that changes to the in-memory graph format do not
    get serialized.
    """
    # We create a dummy class here because symbolic_trace pulls the forward()
    # function off of the class, rather than the instance
    class CodeOnlyModule(torch.nn.Module):
        def __init__(self, body):
            super().__init__()
            self.__dict__ = body

    try:
        CodeOnlyModule.forward = _forward_from_src(body['_code'])
    except KeyError:
        # BC: attribute name was changed from `code` to `_code` to facilitate
        # making `code` into a property and adding a docstring to it
        CodeOnlyModule.forward = _forward_from_src(body['code'])

    from .symbolic_trace import Tracer

    # we shouldn't trace into any of the submodules, they were not
    # because they were not traced in the original GraphModule
    class KeepModules(Tracer):
        def is_leaf_module(self, _: torch.nn.Module, __: str) -> bool:
            return True

    com = CodeOnlyModule(body)
    return GraphModule(com, KeepModules().trace(com))

# copy an attribute value with qualified name 'target' from 'from_module' to 'to_module'
# This installs empty Modules where none exist yet if they are subpaths of target
def _copy_attr(from_module: torch.nn.Module, to_module: torch.nn.Module, target: str):
    *prefix, field = target.split('.')
    for item in prefix:
        f = getattr(from_module, item)
        t = getattr(to_module, item, None)
        if f is t:
            # we have already installed one of its parents
            # (e.g. target = root.linear.weight, but we have already installed root.linear)
            # once we install a parent, we no longer need to copy the children
            # since all the needed properties will already be present
            return

        if t is None:
            t = torch.nn.Module()
            setattr(to_module, item, t)
        from_module, to_module = f, t

    orig = getattr(from_module, field)
    # If it is a tensor and not a parameter attribute of a module, it should be a named buffer.
    # So, we register it as a named buffer in the target module.
    if isinstance(orig, torch.Tensor) and not isinstance(orig, torch.nn.Parameter):
        to_module.register_buffer(field, orig)
    else:
        setattr(to_module, field, orig)


# Assign attribute 'from_obj' to the qualified name 'target' on 'to_module
# This installs empty Modules where none exist yet if they are subpaths of target
def _assign_attr(from_obj: Any, to_module: torch.nn.Module, target: str):
    *prefix, field = target.split('.')
    for item in prefix:
        t = getattr(to_module, item, None)

        if t is None:
            t = torch.nn.Module()
            setattr(to_module, item, t)
        to_module = t

    setattr(to_module, field, from_obj)

class GraphModule(torch.nn.Module):
    """
    GraphModule is an nn.Module generated from an fx.Graph. Graphmodule has a
    ``graph`` attribute, as well as ``code`` and ``forward`` attributes generated
    from that ``graph``.

    .. warning::

        When ``graph`` is reassigned, ``code`` and ``forward`` will be automatically
        regenerated. However, if you edit the contents of the ``graph`` without reassigning
        the ``graph`` attribute itself, you must call ``recompile()`` to update the generated
        code.

    """
    def __new__(cls: 'Type[GraphModule]', *args, **kwargs):
        # each instance of a graph module needs its own forward method
        # so create a new singleton class for each instance.
        # it is a subclass of the user-defined class, the only difference
        # is an extra layer to install the forward method

        class GraphModuleImpl(cls):  # type: ignore
            pass
        return super().__new__(GraphModuleImpl)

    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph):
        """
        Construct a GraphModule.

        Args:

            root (Union[torch.nn.Module, Dict[str, Any]):
                ``root`` can either be an nn.Module instance or a Dict mapping strings to any attribute type.
                In the case that ``root`` is a Module, any references to Module-based objects (via qualified
                name) in the Graph's Nodes' ``target`` field will be copied over from the respective place
                within ``root``'s Module hierarchy into the GraphModule's module hierarchy.
                In the case that ``root`` is a dict, the qualified name found in a Node's ``target`` will be
                looked up directly in the dict's keys. The object mapped to by the Dict will be copied
                over into the appropriate place within the GraphModule's module hierarchy.

            graph (Graph): ``graph`` contains the nodes this GraphModule should use for code generation

        """
        super().__init__()
        if isinstance(root, torch.nn.Module):
            if hasattr(root, 'training'):
                self.training = root.training
            for node in graph.nodes:
                if node.op in ['get_attr', 'call_module']:
                    assert isinstance(node.target, str)
                    _copy_attr(root, self, node.target)
        elif isinstance(root, dict):
            targets_to_copy = []
            for node in graph.nodes:
                if node.op in ['get_attr', 'call_module']:
                    assert isinstance(node.target, str)
                    if node.target not in root:
                        raise RuntimeError('Node ' + str(node) + ' referenced target ' + node.target +
                                           ' but that target was not provided in ``root``!')
                    targets_to_copy.append(node.target)
            # Sort targets in ascending order of the # of atoms.
            # This will ensure that less deeply nested attributes are assigned
            # before more deeply nested attributes. For example, foo.bar
            # will be assigned before foo.bar.baz. Otherwise, we might assign
            # the user-provided ``foo.bar`` and wipe out the previously-assigned
            # ``foo.bar.baz``
            targets_to_copy.sort(key=lambda t: t.count('.'))
            for target_to_copy in targets_to_copy:
                _assign_attr(root[target_to_copy], self, target_to_copy)
        else:
            raise RuntimeError('Unsupported type ' + str(root) + ' passed for root!')
        self.graph = graph

    # TorchScript breaks trying to compile the graph setter because of the
    # continued string literal. Issue here: https://github.com/pytorch/pytorch/issues/44842
    #
    # Shouldn't be an issue since these methods shouldn't be used in TorchScript anyway
    __jit_unused_properties__ = ['graph']

    @property
    def graph(self) -> Graph:
        """
        Return the ``Graph`` underlying this ``GraphModule``
        """
        return self._graph

    @graph.setter
    def graph(self, g) -> None:
        """
        Set the underlying ``Graph`` for this ``GraphModule``. This will internally
        recompile the ``GraphModule`` so that the generated ``forward()`` function
        corresponds to ``g``
        """
        self._graph = g
        self.recompile()

    @property
    def code(self) -> str:
        """
        Return the Python code generated from the ``Graph`` underlying this
        ``GraphModule``.
        """
        if not hasattr(self, '_code'):
            raise RuntimeError('Code has not been generated! Please report a bug to PyTorch')
        return self._code

    def recompile(self) -> None:
        """
        Recompile this GraphModule from its ``graph`` attribute. This should be
        called after editing the contained ``graph``, otherwise the generated
        code of this ``GraphModule`` will be out of date.
        """
        self._code = self._graph.python_code(root_module='self')
        cls = type(self)
        cls.forward = _forward_from_src(self._code)

        cls_call = cls.__call__

        def print_full_traceback(exctype, value, tb):
            traceback.print_exception(exctype, value, tb)

        def wrapped_call(self, *args, **kwargs):
            old_excepthook = sys.excepthook
            try:
                sys.excepthook = print_full_traceback
                return cls_call(self, *args, **kwargs)
            finally:
                sys.excepthook = old_excepthook
        cls.__call__ = wrapped_call

    def __reduce__(self):
        """
        Serialization of GraphModule. We serialize only the generated code, not
        the underlying ``Graph``. This is because ``Graph`` does not have on-disk
        backward-compatibility guarantees, whereas Python source code does.
        On the deserialization side, we symbolically trace through the generated
        code to regenerate the underlying ``Graph``
        """
        dict_without_graph = self.__dict__.copy()
        del dict_without_graph['_graph']
        return (deserialize_graphmodule, (dict_without_graph,))

    # because __reduce__ is defined for serialization,
    # we need to define deepcopy otherwise it will call __reduce__
    # and cause symbolic tracing to occur every time we try to copy the object
    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return GraphModule(fake_mod, self.graph)

    def __copy__(self):
        return GraphModule(self, self.graph)

    def __str__(self) -> str:
        orig_str = super().__str__()
        return '\n'.join([orig_str, self._code])

# workarounds for issues in __torch_function__

# WAR for __torch_function__ not handling tensor lists,
# fix is in https://github.com/pytorch/pytorch/pull/34725
# orig_cat = torch.cat
# def patched_cat(*args, **kwargs):
#     tensors = args[0]
#     for t in tensors:
#         if isinstance(t, Proxy):
#             return t.__torch_function__(patched_cat, (), args, kwargs)
#     return orig_cat(*args, **kwargs)
# patched_cat.__module__ = 'torch'
# patched_cat.__name__ = 'cat'
# torch.cat = patched_cat
