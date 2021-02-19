import torch
import torch.nn as nn
import torch.overrides
from torch.nn.modules.module import _addindent
import linecache
from typing import Type, Dict, List, Any, Union, Optional
from .graph import Graph
import copy
import itertools
import sys
import traceback
import math
from pathlib import Path
import os
import warnings

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

def _forward_from_src(src: str):
    # If you add more globals here, remember to add their names to fx.graph._shadows_builtin_name!
    gbls: Dict[str, Any] = {'inf': math.inf, 'nan': math.nan, 'NoneType' : type(None)}
    exec_with_source(src, gbls)
    return gbls['forward']


def deserialize_graphmodule(body: Dict[Any, Any]) -> torch.nn.Module:
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

    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]], graph: Graph, class_name: str = 'GraphModule'):
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

            name (str): ``name`` denotes the name of this GraphModule for debugging purposes. If it's unset, all
                error messages will report as originating from ``GraphModule``. It may be helpful to set this
                to ``root``'s original name or a name that makes sense within the context of your transform.

        """
        super().__init__()
        self.__class__.__name__ = class_name
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
                self.insert_submodule(target_to_copy, root[target_to_copy])
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
        g._gm = self if not g._owners else None
        g._owners += 1
        self.recompile()

    def to_folder(self, folder: Union[str, os.PathLike], module_name : str = "FxModule"):
        """Dumps out module to ``folder`` with ``module_name`` so that it can be
        imported with ``from <folder> import <module_name>``

        Args:

            folder (Union[str, os.PathLike]): The folder to write the code out to

            module_name (str): Top-level name to use for the ``Module`` while
                writing out the code
        """
        folder = Path(folder)
        Path(folder).mkdir(exist_ok=True)
        torch.save(self.state_dict(), folder / 'state_dict.pt')
        tab = " " * 4
        model_str = f"""
import torch
from torch.nn import *
class {module_name}(torch.nn.Module):
    def __init__(self):
        super().__init__()
"""

        def _gen_model_repr(module_name: str, module: torch.nn.Module) -> Optional[str]:
            safe_reprs = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
            if type(module) in safe_reprs:
                return f"{module.__repr__()}"
            else:
                return None

        blobified_modules = []
        for module_name, module in self.named_children():
            module_str = _gen_model_repr(module_name, module)
            if module_str is None:
                module_file = folder / f'{module_name}.pt'
                torch.save(module, module_file)
                blobified_modules.append(module_name)
                module_repr = module.__repr__().replace('\r', ' ').replace('\n', ' ')
                module_str = f"torch.load(r'{module_file}') # {module_repr}"
            model_str += f"{tab*2}self.{module_name} = {module_str}\n"

        for buffer_name, buffer in self._buffers.items():
            model_str += f"{tab*2}self.register_buffer('{buffer_name}', torch.empty({list(buffer.shape)}))\n"

        for param_name, param in self._parameters.items():
            model_str += f"{tab*2}self.{param_name} = torch.nn.Parameter(torch.empty({list(buffer.shape)}))\n"

        model_str += f"{tab*2}self.load_state_dict(torch.load(r'{folder}/state_dict.pt'))\n"
        model_str += f"{_addindent(self.code, 4)}\n"

        module_file = folder / 'module.py'
        module_file.write_text(model_str)

        init_file = folder / '__init__.py'
        init_file.write_text('from .module import *')

        if len(blobified_modules) > 0:
            warnings.warn("Was not able to save the following children modules as reprs -"
                          f"saved as pickled files instead: {blobified_modules}")

    def has_submodule(self, target: str) -> bool:
        """
        Returns whether or not this GraphModule contains the submodule
        given by ``str``.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block::text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested 
        submodule ``net_b``, which itself has two submodules ``net_c`` 
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
        would call ``has_submodule("net_b.linear")``. To check whether
        we have the ``conv`` submodule, we would call
        ``has_submodule("net_b.net_c.conv")``.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            bool: Whether or not the target string referenced an
                existing submodule. Returns False if the target string
                resolves to something that is not an ``nn.Module``.
        """
        atoms: List[str] = target.split(".")

        for item in atoms:

            if not hasattr(self, item):
                return False

            self = getattr(self, item)

            if not isinstance(self, torch.nn.Module):
                return False

        return True

    def insert_submodule(self, target: str, m: torch.nn.Module) -> bool:
        """
        Adds the given submodule to ``self``.

        This installs empty Modules where none exist yet if they are 
        subpaths of ``target``.

        Args:
            target: The fully-qualified string name of the new submodule
                (See example in ``has_submodule`` for how to specify a
                fully-qualified string.)
            m: The submodule itself; the actual object we want to
                install in the current GraphModule

        Return:
            bool: Whether or not the submodule could be inserted. For
                this method to return True, each object in the chain
                denoted by ``target`` must either a) not exist yet,
                or b) reference an ``nn.Module`` (not a parameter or
                other attribute)

        """
        *prefix, field = target.split('.')
        mod: torch.nn.Module = self

        for item in prefix:

            submod = getattr(self, item, None)

            if submod is None:
                submod = torch.nn.Module()
                setattr(mod, item, submod)

            if not isinstance(submod, torch.nn.Module):
                return False

            mod = submod

        setattr(self, field, m)
        return True

    def delete_submodule(self, target: str) -> bool:
        """
        Deletes the given submodule from ``self``.

        The module will not be deleted if ``target`` is not a valid
        target.

        Args:
            target: The fully-qualified string name of the new submodule
                (See example in ``has_submodule`` for how to denote a
                fully-qualified string.)

        Returns:
            bool: Whether or not the target string referenced a
                submodule we want to delete. A return value of ``False``
                means that the ``target`` was not a valid reference to
                a submodule.
        """
        def find_and_delete(o: Any, target: str) -> bool:
            atoms = target.split(".", 1)
            prefix, path = atoms[0], atoms[1] if len(atoms) > 1 else None

            if not hasattr(o, prefix):
                return False

            # If the submodule we're looking for is on this layer
            if not path:
                if not isinstance(getattr(o, prefix), torch.nn.Module):
                    return False

                delattr(o, prefix)
                return True

            return find_and_delete(getattr(o, prefix), path)

        return find_and_delete(self, target)

    def delete_all_unused_submodules(self) -> None:
        """
        Deletes all unused submodules from ``self``.

        A Module is considered "used" if any one of the following is
        true:
        1. It has children that are used
        2. Its forward is called directly via a ``call_module`` node
        3. It has a non-Module attribute that is used from a 
           ``get_attr`` node

        This method can be called to clean up a GraphModule without
        manually calling ``delete_submodule`` on each unused submodule.
        """
        # Collect all the call_module and get_attr targets as well as 
        # the names of their intermediary modules. For example, if we 
        # have the target `foo.bar.baz`, we'll add `foo`, `foo.bar`,
        # and `foo.bar.baz` to the list
        used: List[str] = [name for node in self.graph.nodes 
                           if node.op == "call_module" or node.op == "get_attr"
                           for name in itertools.accumulate(node.target.split("."),
                                                            lambda x, y: x + "." + y if y else x)]

        to_delete = [name for name, _ in self.named_modules()
                     if name not in used]

        for name in to_delete:
            self.delete_submodule(name)

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

        # Previously, if an error occurred when valid
        # symbolically-traced code was run with an invalid input, the
        # user would see the source of the error as coming from
        # `File "<eval_with_key_N">`, where N is some number. We use
        # this function to generate a more informative error message. We
        # return the traceback itself, a message explaining that the
        # error occurred in a traced Module's generated forward
        # function, and five lines of context surrounding the faulty
        # line
        def generate_error_message(frame_summary: traceback.FrameSummary) -> str:
            # auxiliary variables (for readability)
            err_lineno = frame_summary.lineno
            err_line_len = len(frame_summary.line)
            all_src_lines = _eval_cache[frame_summary.filename]

            # constiuent substrings of the error message
            tb_repr = traceback.format_exc()
            custom_msg = ("Call using an FX-traced Module, "
                          f"line {err_lineno} of the traced Module’s "
                          "generated forward function:")
            before_err = "".join(all_src_lines[err_lineno - 2 : err_lineno])
            marker = "~" * err_line_len + "~~~ <--- HERE"
            err_and_after_err = "\n".join(all_src_lines[err_lineno : err_lineno + 2])

            # joined message
            return "\n".join([tb_repr, custom_msg, before_err, marker, err_and_after_err])

        def wrapped_call(self, *args, **kwargs):
            try:
                return cls_call(self, *args, **kwargs)
            except Exception as e:
                assert e.__traceback__
                topmost_framesummary: traceback.FrameSummary = \
                    traceback.StackSummary.extract(traceback.walk_tb(e.__traceback__))[-1]  # type: ignore
                if "eval_with_key" in topmost_framesummary.filename:
                    print(generate_error_message(topmost_framesummary),
                          file=sys.stderr)
                raise e.with_traceback(None)

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
