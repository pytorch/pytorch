import copy
import itertools
import linecache
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

import torch
import torch.nn as nn
import torch.overrides
from torch.nn.modules.module import _addindent
from torch.package import Importer, PackageExporter, PackageImporter, sys_importer

from ._compatibility import compatibility
from .graph import _custom_builtins, _is_from_torch, _PyTreeCodeGen, Graph, PythonCode

__all__ = [
    "reduce_graph_module",
    "reduce_package_graph_module",
    "reduce_deploy_graph_module",
    "GraphModule",
]

_USER_PRESERVED_ATTRIBUTES_KEY = "_user_preserved_attributes"

# Normal exec loses the source code, however we can work with
# the linecache module to recover it.
# Using _exec_with_source will add it to our local cache
# and then tools like TorchScript will be able to get source info.
class _EvalCacheLoader:
    def __init__(self):
        self.eval_cache = {}
        self.next_id = 0

    def cache(self, src: str, globals: Dict[str, Any], co_fields=None):
        """Store the source in a private cache, and add a lazy entry in linecache
        that allows the source to be retrieved by 'filename'.

        Args:
            src (str): The module source to cache
            globals (dict): The module globals

        Returns:
            str: The cache key (and dummy filename) generated for src.
        """

        key = self._get_key()
        if co_fields:
            key += f" from {co_fields['co_filename']}:{co_fields['co_firstlineno']} in {co_fields['co_name']}"
        self.eval_cache[key] = src

        # Don't mutate globals so that this loader is only used
        # to populate linecache, and doesn't interact with other modules
        # that might check `__loader__`
        globals_copy = globals.copy()
        globals_copy["__file__"] = key
        globals_copy["__name__"] = key
        globals_copy["__loader__"] = self
        linecache.lazycache(key, globals_copy)

        return key

    # Part of the loader protocol (PEP 302)
    # linecache will use this method when trying to find source code
    def get_source(self, module_name) -> Optional[str]:
        if module_name in self.eval_cache:
            return self.eval_cache[module_name]
        return None

    def _get_key(self):
        key = f"<eval_with_key>.{self.next_id}"
        self.next_id += 1
        return key


_loader = _EvalCacheLoader()


def _exec_with_source(src: str, globals: Dict[str, Any], co_fields=None):
    key = _loader.cache(src, globals, co_fields)
    exec(compile(src, key, "exec"), globals)


def _forward_from_src(src: str, globals: Dict[str, Any], co_fields=None):
    # avoid mutating the passed in dict
    globals_copy = globals.copy()
    _exec_with_source(src, globals_copy, co_fields)
    forward_fn = globals_copy["forward"]
    del globals_copy["forward"]
    return forward_fn


def _format_import_statement(name: str, obj: Any, importer: Importer) -> str:
    if name in _custom_builtins:
        return _custom_builtins[name].import_str
    if _is_from_torch(name):
        return "import torch"
    module_name, attr_name = importer.get_name(obj)
    return f"from {module_name} import {attr_name} as {name}"


def _format_import_block(globals: Dict[str, Any], importer: Importer):
    import_strs: Set[str] = set()
    for name, obj in globals.items():
        import_strs.add(_format_import_statement(name, obj, importer))
    # Sort the imports so we have a stable import block that allows us to
    # hash the graph module and get a consistent key for use in a cache.
    return "\n".join(sorted(import_strs))


@compatibility(is_backward_compatible=True)
def reduce_graph_module(body: Dict[Any, Any], import_block: str) -> torch.nn.Module:
    # BC: attribute name was changed from `code` to `_code` to facilitate
    # making `code` into a property and adding a docstring to it
    fn_src = body.get("_code") or body["code"]
    forward = _forward_from_src(import_block + fn_src, {})
    return _deserialize_graph_module(forward, body)


@compatibility(is_backward_compatible=True)
def reduce_package_graph_module(
    importer: PackageImporter, body: Dict[Any, Any], generated_module_name: str
) -> torch.nn.Module:
    forward = importer.import_module(generated_module_name).forward
    return _deserialize_graph_module(forward, body)


@compatibility(is_backward_compatible=True)
def reduce_deploy_graph_module(
    importer: PackageImporter, body: Dict[Any, Any], import_block: str
) -> torch.nn.Module:
    ns = {}
    ns["__builtins__"] = importer.patched_builtins
    fn_src = body.get("_code")
    assert fn_src is not None
    forward = _forward_from_src(import_block + fn_src, ns)
    return _deserialize_graph_module(forward, body)


# We create a dummy class here because symbolic_trace pulls the forward()
# function off of the class, rather than the instance. This class is used
# in _deserialize_graph_module() below.
class _CodeOnlyModule(torch.nn.Module):
    def __init__(self, body):
        super().__init__()
        self.__dict__ = body


def _deserialize_graph_module(forward, body: Dict[Any, Any]) -> torch.nn.Module:
    """
    Deserialize a GraphModule given the dictionary of the original module,
    using the code to reconstruct the graph. We delete the actual graph before
    saving the dictionary so that changes to the in-memory graph format do not
    get serialized.
    """

    # Try to retrieve the forward source in a backward-compatible way
    _CodeOnlyModule.forward = forward

    tracer_cls = body.get("_tracer_cls")
    if tracer_cls is None:
        from ._symbolic_trace import Tracer

        tracer_cls = Tracer

    graphmodule_cls_name = body.get("_graphmodule_cls_name", "GraphModule")

    # This is a workaround for a mypy linter issue related to
    # passing base class as an argument - https://github.com/python/mypy/issues/5865.
    cls_tracer: Any = tracer_cls

    class KeepModules(cls_tracer):
        # we shouldn't trace into any of the submodules,
        # because they were not traced in the original GraphModule
        def is_leaf_module(self, _: torch.nn.Module, __: str) -> bool:
            return True

    com = _CodeOnlyModule(body)

    tracer_extras = body.get("_tracer_extras", {})
    graph = KeepModules().trace(com, **tracer_extras)

    # Manually set Tracer class on the reconstructed Graph, to avoid
    # referencing the private local subclass KeepModules.
    graph._tracer_cls = tracer_cls
    gm = GraphModule(com, graph, class_name=graphmodule_cls_name)

    # The GraphModule constructor only retains attributes referenced by the graph.
    # In this case, our goal is return a GraphModule as close to identical as the one
    # put into the package. If any additional attributes were present in body,
    # we should keep them.
    for k, v in body.items():
        if not hasattr(gm, k):
            setattr(gm, k, v)
    return gm


# copy an attribute value with qualified name 'target' from 'from_module' to 'to_module'
# This installs empty Modules where none exist yet if they are subpaths of target
def _copy_attr(from_module: torch.nn.Module, to_module: torch.nn.Module, target: str):
    *prefix, field = target.split(".")
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
    *prefix, field = target.split(".")
    for item in prefix:
        t = getattr(to_module, item, None)

        if t is None:
            t = torch.nn.Module()
            setattr(to_module, item, t)
        to_module = t

    # If it is a tensor and not a parameter attribute of a module, it should be a named buffer.
    # So, we register it as a named buffer in the target module.
    if isinstance(from_obj, torch.Tensor) and not isinstance(
        from_obj, torch.nn.Parameter
    ):
        to_module.register_buffer(field, from_obj)
    else:
        setattr(to_module, field, from_obj)


class _WrappedCall:
    def __init__(self, cls, cls_call):
        self.cls = cls
        self.cls_call = cls_call

    # Previously, if an error occurred when valid
    # symbolically-traced code was run with an invalid input, the
    # user would see the source of the error as coming from
    # `File "<eval_with_key_N">`, where N is some number. We use
    # this function to generate a more informative error message. We
    # return the traceback itself, a message explaining that the
    # error occurred in a traced Module's generated forward
    # function, and five lines of context surrounding the faulty
    # line
    @staticmethod
    def _generate_error_message(frame_summary: traceback.FrameSummary) -> str:
        # auxiliary variables (for readability)
        err_lineno = frame_summary.lineno
        assert err_lineno is not None
        line = frame_summary.line
        assert line is not None
        err_line_len = len(line)
        all_src_lines = linecache.getlines(frame_summary.filename)

        # constituent substrings of the error message
        tb_repr = traceback.format_exc()
        custom_msg = (
            "Call using an FX-traced Module, "
            f"line {err_lineno} of the traced Module's "
            "generated forward function:"
        )
        before_err = "".join(all_src_lines[err_lineno - 2 : err_lineno])
        marker = "~" * err_line_len + "~~~ <--- HERE"
        err_and_after_err = "\n".join(all_src_lines[err_lineno : err_lineno + 2])

        # joined message
        return "\n".join([tb_repr, custom_msg, before_err, marker, err_and_after_err])

    def __call__(self, obj, *args, **kwargs):
        try:
            if self.cls_call is not None:
                return self.cls_call(obj, *args, **kwargs)
            else:
                return super(self.cls, obj).__call__(*args, **kwargs)  # type: ignore[misc]
        except Exception as e:
            assert e.__traceback__
            topmost_framesummary: traceback.FrameSummary = (
                traceback.StackSummary.extract(traceback.walk_tb(e.__traceback__))[-1]
            )  # type: ignore[arg-type]
            if "eval_with_key" in topmost_framesummary.filename:
                print(
                    _WrappedCall._generate_error_message(topmost_framesummary),
                    file=sys.stderr,
                )
                raise e.with_traceback(None)
            else:
                raise e


@compatibility(is_backward_compatible=True)
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

    def __new__(cls: "Type[GraphModule]", *args, **kwargs):
        # each instance of a graph module needs its own forward method
        # so create a new singleton class for each instance.
        # it is a subclass of the user-defined class, the only difference
        # is an extra layer to install the forward method

        # address issue described at https://github.com/pytorch/pytorch/issues/63883
        # in other words, traverse class hierarchy to fix the redundant class definition problem
        for t in cls.__mro__:
            c = t.__qualname__.split(".")[-1]
            if c != "GraphModuleImpl":
                cls = t
                break

        class GraphModuleImpl(cls):  # type: ignore[misc, valid-type]
            pass

        return super().__new__(GraphModuleImpl)

    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: Graph,
        class_name: str = "GraphModule",
    ):
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

            class_name (str): ``name`` denotes the name of this GraphModule for debugging purposes. If it's unset, all
                error messages will report as originating from ``GraphModule``. It may be helpful to set this
                to ``root``'s original name or a name that makes sense within the context of your transform.
        """
        super().__init__()
        self.__class__.__name__ = class_name
        if isinstance(root, torch.nn.Module):
            if hasattr(root, "training"):
                self.training = root.training

            # When we pickle/unpickle graph module, we don't want to drop any module or attributes.
            if isinstance(root, _CodeOnlyModule):
                for k, _ in root.named_children():
                    _copy_attr(root, self, k)

                for k, _ in root.named_buffers():
                    _copy_attr(root, self, k)

                for k, _ in root.named_parameters():
                    _copy_attr(root, self, k)

            for node in graph.nodes:
                if node.op in ["get_attr", "call_module"]:
                    assert isinstance(node.target, str)
                    _copy_attr(root, self, node.target)
        elif isinstance(root, dict):
            targets_to_copy = []
            for node in graph.nodes:
                if node.op in ["get_attr", "call_module"]:
                    assert isinstance(node.target, str)
                    if node.target not in root:
                        raise RuntimeError(
                            "Node "
                            + str(node)
                            + " referenced target "
                            + node.target
                            + " but that target was not provided in ``root``!"
                        )
                    targets_to_copy.append(node.target)
            # Sort targets in ascending order of the # of atoms.
            # This will ensure that less deeply nested attributes are assigned
            # before more deeply nested attributes. For example, foo.bar
            # will be assigned before foo.bar.baz. Otherwise, we might assign
            # the user-provided ``foo.bar`` and wipe out the previously-assigned
            # ``foo.bar.baz``
            targets_to_copy.sort(key=lambda t: t.count("."))
            for target_to_copy in targets_to_copy:
                _assign_attr(root[target_to_copy], self, target_to_copy)
        else:
            raise RuntimeError("Unsupported type " + str(root) + " passed for root!")

        self.graph = graph

        # Store the Tracer class responsible for creating a Graph separately as part of the
        # GraphModule state, except when the Tracer is defined in a local namespace.
        # Locally defined Tracers are not pickleable. This is needed because torch.package will
        # serialize a GraphModule without retaining the Graph, and needs to use the correct Tracer
        # to re-create the Graph during deserialization.
        self._tracer_cls = None
        if (
            self.graph._tracer_cls
            and "<locals>" not in self.graph._tracer_cls.__qualname__
        ):
            self._tracer_cls = self.graph._tracer_cls

        self._tracer_extras = {}
        if self.graph._tracer_extras:
            self._tracer_extras = self.graph._tracer_extras

        # Dictionary to store metadata
        self.meta: Dict[str, Any] = {}

    # TorchScript breaks trying to compile the graph setter because of the
    # continued string literal. Issue here: https://github.com/pytorch/pytorch/issues/44842
    #
    # Shouldn't be an issue since these methods shouldn't be used in TorchScript anyway
    __jit_unused_properties__ = ["graph"]

    @property
    def graph(self) -> Graph:
        """
        Return the ``Graph`` underlying this ``GraphModule``
        """
        return self._graph

    @graph.setter
    def graph(self, g: Graph) -> None:
        """
        Set the underlying ``Graph`` for this ``GraphModule``. This will internally
        recompile the ``GraphModule`` so that the generated ``forward()`` function
        corresponds to ``g``
        """
        assert isinstance(g, Graph), f"Expected a Graph instance, but got {type(g)}"
        self._graph = g
        g.owning_module = self
        self.recompile()

    @compatibility(is_backward_compatible=False)
    def to_folder(self, folder: Union[str, os.PathLike], module_name: str = "FxModule"):
        """Dumps out module to ``folder`` with ``module_name`` so that it can be
        imported with ``from <folder> import <module_name>``

        Args:

            folder (Union[str, os.PathLike]): The folder to write the code out to

            module_name (str): Top-level name to use for the ``Module`` while
                writing out the code
        """
        folder = Path(folder)
        Path(folder).mkdir(exist_ok=True)
        torch.save(self.state_dict(), folder / "state_dict.pt")
        tab = " " * 4
        custom_builtins = "\n".join([v.import_str for v in _custom_builtins.values()])
        model_str = f"""
import torch
{custom_builtins}

from torch.nn import *
class {module_name}(torch.nn.Module):
    def __init__(self):
        super().__init__()
"""

        def _gen_model_repr(module_name: str, module: torch.nn.Module) -> Optional[str]:
            safe_reprs = [
                nn.Linear,
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
            ]
            if type(module) in safe_reprs:
                return f"{module.__repr__()}"
            else:
                return None

        blobified_modules = []
        for module_name, module in self.named_children():
            module_str = _gen_model_repr(module_name, module)
            if module_str is None:
                module_file = folder / f"{module_name}.pt"
                torch.save(module, module_file)
                blobified_modules.append(module_name)
                module_repr = module.__repr__().replace("\r", " ").replace("\n", " ")
                module_str = f"torch.load(r'{module_file}') # {module_repr}"
            model_str += f"{tab*2}self.{module_name} = {module_str}\n"

        for buffer_name, buffer in self._buffers.items():
            if buffer is None:
                continue
            model_str += f"{tab*2}self.register_buffer('{buffer_name}', torch.empty({list(buffer.shape)}, dtype={buffer.dtype}))\n"

        for param_name, param in self._parameters.items():
            if param is None:
                continue
            model_str += f"{tab*2}self.{param_name} = torch.nn.Parameter(torch.empty({list(param.shape)}, dtype={param.dtype}))\n"

        model_str += (
            f"{tab*2}self.load_state_dict(torch.load(r'{folder}/state_dict.pt'))\n"
        )
        model_str += f"{_addindent(self.code, 4)}\n"

        module_file = folder / "module.py"
        module_file.write_text(model_str)

        init_file = folder / "__init__.py"
        init_file.write_text("from .module import *")

        if len(blobified_modules) > 0:
            warnings.warn(
                "Was not able to save the following children modules as reprs -"
                f"saved as pickled files instead: {blobified_modules}"
            )

    @compatibility(is_backward_compatible=True)
    def add_submodule(self, target: str, m: torch.nn.Module) -> bool:
        """
        Adds the given submodule to ``self``.

        This installs empty Modules where none exist yet if they are
        subpaths of ``target``.

        Args:
            target: The fully-qualified string name of the new submodule
                (See example in ``nn.Module.get_submodule`` for how to
                specify a fully-qualified string.)
            m: The submodule itself; the actual object we want to
                install in the current Module

        Return:
            bool: Whether or not the submodule could be inserted. For
                this method to return True, each object in the chain
                denoted by ``target`` must either a) not exist yet,
                or b) reference an ``nn.Module`` (not a parameter or
                other attribute)
        """
        *prefix, field = target.split(".")
        mod: torch.nn.Module = self

        for item in prefix:

            submod = getattr(mod, item, None)

            if submod is None:
                submod = torch.nn.Module()
                setattr(mod, item, submod)

            if not isinstance(submod, torch.nn.Module):
                return False

            mod = submod

        mod.add_module(field, m)
        return True

    @compatibility(is_backward_compatible=True)
    def delete_submodule(self, target: str) -> bool:
        """
        Deletes the given submodule from ``self``.

        The module will not be deleted if ``target`` is not a valid
        target.

        Args:
            target: The fully-qualified string name of the new submodule
                (See example in ``nn.Module.get_submodule`` for how to
                specify a fully-qualified string.)

        Returns:
            bool: Whether or not the target string referenced a
                submodule we want to delete. A return value of ``False``
                means that the ``target`` was not a valid reference to
                a submodule.
        """
        atoms = target.split(".")
        path, target_submod = atoms[:-1], atoms[-1]
        mod: torch.nn.Module = self

        # Get the parent module
        for item in path:

            if not hasattr(mod, item):
                return False

            mod = getattr(mod, item)

            if not isinstance(mod, torch.nn.Module):
                return False

        if not hasattr(mod, target_submod):
            return False

        if not isinstance(getattr(mod, target_submod), torch.nn.Module):
            return False

        delattr(mod, target_submod)
        return True

    @compatibility(is_backward_compatible=True)
    def delete_all_unused_submodules(self) -> None:
        """
        Deletes all unused submodules from ``self``.

        A Module is considered "used" if any one of the following is
        true:
        1. It has children that are used
        2. Its forward is called directly via a ``call_module`` node
        3. It has a non-Module attribute that is used from a
        ``get_attr`` node

        This method can be called to clean up an ``nn.Module`` without
        manually calling ``delete_submodule`` on each unused submodule.
        """
        used: List[str] = []

        for node in self.graph.nodes:

            if node.op == "call_module" or node.op == "get_attr":

                # A list of strings representing the different parts
                # of the path. For example, `foo.bar.baz` gives us
                # ["foo", "bar", "baz"]
                fullpath = node.target.split(".")

                # If we're looking at multiple parts of a path, join
                # join them with a dot. Otherwise, return that single
                # element without doing anything to it.
                def join_fn(x: str, y: str) -> str:
                    return ".".join([x, y] if y else [x])

                # Progressively collect all the names of intermediate
                # modules. For example, if we have the target
                # `foo.bar.baz`, we'll add `foo`, `foo.bar`, and
                # `foo.bar.baz` to the list.
                for path in itertools.accumulate(fullpath, join_fn):
                    used.append(path)

                # For a `call_module` node, also register all recursive submodules
                # as used
                if node.op == "call_module":
                    try:
                        submod = self.get_submodule(node.target)

                        for submod_name, _ in submod.named_modules():
                            if submod_name != "":
                                used.append(".".join([node.target, submod_name]))
                    except AttributeError:
                        # Node referenced nonexistent submodule, don't need to
                        # worry about GCing anything
                        pass

        to_delete = [name for name, _ in self.named_modules() if name not in used]

        for name in to_delete:
            self.delete_submodule(name)

    @property
    def code(self) -> str:
        """
        Return the Python code generated from the ``Graph`` underlying this
        ``GraphModule``.
        """
        if not hasattr(self, "_code"):
            raise RuntimeError(
                "Code has not been generated! Please report a bug to PyTorch"
            )
        return self._code

    @compatibility(is_backward_compatible=True)
    def recompile(self) -> PythonCode:
        """
        Recompile this GraphModule from its ``graph`` attribute. This should be
        called after editing the contained ``graph``, otherwise the generated
        code of this ``GraphModule`` will be out of date.
        """
        if isinstance(self._graph._codegen, _PyTreeCodeGen):
            self._in_spec = self._graph._codegen.pytree_info.in_spec
            self._out_spec = self._graph._codegen.pytree_info.out_spec
        python_code = self._graph.python_code(root_module="self")
        self._code = python_code.src
        self._lineno_map = python_code._lineno_map

        cls = type(self)
        co_fields = self._graph._co_fields if hasattr(self._graph, "_co_fields") else {}
        cls.forward = _forward_from_src(self._code, python_code.globals, co_fields)

        # Determine whether this class explicitly defines a __call__ implementation
        # to wrap. If it does, save it in order to have wrapped_call invoke it.
        # If it does not, wrapped_call can use a dynamic call to super() instead.
        # In most cases, super().__call__ should be torch.nn.Module.__call__.
        # We do not want to hold a reference to Module.__call__ here; doing so will
        # bypass patching of torch.nn.Module.__call__ done while symbolic tracing.
        cls_call = cls.__call__ if "__call__" in vars(cls) else None

        if "_wrapped_call" not in vars(cls):
            cls._wrapped_call = _WrappedCall(cls, cls_call)  # type: ignore[attr-defined]

        def call_wrapped(self, *args, **kwargs):
            return self._wrapped_call(self, *args, **kwargs)

        cls.__call__ = call_wrapped

        return python_code

    # Passing Tracer as argument allows subclasses extending fx.GraphModule
    # define their own Tracer (extending fx.Tracer).
    def __reduce_deploy__(self, importer: Importer):
        dict_without_graph = self.__dict__.copy()
        dict_without_graph["_graphmodule_cls_name"] = self.__class__.__name__
        del dict_without_graph["_graph"]

        python_code = self.recompile()
        import_block = _format_import_block(python_code.globals, importer)
        return (reduce_deploy_graph_module, (dict_without_graph, import_block))

    def __reduce_package__(self, exporter: PackageExporter):
        dict_without_graph = self.__dict__.copy()
        dict_without_graph["_graphmodule_cls_name"] = self.__class__.__name__
        del dict_without_graph["_graph"]

        generated_module_name = f"fx-generated._{exporter.get_unique_id()}"
        python_code = self.recompile()
        import_block = _format_import_block(python_code.globals, exporter.importer)
        module_code = import_block + self.code
        exporter.save_source_string(generated_module_name, module_code)
        return (
            reduce_package_graph_module,
            (dict_without_graph, generated_module_name),
        )

    def __reduce__(self):
        """
        Serialization of GraphModule. We serialize only the generated code, not
        the underlying ``Graph``. This is because ``Graph`` does not have on-disk
        backward-compatibility guarantees, whereas Python source code does.
        On the deserialization side, we symbolically trace through the generated
        code to regenerate the underlying ``Graph``
        """
        dict_without_graph = self.__dict__.copy()
        python_code = self.recompile()
        import_block = _format_import_block(python_code.globals, sys_importer)
        del dict_without_graph["_graph"]
        return (reduce_graph_module, (dict_without_graph, import_block))

    # because __reduce__ is defined for serialization,
    # we need to define deepcopy otherwise it will call __reduce__
    # and cause symbolic tracing to occur every time we try to copy the object
    def __deepcopy__(self, memo):
        res = type(self).__new__(type(self))
        memo[id(self)] = res
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__, memo)
        GraphModule.__init__(res, fake_mod, fake_mod.__dict__["_graph"])
        # hooks are lost during `GraphModule.__init__`, so we need to copy over
        # them explicitly, note right now we are only copying state_dict related
        # hooks, to reduce bc-related issues, we can copy forward/backward related
        # hooks in the future as well if needed
        extra_preserved_attrs = [
            "_state_dict_hooks",
            "_load_state_dict_pre_hooks",
            "_load_state_dict_post_hooks",
        ]
        for attr in extra_preserved_attrs:
            if attr in self.__dict__:
                setattr(res, attr, copy.deepcopy(self.__dict__[attr], memo))
        res.meta = copy.deepcopy(getattr(self, "meta", {}), memo)
        if _USER_PRESERVED_ATTRIBUTES_KEY in res.meta:
            for attr_name, attr in res.meta[_USER_PRESERVED_ATTRIBUTES_KEY].items():
                setattr(res, attr_name, attr)
        return res

    def __copy__(self):
        res = GraphModule(self, self.graph)
        res.meta = getattr(self, "meta", {})
        return res

    @compatibility(is_backward_compatible=False)
    def print_readable(self, print_output=True):
        """
        Return the Python code generated for current GraphModule and its children GraphModules
        """
        verbose_python_code = self._graph.python_code(root_module="self", verbose=True)
        module_code = verbose_python_code.src
        module_code = module_code.lstrip("\n")
        module_code = f"class {self._get_name()}(torch.nn.Module):\n" + module_code
        module_code = _addindent(module_code, 4)

        submodule_code_list = [""]
        for submodule in self.children():
            if isinstance(submodule, GraphModule):
                submodule_code_list.append(submodule.print_readable(print_output=False))
        submodule_code = "\n".join(submodule_code_list)
        submodule_code = _addindent(submodule_code, 4)

        output = module_code + submodule_code
        if print_output:
            print(module_code + submodule_code)
        return output

    def __str__(self) -> str:
        orig_str = super().__str__()
        print_readable_reminder = (
            "# To see more debug info, please use `graph_module.print_readable()`"
        )
        return "\n".join([orig_str, self._code, print_readable_reminder])

    def _replicate_for_data_parallel(self):
        new_gm = self.__copy__()
        new_gm._is_replica = True
        return new_gm


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
