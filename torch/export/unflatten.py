# mypy: allow-untyped-defs
import abc
import copy
import logging
import operator
import re
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch._library.fake_class_registry import FakeScriptObject
from torch.export._tree_utils import reorder_kwargs
from torch.export.exported_program import (
    ConstantArgument,
    ExportedProgram,
    ExportGraphSignature,
    InputKind,
    ModuleCallSignature,
    SymBoolArgument,
    SymFloatArgument,
    SymIntArgument,
    TensorArgument,
)
from torch.fx._symbolic_trace import is_fx_tracing
from torch.fx.graph_module import _get_attr, _get_attr_via_attr_list, _print_readable
from torch.utils._pytree import GetAttrKey, SequenceKey

from ._remove_effect_tokens_pass import _remove_effect_tokens


log = logging.getLogger(__name__)


__all__ = [
    "FlatArgsAdapter",
    "InterpreterModule",
    "InterpreterModuleDispatcher",
    "UnflattenedModule",
    "unflatten",
]


class _AttrKind(Enum):
    PARAMETER = "parameter"
    BUFFER = "buffer"
    CONSTANT = "constant"
    MODULE = "module"


RUN_WITH_INTERPRETER = True


@contextmanager
def _disable_interpreter():
    global RUN_WITH_INTERPRETER
    old_flag = RUN_WITH_INTERPRETER
    RUN_WITH_INTERPRETER = False
    try:
        yield
    finally:
        RUN_WITH_INTERPRETER = old_flag


# Assign attribute 'from_obj' to the qualified name 'target' on 'to_module
# This installs empty Modules where none exist yet if they are subpaths of target
def _assign_attr(
    from_obj: Union[torch.Tensor, torch.ScriptObject, torch.nn.Module],
    to_module: torch.nn.Module,
    target: str,
    attr_kind: _AttrKind,
    persistent: bool = True,
):
    *prefix, field = target.split(".")
    # We need to generate all submodules of `to_module` that are at `prefix` and
    # variants of `prefix` that differ only by call name. All of these submodules
    # will then be assigned `from_obj` at `field` so that they can share this attribute.
    # For example, if target is foo.bar.f, foo has another call name foo@1,
    # and bar has other call names bar@1, bar@2, then we will assign f to
    # foo.bar, foo.bar@1, foo.bar@2, foo@1.bar, foo@1.bar@1, foo@1.bar@2.
    to_modules = {to_module}
    for item in prefix:
        ts: Set[torch.nn.Module] = set()
        for to_module in to_modules:
            if not hasattr(to_module, item):
                setattr(to_module, item, torch.nn.Module())
            ts.update(
                t_call  # type: ignore[misc]
                for k, t_call in to_module._modules.items()
                if _is_call_name(k, item)
            )
        to_modules = ts

    for to_module in to_modules:
        if attr_kind == _AttrKind.PARAMETER:
            assert isinstance(from_obj, torch.nn.Parameter)
            to_module.register_parameter(field, from_obj)
        elif attr_kind == _AttrKind.BUFFER:
            assert isinstance(from_obj, torch.Tensor)
            to_module.register_buffer(field, from_obj, persistent=persistent)
        elif attr_kind == _AttrKind.CONSTANT:
            assert not isinstance(
                from_obj, FakeScriptObject
            ), "FakeScriptObject should only exist during tracing."
            assert isinstance(
                from_obj,
                (
                    torch.Tensor,
                    torch.ScriptObject,
                ),
            )
            setattr(to_module, field, from_obj)
        elif attr_kind == _AttrKind.MODULE:
            assert isinstance(from_obj, torch.nn.Module)
            setattr(to_module, field, from_obj)


class InterpreterModule(torch.nn.Module):
    """A module that uses torch.fx.Interpreter to execute instead of the usual
    codegen that GraphModule uses. This provides better stack trace information
    and makes it easier to debug execution.
    """

    graph_module: Optional[torch.fx.GraphModule]

    def __init__(
        self,
        graph: torch.fx.Graph,
    ):
        super().__init__()
        self.graph = graph
        self.graph.owning_module = self
        self._run_with_interpreter = RUN_WITH_INTERPRETER

    def forward(self, *args, **kwargs):
        assert self.graph_module is not None, "Didn't finalize this InterpreterModule"
        if not is_fx_tracing() and (
            torch.compiler.is_dynamo_compiling() or not self._run_with_interpreter
        ):
            # Dynamo cannot trace through torch.fx.Interpreter, so fall back to
            # GraphModule codegen in this instance.
            # Patch the codegened forward to run with this InterpreterModule,
            # so attribute accesses, etc. are on this module instead.
            return type(self.graph_module).forward(self, *args, **kwargs)
        else:
            if kwargs:
                # Handle **kwargs. FX only natively supports positional
                # arguments (through placeholders). So in order to pass in
                # kwargs, we must correspond the names of the placeholders with
                # the keys in the kwarg dict.
                arg_list = list(args)
                kwarg_names = self.arg_names[len(arg_list) :]
                arg_list.extend(
                    kwargs[kwarg_name]
                    for kwarg_name in kwarg_names
                    if kwarg_name in kwargs
                )

                # Assert that the kwargs passed in exactly match the positional
                # arguments specified by the GraphModule. This should be
                # guaranteed by the unflattening process.
                assert len(kwarg_names) == len(kwargs)
                assert len(arg_list) == len(self.arg_names)
                args = tuple(arg_list)

            return torch.fx.Interpreter(self, graph=self.graph).run(
                *args, enable_io_processing=False
            )

    def finalize(self):
        # We need to "finalize" because GraphModule populates its own state_dict
        # based on the get_attrs observed in the graph. So we need to fully
        # construct the graph and call _sink_params before generating this
        # GraphModule.

        # need to set `graph_module` directly on the dict to avoid it getting
        # registered as a submodule.
        self.__dict__["graph_module"] = torch.fx.GraphModule(self, self.graph)
        self.graph.lint()

        # Cache arg names for kwarg handling (see forward())
        self.arg_names = []
        for node in self.graph.nodes:
            if node.op == "placeholder":
                self.arg_names.append(node.target)

    def print_readable(
        self,
        print_output=True,
        include_stride=False,
        include_device=False,
        colored=False,
    ):
        return _print_readable(
            self,
            "InterpreterModule",
            print_output,
            include_stride,
            include_device,
            colored,
        )


class InterpreterModuleDispatcher(torch.nn.Module):
    """
    A module that carries a sequence of InterpreterModules corresponding to
    a sequence of calls of that module. Each call to the module dispatches
    to the next InterpreterModule, and wraps back around after the last.
    """

    def __init__(self, attrs: Set[str], call_modules: List[InterpreterModule]):
        super().__init__()
        assert call_modules
        self._modules = call_modules[0]._modules
        for accessor in attrs:
            setattr(self, accessor, getattr(call_modules[0], accessor))
        self._call_modules = call_modules
        self._num_calls = 0

    def forward(self, *args, **kwargs):
        call_module = self._call_modules[self._num_calls]
        self._num_calls = (self._num_calls + 1) % len(self._call_modules)
        try:
            return call_module(*args, **kwargs)
        except Exception:
            self._num_calls = 0
            raise

    def call_modules(self):
        return self._call_modules

    def print_readable(
        self,
        print_output=True,
        include_stride=False,
        include_device=False,
        colored=False,
    ):
        outputs = [
            mod.print_readable(
                print_output,
                include_stride,
                include_device,
                colored,
            )
            for mod in self._call_modules
        ]
        return "\n".join(outputs)


class FlatArgsAdapter(abc.ABC):
    """
    Adapts input arguments with ``input_spec`` to align ``target_spec``.
    """

    @abc.abstractmethod
    def adapt(
        self,
        target_spec: pytree.TreeSpec,
        input_spec: pytree.TreeSpec,
        input_args: List[Any],
    ) -> List[Any]:
        """NOTE: This adapter may mutate given ``input_args_with_path``."""
        ...


class UnflattenedModule(torch.nn.Module):
    def __init__(
        self,
        export_module: ExportedProgram,
        flat_args_adapter: Optional[FlatArgsAdapter] = None,
    ):
        super().__init__()
        if export_module.graph_signature.backward_signature is not None:
            raise ValueError("Unflattening on JointExportModule NYI")

        fqn_list = [entry.fqn for entry in export_module.module_call_graph]
        assert fqn_list[0] == ""
        export_graph = deepcopy(export_module.graph)
        self.graph_signature = deepcopy(export_module.graph_signature)
        self.graph = torch.fx.Graph()
        self.graph.owning_module = self
        self.module_call_graph = deepcopy(export_module.module_call_graph)
        self.flat_args_adapter = flat_args_adapter
        # Flag to indicate whether args have been adapted.
        self.adapted = False
        self._run_with_interpreter = RUN_WITH_INTERPRETER

        _inplace_buffer_mutations(export_graph, self.graph_signature)

        self.ivals = _IVals()
        # record any intermediate value x that is used, with the modules that used it,
        # and generate instructions to read the corresponding attribute
        seen_modules, seen_attrs = _outline_submodules(export_graph, self)
        # for each read intermediate value x, find the module that created it,
        # and generate instructions to update the corresponding attribute;
        # finally, initialize all these attributes
        self.ivals.create(seen_modules.values(), self)
        # move attributes that correspond to graph arguments for HOPs
        # from exported program to unflattened submodules
        _copy_graph_attrs(export_module._graph_module, self, seen_attrs)

        self.range_constraints = export_module.range_constraints
        self.equality_constraints: List = []

        # aliasing/unused param or buffer issues:
        # in strict-mode export, dynamo export will deduplicate aliased tensors,
        # and ignore unused tensors. For aliasing, this causes issues when some aliases
        # are unused, and we're unable to match the placeholder node to the correct FQN.
        # This leads to the graph signature potentially having the wrong target FQN,
        # and downstream issues where parameters are assigned to the wrong target attribute,
        # mismatching the relevant placeholder node in the unflattened module.
        # To resolve this we restore (_assign_attr) all aliased/unused tensors in
        # the state_dict as module attributes, but only keep the used tensors in the
        # graph's forward pass (_sink_params).
        state_dict = export_module.state_dict
        assigned_params: Set[str] = set()  # tracking unused params
        id_to_param: Dict[int, torch.nn.Parameter] = {}  # handling weight-sharing
        for name in self.graph_signature.parameters:  # this loop adds used params
            param = state_dict[name]
            if id(param) not in id_to_param:
                id_to_param[id(param)] = torch.nn.Parameter(
                    param.clone(), requires_grad=param.requires_grad
                )

            _assign_attr(
                id_to_param[id(param)],
                self,
                name,
                attr_kind=_AttrKind.PARAMETER,
            )
            assigned_params.add(name)

        non_persistent_buffers = set(self.graph_signature.non_persistent_buffers)
        assigned_buffers: Set[str] = set()  # tracking unused buffers
        id_to_buffer: Dict[int, Tuple[torch.nn.Parameter, bool]] = {}
        for name in self.graph_signature.buffers:  # this loop adds used buffers
            if name in non_persistent_buffers:
                persistent = False
                buffer = export_module.constants[name]
            else:
                persistent = True
                buffer = state_dict[name]

            if id(buffer) not in id_to_buffer:
                id_to_buffer[id(buffer)] = (buffer.clone(), persistent)

            _assign_attr(
                id_to_buffer[id(buffer)][0],
                self,
                name,
                attr_kind=_AttrKind.BUFFER,
                persistent=persistent,
            )
            assigned_buffers.add(name)

        # restore aliased/unused params and buffers
        # these appear in state dict but not graph signature
        for name, tensor in state_dict.items():
            if name in assigned_params or name in assigned_buffers:  # already assigned
                continue

            is_buffer = False
            if id(tensor) in id_to_buffer or not isinstance(
                tensor, torch.nn.Parameter
            ):  # aliased buffer
                is_buffer = True

            if is_buffer:
                if (
                    id(tensor) not in id_to_buffer
                ):  # this is completely unused (not weight-sharing)
                    id_to_buffer[id(tensor)] = (
                        tensor,
                        True,
                    )  # assign to respect original model
                _assign_attr(
                    id_to_buffer[id(tensor)][0],
                    self,
                    name,
                    attr_kind=_AttrKind.BUFFER,
                    persistent=True,
                )
            else:
                if id(tensor) not in id_to_param:  # this is unused
                    id_to_param[id(tensor)] = tensor
                _assign_attr(
                    id_to_param[id(tensor)],
                    self,
                    name,
                    attr_kind=_AttrKind.PARAMETER,
                )

        # use id map so we don't double-clone aliased constants
        id_to_const: Dict[int, Union[torch.Tensor, torch._C.ScriptObject]] = {}
        for fqn, constant in export_module.constants.items():
            if id(constant) not in id_to_const:
                if isinstance(constant, torch.Tensor):
                    constant = constant.clone()
                id_to_const[id(constant)] = constant
            _constant = id_to_const[id(constant)]
            _assign_attr(
                _constant,
                self,
                fqn,
                attr_kind=_AttrKind.CONSTANT,
            )

        # This is to handle parameters/buffers that point to the same tensor
        # object id -> list of (node_name, target_name)
        consts_map: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
        consts_targets: Set[str] = set()

        def add_to_consts_map(obj_id, node_name, target_name):
            name_list = consts_map[obj_id]
            name_list.append((node_name, target_name))

        added_params_buffers: Set[str] = set()  # track aliased/unused params, buffers
        for s in self.graph_signature.input_specs:
            if s.kind == InputKind.PARAMETER or (
                s.kind == InputKind.BUFFER and s.persistent
            ):
                assert hasattr(s.arg, "name")
                assert isinstance(s.target, str)
                add_to_consts_map(
                    id(export_module.state_dict[s.target]), s.arg.name, s.target
                )
                consts_targets.add(s.target)
                added_params_buffers.add(s.target)
            elif (
                (s.kind == InputKind.BUFFER and not s.persistent)
                or s.kind == InputKind.CONSTANT_TENSOR
                or s.kind == InputKind.CUSTOM_OBJ
            ):
                assert hasattr(s.arg, "name")
                assert isinstance(s.target, str)
                add_to_consts_map(
                    id(export_module.constants[s.target]), s.arg.name, s.target
                )
                consts_targets.add(s.target)

        # add constants that are aliased and don't appear in graph signature
        for const_name, const in export_module.constants.items():
            if const_name not in consts_targets:
                assert (
                    id(const) in consts_map
                ), "Constants should be either aliased or appear in graph signature"
                ph_name, _ = consts_map[id(const)][0]
                add_to_consts_map(id(const), ph_name, const_name)
                added_params_buffers.add(s.target)

        # add aliased/unused params and buffers that don't appear in graph signature
        for fqn, tensor in export_module.state_dict.items():
            if fqn not in added_params_buffers:
                if id(tensor) not in consts_map:
                    # completely unused (no weight-sharing), ignore.
                    # this weight doesn't appear in graph module,
                    # so won't cause FQN assignment issues
                    continue
                ph_name, _ = consts_map[id(tensor)][0]
                add_to_consts_map(id(tensor), ph_name, fqn)

        # node name -> list of possible targets
        inputs_to_state: Dict[str, List[str]] = {}
        for node_target in consts_map.values():
            targets = [t[1] for t in node_target]
            for n, _ in node_target:
                inputs_to_state[n] = targets

        _sink_params(self, inputs_to_state, [])

        redirected_call_indices = _deduplicate_modules(seen_modules.values())
        fqn_list = [fqn for fqn in fqn_list if fqn not in redirected_call_indices]

        self._dispatch_modules(redirected_call_indices, consts_targets)
        fqn_list = [fqn for fqn in fqn_list if "@" not in fqn]

        # Cache so we don't have to compute this every time.
        # NOTE: this needs to be kept in sync with the placeholders in
        # self.graph, but currently we have no way to guarantee that.
        self.input_placeholders = [
            node for node in self.graph.nodes if node.op == "placeholder"
        ]
        self.check_input_constraints = True
        # TODO(zhxchen17) We can register modules ahead of time instead of reorder later.
        fqn_order = {fqn: i for i, fqn in enumerate(fqn_list)}
        # In the case of legacy IR, we might be missing some modules from metadata.
        for name, _ in self.named_modules(remove_duplicate=False):
            if name not in fqn_order:
                fqn_order[name] = len(fqn_order)
        _reorder_submodules(self, fqn_order)
        self.graph.lint()

    def _print_graph(self):
        for fqn, mod in self.named_modules():
            print(fqn + ":")
            if hasattr(mod, "graph") and isinstance(mod.graph, torch.fx.Graph):
                print(mod.graph)

    def _adapt_flat_args(self, flat_args, in_spec):
        signature = self.module_call_graph[0].signature
        if in_spec == signature.in_spec:
            return flat_args

        if self.flat_args_adapter is None:
            raise TypeError(
                "There is no flat args adapter sepcified. "
                "Are you sure you are calling this with the right arguments? "
            )
        else:
            flat_args = self.flat_args_adapter.adapt(
                target_spec=signature.in_spec,
                input_spec=in_spec,
                input_args=flat_args,
            )

            if len(flat_args) != signature.in_spec.num_leaves:
                raise TypeError(
                    f"Flat args adaption failed, number of args mismatch "
                    f"Adatped: {len(flat_args)} \n"
                    f"Exported module: {signature.in_spec.num_leaves}"
                )
            return flat_args

    def process_forward_inputs(self, *args, **kwargs):
        signature = self.module_call_graph[0].signature

        reordered_kwargs = reorder_kwargs(kwargs, signature.in_spec)

        flat_args_with_path, in_spec = pytree.tree_flatten_with_path(
            (args, reordered_kwargs)
        )
        flat_args = [x[1] for x in flat_args_with_path]

        if is_fx_tracing():
            return flat_args

        if in_spec != signature.in_spec:
            if not self.adapted:
                print(
                    "Input treespec does not match with exported module's: \n"
                    f"Input treespec: {in_spec}. ",
                    f"Exported module treespec: {signature.in_spec}",
                )
                print("Adapting flat arg to match exported module's treespec")
            flat_args = self._adapt_flat_args(flat_args, in_spec)
            self.adapted = True

        if self.check_input_constraints:
            # Import here to avoid an unfortunate circular dependency.
            # TODO(suo): untangle this.
            from torch._export.utils import _check_input_constraints_for_graph

            if self.adapted is True:
                # TODO(suo): The FlatArgsAdapter returns a list of flat args,
                # which we don't have keypaths for. For now, just create a dummy
                # keypath to associate with the arg.
                new_flat_args_with_path = [  # type: ignore[var-annotated]
                    ((SequenceKey(idx=0), GetAttrKey(name="<unknown location>")), arg)
                    for arg in flat_args
                ]
            else:
                new_flat_args_with_path = flat_args_with_path  # type: ignore[assignment]

            _check_input_constraints_for_graph(
                self.input_placeholders, new_flat_args_with_path, self.range_constraints
            )

        return flat_args

    def forward(self, *args, **kwargs):
        flat_args = torch._dynamo.disable(self.process_forward_inputs)(*args, **kwargs)
        signature = self.module_call_graph[0].signature

        if is_fx_tracing():
            return_val = torch.fx.Interpreter(self, graph=self.graph).run(
                *flat_args, enable_io_processing=False
            )
            # For scalar return value, fx.Graph wraps in a tuple
            if isinstance(return_val, tuple) and len(return_val) == 1:
                return return_val[0]
            return return_val

        if torch.compiler.is_dynamo_compiling() and not self._run_with_interpreter:
            tree_out = torch.fx.GraphModule(self, self.graph)(*flat_args)
        else:
            tree_out = torch.fx.Interpreter(self, graph=self.graph).run(
                *flat_args, enable_io_processing=False
            )
        return pytree.tree_unflatten(tree_out, signature.out_spec)

    def _dispatch_modules(self, redirected_call_indices, consts_targets):
        """For a module whose call signatures are preserved, replace
        multiple modules corresponding to multiple calls to that module
        with a single dispatcher module that tracks which module to call.
        """

        # for each fqn whose module call signature is preserved,
        # map that fqn to a list of called modules
        called_modules = defaultdict(list)
        for entry in self.module_call_graph:
            if entry.fqn and entry.signature:
                # some modules were removed and their fqns redirected to other
                # fqns during deduplication
                fqn = entry.fqn
                mod = _get_attr(self, redirected_call_indices.get(fqn, fqn))
                base, idx = fqn.split("@") if "@" in fqn else [fqn, "0"]
                called_modules[base].append((int(idx), mod))

        attrs_map = defaultdict(set)
        for target in consts_targets:
            if "." in target:
                orig_fqn, name = target.rsplit(".", 1)
                attrs_map[orig_fqn].add(name)
            else:
                attrs_map[""].add(target)

        # replace multiple call modules with a single dispatcher module
        for orig_fqn, indexed_call_modules in called_modules.items():
            call_modules = [mod for _, mod in sorted(indexed_call_modules)]
            if len(call_modules) > 1:
                for i, call_module in enumerate(call_modules):
                    fqn = _call_name(orig_fqn, i + 1)
                    if fqn not in redirected_call_indices:
                        *prefix, name = fqn.split(".")
                        _get_attr_via_attr_list(self, prefix)._modules.pop(name)
                self.set_submodule(
                    orig_fqn,
                    InterpreterModuleDispatcher(attrs_map[orig_fqn], call_modules),
                )

        # elide call indices in call modules because they are
        # tracked automatically inside the dispatcher module
        def elide_call_indices(prefix, graph):
            for node in graph.nodes:
                if node.op == "call_module":
                    fqn = node.target.split("@")[0]
                    path = f"{prefix}.{fqn}" if prefix else fqn
                    if path in called_modules:
                        node.target = fqn

        for fqn, mod in self.named_modules(remove_duplicate=False):
            if hasattr(mod, "graph"):
                elide_call_indices(fqn, mod.graph)
            elif hasattr(mod, "_call_modules"):
                for mod_ in mod._call_modules:
                    assert hasattr(mod_, "graph")
                    elide_call_indices(fqn, mod_.graph)

    def print_readable(
        self,
        print_output=True,
        include_stride=False,
        include_device=False,
        colored=False,
    ):
        return _print_readable(
            self,
            "UnflattenedModule",
            print_output,
            include_stride,
            include_device,
            colored,
        )


def unflatten(
    module: ExportedProgram, flat_args_adapter: Optional[FlatArgsAdapter] = None
) -> UnflattenedModule:
    """Unflatten an ExportedProgram, producing a module with the same module
    hierarchy as the original eager module. This can be useful if you are trying
    to use :mod:`torch.export` with another system that expects a module
    hierachy instead of the flat graph that :mod:`torch.export` usually produces.

    .. note:: The args/kwargs of unflattened modules will not necessarily match
        the eager module, so doing a module swap (e.g. :code:`self.submod =
        new_mod`) will not necessarily work. If you need to swap a module out, you
        need to set the :code:`preserve_module_call_signature` parameter of
        :func:`torch.export.export`.

    Args:
        module (ExportedProgram): The ExportedProgram to unflatten.
        flat_args_adapter (Optional[FlatArgsAdapter]): Adapt flat args if input TreeSpec does not match with exported module's.

    Returns:
        An instance of :class:`UnflattenedModule`, which has the same module
        hierarchy as the original eager module pre-export.
    """
    module = _remove_effect_tokens(module)
    return UnflattenedModule(module, flat_args_adapter)


def _inplace_buffer_mutations(
    graph: torch.fx.Graph,
    graph_signature: ExportGraphSignature,
) -> None:
    """Transform buffer mutations from their functionalized form into a copy_
    node in the graph.

    Functionalization represents buffer mutation by passing the buffer as an input and output. So for example, the eager code:
        def forward(self, x):
            self.buffer += x
            return x * x

    Will become a graph that looks like:
        def forward(self, buffer, x):
            mutated_buffer = aten.add(buffer, x)
            mul = aten.mul(x, x)
            return (mutated_buffer, mul)

    We want to inplace this into something that looks like the original eager code:
        def forward(self, buffer, x):
            mutated_buffer = aten.add(buffer, x)
            buffer.copy_(mutated_buffer)
            mul = aten.mul(x, x)
            return (mul,)
    """
    output_node = next(iter(reversed(graph.nodes)))
    assert output_node.op == "output" and len(output_node.args) == 1
    return_args = output_node.args[0]

    mutation_node_to_buffer = graph_signature.buffers_to_mutate
    mutations = return_args[: len(mutation_node_to_buffer)]
    buffers_to_inputs = {v: k for k, v in graph_signature.inputs_to_buffers.items()}
    input_name_to_node = {
        node.name: node for node in graph.nodes if node.op == "placeholder"
    }

    for mutation in mutations:
        buffer_name = mutation_node_to_buffer[mutation.name]
        input_name = buffers_to_inputs[buffer_name]
        input_node = input_name_to_node[input_name]

        with graph.inserting_after(mutation):
            new_node = graph.create_node(
                "call_function", torch.ops.aten.copy_, (input_node, mutation)
            )
            for k, v in mutation.meta.items():
                new_node.meta[k] = v
        # Replace all uses of the previously functional mutation with our copy_ output.
        mutation.replace_all_uses_with(new_node, lambda x: x is not new_node)

    # Remove the mutated buffer from the graph outputs, since we don't need to
    # thread it through anymore. We don't need to handle the inputs, which will
    # be handled by _sink_params.
    user_outputs = tuple(
        return_args[len(mutation_node_to_buffer) :],
    )
    output_node.args = ((user_outputs),)


def _is_prefix(candidate, target):
    """Check whether `candidate` is a prefix of `target`."""
    return len(candidate) < len(target) and target[: len(candidate)] == candidate


def _compute_accessor(parent_fqn: str, child_fqn: str) -> str:
    if parent_fqn == "":
        # Handle the root module correctly.
        return child_fqn

    parent_split = parent_fqn.split(".")
    child_split = child_fqn.split(".")

    # TODO: support skip connection by inlining the child module.
    if child_split[: len(parent_split)] != parent_split:
        raise RuntimeError(
            f"Child module '{child_fqn}' is not a descendant of parent module '{parent_fqn}'."
            "This is currently unsupported."
            "Please try to make child module attach to parent module directly."
        )
    return ".".join(child_split[len(parent_split) :])


def _check_graph_equivalence(x: torch.nn.Module, y: torch.nn.Module):
    def graph_dump(graph: torch.fx.Graph) -> str:
        ret = []
        nodes_idx: Dict[int, int] = {}

        def arg_dump(arg) -> str:
            if isinstance(arg, torch.fx.Node):
                return "%" + str(nodes_idx[id(arg)])
            return str(arg)

        for i, node in enumerate(graph.nodes):
            args_dump = [str(arg) for arg in pytree.tree_map(arg_dump, node.args)]
            args_dump += [
                f"{key}={value}"
                for key, value in pytree.tree_map(arg_dump, node.kwargs).items()
            ]
            target = node.target if node.op in ("call_function", "get_attr") else ""
            ret.append(f"{i}: {node.op}[{target}]({', '.join(args_dump)})")
            nodes_idx[id(node)] = i
        return "\n".join(ret)

    assert isinstance(x.graph, torch.fx.Graph)
    assert isinstance(y.graph, torch.fx.Graph)
    return graph_dump(x.graph) == graph_dump(y.graph)


def _add_spec(gm: torch.nn.Module, spec) -> str:
    i = 0
    while hasattr(gm, f"_spec_{i}"):
        i += 1
    name = f"_spec_{i}"
    setattr(gm, name, spec)
    return name


def _generate_flatten(gm: torch.fx.GraphModule, node) -> torch.fx.Node:
    flatten = gm.graph.call_function(pytree.tree_flatten, (node,))
    getitem_0 = gm.graph.call_function(operator.getitem, (flatten, 0))
    return getitem_0


def _generate_flatten_spec(
    gm: Union[torch.fx.GraphModule, InterpreterModule, UnflattenedModule], node, spec
) -> torch.fx.Node:
    name = _add_spec(gm, spec)
    spec_node = gm.graph.get_attr(name)
    return gm.graph.call_function(fx_pytree.tree_flatten_spec, (node, spec_node))


def _generate_unflatten(
    gm: Union[torch.fx.GraphModule, InterpreterModule, UnflattenedModule], nodes, spec
) -> torch.fx.Node:
    name = _add_spec(gm, spec)
    spec_node = gm.graph.get_attr(name)
    return gm.graph.call_function(pytree.tree_unflatten, (nodes, spec_node))


def _get_submodule(mod: torch.nn.Module, target: str):
    *prefix, field = target.split(".")

    for item in prefix:
        submod = getattr(mod, item, None)

        if submod is None:
            return None

        if not isinstance(submod, torch.nn.Module):
            return None

        mod = submod

    return getattr(mod, field, None)


def _add_submodule(
    mod: torch.nn.Module,
    target: str,
    module_to_add: torch.nn.Module,
    create_module: Optional[Callable[[str], torch.nn.Module]] = None,
):
    *prefix, field = target.split(".")

    for i, item in enumerate(prefix):
        submod = getattr(mod, item, None)

        if submod is None:
            if create_module is not None:
                submod = create_module(".".join(prefix[: i + 1]))
            else:
                submod = torch.nn.Module()
            setattr(mod, item, submod)

        if not isinstance(submod, torch.nn.Module):
            return False

        mod = submod

    mod.add_module(field, module_to_add)


def _call_name(base: str, n: int) -> str:
    # Given n >= 0, generate call names to a submodule `base` of the form
    # `base`, `base@1`, `base@2`, etc.
    return base if n == 1 else f"{base}@{n - 1}"


def _is_call_name(call_name: str, base: str) -> bool:
    # Recognize when call_name = _call_name(base, n) for some n >= 0.
    return re.match(re.escape(base) + r"(@\d+)?$", call_name) is not None


class _ModuleFrame:
    def __init__(
        self,
        flat_graph: torch.fx.Graph,
        nodes: Tuple[torch.fx.Node, ...],
        seen_nodes,
        seen_modules,
        seen_attrs,
        created_modules,
        parent,
        module_stack: List[Tuple[str, int]],
        module_id,
        module_call_graph: Dict[str, ModuleCallSignature],
        module: Optional[Union[torch.fx.GraphModule, UnflattenedModule]] = None,
    ):
        self.flat_graph = flat_graph
        self.nodes = nodes
        self.seen_nodes = seen_nodes
        self.seen_modules = seen_modules
        self.seen_attrs = seen_attrs
        self.created_modules = created_modules
        self.parent = parent
        self.module_stack = module_stack
        self.module_id = module_id

        self.module_call_graph = module_call_graph
        self.verbose = False

        self.fqn, num_calls = self.module_stack[-1]
        # generate call name for self.fqn
        self.child_fqn = _call_name(self.fqn, num_calls + 1)

        self.module: Union[torch.fx.GraphModule, UnflattenedModule, InterpreterModule]
        if module is not None:
            self.module = module
            self.ivals = module.ivals if hasattr(module, "ivals") else {}  # type: ignore[var-annotated]
        else:
            self.module = self.created_modules.get(
                self.fqn,
                InterpreterModule(torch.fx.Graph()),
            )
            self.ivals = parent.ivals

        self.graph = self.module.graph

        # Mapping of nodes in the flat graph to nodes in this graph.
        self.node_map: Dict[torch.fx.Node, torch.fx.Node] = {}
        self.node_to_placeholder = {}

        self.parent_call_module: Optional[torch.fx.Node] = None
        if parent is not None:
            accessor = _compute_accessor(parent.fqn, self.child_fqn)

            def create_module(fqn):
                path = f"{parent.fqn}.{fqn}" if parent.fqn else fqn
                if path in self.created_modules:
                    return self.created_modules[path]
                submod = InterpreterModule(torch.fx.Graph())
                self.created_modules[path] = submod
                return submod

            _add_submodule(parent.module, accessor, self.module, create_module)
            self.parent_call_module = parent.graph.call_module(accessor)
            if self.seen_modules[self.module_id]:
                base_module_frame = self.seen_modules[self.module_id][0]
                self.module._modules = base_module_frame.module._modules
            self.seen_modules[self.module_id].append(
                _SubmoduleEntry(
                    parent_fqn=self.parent.fqn,
                    parent_module=self.parent.module,
                    parent_call_module=self.parent_call_module,
                    fqn=self.fqn,
                    call_idx=num_calls + 1,
                    module=self.module,
                )
            )

        signature = module_call_graph.get(self.child_fqn)
        if signature is not None and self.parent is not None:
            assert signature.in_spec.num_children == 2
            assert signature.in_spec.type is tuple
            args_spec, kwargs_spec = signature.in_spec.children()
            assert args_spec.type is tuple
            assert kwargs_spec.type is dict

            with self.graph.inserting_after(None):
                arg_nodes = [
                    self.graph.placeholder(f"_positional_arg_{idx}")
                    for idx in range(args_spec.num_children)
                ]
                kwarg_nodes = {}
                for name in kwargs_spec.context:
                    kwarg_nodes[name] = self.graph.placeholder(name)
                flat_args = _generate_flatten_spec(
                    self.module,
                    (tuple(arg_nodes), kwarg_nodes),
                    signature.in_spec,
                )
                for idx, arg in enumerate(signature.inputs):
                    flat_arg_node = self.graph.create_node(
                        op="call_function",
                        target=operator.getitem,
                        args=(flat_args, idx),
                        name=(
                            arg.name
                            if not isinstance(arg, ConstantArgument)
                            else f"_constant_{idx}"
                        ),
                    )
                    if isinstance(arg, ConstantArgument):
                        continue

                    if arg.name in self.seen_nodes:
                        flat_arg_node.meta = copy.copy(self.seen_nodes[arg.name].meta)
                        self.node_to_placeholder[
                            self.seen_nodes[arg.name]
                        ] = flat_arg_node

            with self.parent.graph.inserting_before(self.parent_call_module):
                input_nodes: List[Optional[torch.fx.Node]] = []
                for input in signature.inputs:
                    if isinstance(input, ConstantArgument):
                        input_nodes.append(input.value)  # type: ignore[arg-type]
                    elif input.name not in self.seen_nodes:
                        input_nodes.append(None)
                    else:
                        assert isinstance(
                            input,
                            (
                                TensorArgument,
                                SymIntArgument,
                                SymBoolArgument,
                                SymFloatArgument,
                            ),
                        )
                        input_nodes.append(
                            self.parent.remap_input(self.seen_nodes[input.name])
                        )

                inputs_node = _generate_unflatten(
                    self.parent.module,
                    input_nodes,
                    signature.in_spec,
                )

                args_node = self.parent.graph.call_function(
                    operator.getitem, (inputs_node, 0)
                )
                kwargs_node = self.parent.graph.call_function(
                    operator.getitem, (inputs_node, 1)
                )
                arg_nodes = [
                    self.parent.graph.call_function(operator.getitem, (args_node, i))
                    for i in range(args_spec.num_children)
                ]
                kwarg_nodes = {
                    k: self.parent.graph.call_function(
                        operator.getitem, (kwargs_node, k)
                    )
                    for k in kwargs_spec.context
                }
            assert self.parent_call_module is not None
            self.parent_call_module.args = tuple(arg_nodes)
            self.parent_call_module.kwargs = kwarg_nodes  # type: ignore[assignment]

    def add_placeholder(self, x):
        assert self.fqn != "", f"Cannot add placeholder {x} to root module"
        assert x.graph is self.flat_graph
        # x is not in subgraph, create a new placeholder for subgraph
        with self.graph.inserting_before(None):
            placeholder_node = self.graph.placeholder(x.name, type_expr=x.type)
        # copy all meta fields, even if some fields might be irrelevant for
        # the placeholder node
        placeholder_node.meta = copy.copy(x.meta)
        self.node_to_placeholder[x] = placeholder_node

    def copy_sym_call_function(self, x):
        # This only exists because we deduplicate sym_size nodes in the flat export graph,
        # and if preserve_module_call_signature is set, we may not be able to pass sym_size
        # nodes, or their downstream users, as inputs to submodule calls.
        # To avoid this we copy these call_function nodes with sym_type results.
        # This should however only be done for sym_type nodes - call_function nodes on tensors
        # should not be deduplicated in the first place.
        args = pytree.tree_map_only(torch.fx.Node, self.remap_input, x.args)
        kwargs = pytree.tree_map_only(torch.fx.Node, self.remap_input, x.kwargs)
        node = self.graph.call_function(x.target, args, kwargs)
        node.meta = copy.copy(x.meta)
        self.node_map[x] = node
        return node

    def remap_input(self, x):
        assert x.graph is self.flat_graph
        if x in self.node_map:
            return self.node_map[x]
        self.print(f"remap_input({x})")
        if x in self.node_to_placeholder:
            return self.node_to_placeholder[x]
        elif (
            x.op == "placeholder"
            or self.module_call_graph.get(self.fqn) is None
            # allow placeholder creation if we are not preserving module call signature
        ):
            self.add_placeholder(x)
            if self.parent_call_module is not None:
                # Important to *prepend* the output to match how we are
                # inserting placeholder nodes.
                with self.parent.graph.inserting_before(self.parent_call_module):
                    self.parent_call_module.insert_arg(0, self.parent.remap_input(x))
            return self.node_to_placeholder[x]
        elif x.op == "call_function" and (
            x.target
            in (
                torch.ops.aten.sym_size.int,
                torch.ops.aten.item.default,
                torch.ops.aten.unbind.int,
                torch.ops.aten.sum.dim_IntList,
                torch.ops.aten.view.default,
                torch.ops.aten.diff.default,
            )
            or (hasattr(x.target, "__module__") and x.target.__module__ == "_operator")
        ):
            # export deduplicates sym_size nodes, and may need to re-copy them
            # if module call signature needs to be preserved
            self.copy_sym_call_function(x)
            return self.node_map[x]
        elif self.module_call_graph.get(self.fqn) is not None:
            # x is an ival that is not in placeholders, so create a
            # get_attr node corresponding to attribute __ival__x
            return self.ivals.read(self.fqn, self.graph, x)  # type: ignore[operator, union-attr]
        else:
            raise RuntimeError(
                f"Could not run remap_input() on op type: {x.op} for node {x}"
            )

    def finalize_outputs(self):
        self.created_modules.pop(self.fqn, None)

        orig_outputs = []

        signature = self.module_call_graph.get(self.child_fqn)
        if signature is not None and self.parent is not None:
            for output in signature.outputs:
                if isinstance(
                    output,
                    (TensorArgument, SymIntArgument, SymBoolArgument, SymFloatArgument),
                ):
                    if output.name in self.seen_nodes:
                        orig_outputs.append(self.seen_nodes[output.name])
                    else:
                        orig_outputs.append(None)
                else:
                    raise RuntimeError(
                        f"Unsupported data type for output node: {output}"
                    )

            def get_actual_output_node(output):
                if output is None:
                    return None

                seen_node = self.seen_nodes[output.name]
                if seen_node in self.node_map:
                    return self.node_map[seen_node]
                elif seen_node in self.node_to_placeholder:
                    return self.node_to_placeholder[seen_node]
                else:
                    raise RuntimeError(
                        f"Could not find output node {output}. Graph: {self.graph}"
                    )

            tree_out_node = _generate_unflatten(
                self.module,
                tuple(get_actual_output_node(output) for output in orig_outputs),
                signature.out_spec,
            )
            parent_out: Optional[torch.fx.Node] = _generate_flatten_spec(
                self.parent.module, self.parent_call_module, signature.out_spec
            )
            graph_outputs: Union[torch.fx.Node, List[torch.fx.Node]] = tree_out_node
        else:
            graph_outputs = []
            # Iterate through nodes we have copied into self.graph.
            for orig_node in self.node_map.keys():
                for user_node in orig_node.users:
                    if user_node.name not in self.seen_nodes:
                        # external user node, need to expose as an output
                        orig_outputs.append(orig_node)
                        graph_outputs.append(self.node_map[orig_node])
                        break

            parent_out = self.parent_call_module
            if len(graph_outputs) == 1:
                graph_outputs = graph_outputs[0]

        assert isinstance(graph_outputs, (list, torch.fx.Node))

        self.graph.output(graph_outputs)

        # Rewrite outputs in parent module
        if parent_out is None:
            return

        parent_out.meta["val"] = (
            graph_outputs.meta.get("val")
            if isinstance(graph_outputs, torch.fx.Node)
            else [o.meta.get("val") for o in graph_outputs]
        )

        if len(orig_outputs) == 1 and signature is None:
            self.parent.node_map[orig_outputs[0]] = parent_out
        else:
            for i, orig_output in enumerate(orig_outputs):
                if orig_output is None:
                    continue
                # Use Proxy to record getitem access.
                proxy_out = torch.fx.Proxy(parent_out)[i].node  # type: ignore[index]
                proxy_out.meta["val"] = orig_output.meta.get("val")
                self.parent.node_map[orig_output] = proxy_out

    def copy_node(self, node):
        self.print("copying", node.format_node())
        self.node_map[node] = self.graph.node_copy(node, self.remap_input)
        self.seen_nodes[node.name] = node

    def run_outer(self):
        i = 0
        for node in self.flat_graph.nodes:
            self.print(i, node.meta.get("nn_module_stack"), node.format_node())
            i += 1

        # Copy all graph inputs
        node_idx: int = 0
        node = self.nodes[node_idx]
        while node.op == "placeholder":
            self.copy_node(node)
            node_idx += 1
            node = self.nodes[node_idx]

        self.run_from(node_idx)

        # Copy graph outputs
        for node in self.flat_graph.nodes:
            if node.op == "output":
                self.copy_node(node)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def run_from(self, node_idx):
        module_idx = 0
        # Walk through the graph, building up a new graph with the right submodules
        while node_idx < len(self.nodes):
            node = self.nodes[node_idx]
            assert node.op != "placeholder"

            self.print()
            self.print("STEP", node_idx, node.format_node())
            self.print(self.module_stack)
            depth = len(self.module_stack)
            if node.op == "output":
                if depth == 1:
                    # We want the output node of the original graph to be handled
                    # specially by the outermost stack frame (in run_outer). So
                    # skip finalization here.
                    return node_idx

                # We've reached the end of the graph. Wrap up all the existing stack frames.
                self.finalize_outputs()
                return node_idx

            if len(node.meta.get("nn_module_stack", {})) == 0:
                raise RuntimeError(f"Unable to find nn_module_stack for node {node}")

            nn_module_stack = node.meta["nn_module_stack"]
            from torch._export.passes._node_metadata_hook import (
                _EMPTY_NN_MODULE_STACK_KEY,
            )

            if (
                len(nn_module_stack) == 1
                and _EMPTY_NN_MODULE_STACK_KEY in nn_module_stack
            ):
                # Empty case from the node_metadata_hook
                node_module_stack = self.module_stack
            else:
                node_module_stack = [
                    (path, int(k.split("@")[-1]) if "@" in k else 0)
                    for k, (path, ty) in node.meta["nn_module_stack"].items()
                ]

            if node_module_stack[:depth] != self.module_stack:
                # This means that the current module is done executing and the
                # current node is the beginning of a new module.
                #
                # In this case, we should finalize this module and return without
                # incrementing the node counter.
                self.finalize_outputs()
                self.print("outlining", self.fqn)
                self.print(self.graph)
                return node_idx

            assert node_module_stack is not None

            if _is_prefix(self.module_stack, node_module_stack):
                # This means that the current node represents the execution of a new
                # module.
                next_module = node_module_stack[depth]
                self.print("Creating new stack frame for", next_module)
                # Run a nested version of module outliner from the current node
                # counter. Once it is complete, continue from that point.
                next_module_key = list(node.meta["nn_module_stack"].keys())[depth]
                node_idx = _ModuleFrame(
                    self.flat_graph,
                    self.nodes,
                    self.seen_nodes,
                    self.seen_modules,
                    self.seen_attrs,
                    self.created_modules,
                    self,
                    self.module_stack + [next_module],
                    next_module_key.split("@")[0],
                    self.module_call_graph,
                ).run_from(node_idx)
                module_idx += 1
                continue

            # The only remaining possibility is that we are in the right stack
            # frame. Copy the node into this frame's graph and increment the node counter.
            assert node_module_stack == self.module_stack

            if node.op == "get_attr":
                # this must be a graph argument for a HOP
                self.seen_attrs[self.child_fqn].add(node.target)

            self.copy_node(node)
            node_idx += 1


@dataclass
class _SubmoduleEntry:
    parent_fqn: str
    parent_module: torch.nn.Module
    parent_call_module: torch.fx.Node
    fqn: str
    call_idx: int
    module: torch.nn.Module


def _outline_submodules(orig_graph: torch.fx.Graph, root_module: UnflattenedModule):
    seen_nodes: Dict[str, torch.fx.Node] = {}
    seen_modules: Dict[int, List[_SubmoduleEntry]] = defaultdict(list)
    seen_attrs: Dict[str, Set[str]] = defaultdict(set)
    created_modules: Dict[str, torch.nn.Module] = {}
    _ModuleFrame(
        orig_graph,
        tuple(orig_graph.nodes),
        seen_nodes,
        seen_modules,
        seen_attrs,
        created_modules,
        None,
        [("", 0)],
        "",
        {
            entry.fqn: entry.signature
            for entry in root_module.module_call_graph
            if entry.signature
        },
        module=root_module,
    ).run_outer()
    return seen_modules, seen_attrs


def _reorder_submodules(
    parent: torch.nn.Module, fqn_order: Dict[str, int], prefix: str = ""
):
    # TODO Can be optimized by adding submodules ahead of time.
    if prefix == "":
        for fqn in list(fqn_order.keys())[1:]:
            if _get_submodule(parent, fqn) is None:
                _add_submodule(parent, fqn, torch.nn.Module())

    children = []
    for name, child in list(parent._modules.items()):
        if child is None:
            continue
        fqn = prefix + name
        _reorder_submodules(child, fqn_order, prefix=fqn + ".")
        delattr(parent, name)
        children.append((fqn_order[fqn], name, child))
    children.sort(key=operator.itemgetter(0))
    for _, name, child in children:
        parent.register_module(name, child)


class _IVals:
    """
    Collect the intermediate values of buffer mutations in a graph,
    along with the module call fqns that create and use them. Later,
    in each fqn associated with an intermediate value we will install
    a corresponding attribute, so that it can be updated and read.

    Example: in the following graph, suppose that buf_in and buf_out
    are the input and output values of a buffer.

        buf_in = placeholder()
        ...
        ival1 = f0(buf_in, ...)  # inside self.n0(...)
        ...
        ival2 = f1(ival1, ...)  # inside self.n1(...)
        ...
        buf_out = f2(ival2, ...)  # inside self.n2(...)
        return buf_out, ...

    Here ival1 and ival2 are intermediate values created inside
    calls to n0 and n1 respectively, and used inside calls to
    n1 and n2 respectively.

    Thus our analysis will produce {ival1: {n0, n1}, ival2: {n1, n2}}.
    """

    def __init__(self):
        # ival node name -> set of fqns that create and use it
        self.fqns = defaultdict(set)
        # ival node name -> tensor storage for corresponding attribute
        self.storage = {}

    def read(self, fqn, graph, node):
        """
        Read attribute corresponding to a given intermediate value.
        """
        # to read ival x, get attribute __ival__x
        with graph.inserting_before(None):
            ival_node = graph.get_attr("__ival__" + node.name, type_expr=node.type)
            ival_node.meta = copy.copy(node.meta)

        if node.name not in self.storage:
            # create empty tensor matching fake, using a cache
            # to ensure the same tensor is returned per ival_name
            fake = node.meta["val"]
            self.storage[node.name] = torch.empty(fake.shape, dtype=fake.dtype)
        self.fqns[node.name].add(fqn)

        return ival_node

    def update(self, fqn, graph, node):
        """
        Update attribute corresponding to a given intermediate value.
        """
        self.fqns[node.name].add(fqn)

        # to update ival x, get attribute __ival__x and copy x to __ival__x
        with graph.inserting_after(node):
            ival_node = graph.get_attr("__ival__" + node.name, type_expr=node.type)
            ival_node.meta = copy.copy(node.meta)
        with graph.inserting_after(ival_node):
            new_ival_node = graph.create_node(
                "call_function", torch.ops.aten.copy_, (ival_node, node)
            )
            new_ival_node.meta = copy.copy(node.meta)

    def create(self, partitions, root_module):
        """
        Update attributes corresponding to intermediate values that were read.
        Finally, initialize attributes in all modules that read or update
        corresponding intermediate values.
        """

        entries = [("", root_module)]
        for shared_submodules in partitions:
            for entry in shared_submodules:
                entries.append((entry.fqn, entry.module))
                graph = entry.module.graph
                for node in graph.nodes:
                    if node.name in self.storage:
                        self.update(entry.fqn, graph, node)

        # fqn -> list of ival node names read or updated through it
        ivals = defaultdict(list)
        for name, fqns in self.fqns.items():
            for fqn in fqns:
                ivals[fqn].append(name)

        for fqn, mod in entries:
            for name in ivals[fqn]:
                ival_name = f"__ival__{name}"
                # for a ival named x created in module call m,
                # create attribute m.__ival__x, initially empty
                setattr(mod, ival_name, self.storage[name])


def _copy_graph_attrs(
    gm: torch.fx.GraphModule,
    root_module: UnflattenedModule,
    seen_attrs: Dict[str, Set[str]],
):
    for child_fqn, names in seen_attrs.items():
        module = _get_attr(root_module, child_fqn) if child_fqn else root_module
        for name in names:
            val = getattr(gm, name)
            setattr(module, name, val)


def _deduplicate_modules(partitions):
    redirected_call_indices = {}
    for shared_submodules in partitions:
        for i, entry in enumerate(shared_submodules):
            child_fqn = _call_name(entry.fqn, entry.call_idx)
            target = _compute_accessor(entry.parent_fqn, child_fqn)
            deduplicated = False
            # Iterate over all previously seen modules, and deduplicate if possible
            for seen in shared_submodules[:i]:
                if _check_graph_equivalence(seen.module, entry.module):
                    parent = entry.parent_module
                    # Since graphs are equivalent, we can deduplicate.
                    # There are two cases.
                    if seen.fqn == entry.fqn:
                        # Case 1: The current module has the same fqn as the seen module.
                        # In this case we have generated a call name that can be optimized away.
                        # So we remove the current module from the hierarchy and replace
                        # the current call name with the seen call name in the parent graph.
                        *prefix, name = target.split(".")
                        _get_attr_via_attr_list(parent, prefix)._modules.pop(name)
                        seen_child_fqn = _call_name(seen.fqn, seen.call_idx)
                        seen_target = _compute_accessor(
                            entry.parent_fqn, seen_child_fqn
                        )
                        entry.parent_call_module.target = seen_target
                        redirected_call_indices[child_fqn] = seen_child_fqn
                        break
                    elif not deduplicated:
                        # Case 2: The current module has a different fqn than the seen module.
                        # In this case we replace the current module with the seen module.
                        # There should be nothing pointing to the current module any more,
                        # so it can be garbage collected.
                        # NOTE: We *do not* replace the current call name with the seen call name
                        # in the parent graph, because this will lose information on which fqn
                        # was actually called. However, it is possible that the current call name
                        # will be optimized away when we find another seen module with the same fqn,
                        # so we do not break out of the loop yet.
                        parent.set_submodule(target, seen.module)
                        deduplicated = True

    return redirected_call_indices


def _sink_params(
    module: torch.nn.Module,
    inputs_to_state: Dict[str, List[str]],
    scope: List[str],
    module_id_to_inputs_removed: Optional[Dict[int, Set[str]]] = None,
):
    """Sink params, buffers, and constants from graph inputs into get_attr nodes.

    Exported modules are purely functional, so they pass their parameters and
    buffers in as inputs to the graph.

    To replicate eager's semantics, we need to get them from the module state
    via get_attr instead.

    module: GraphModule, potentially containing nested submodules.
    inputs_to_state: mapping graph input names to the corresponding key in the state_dict.
    scope: tracks where we are in the module hierarchy, so that we can emit the
        right `getattr(self, "foo.bar")` calls, etc.
    module_id_to_inputs_removed: records inputs removed by child modules, mapping
        the module object id to the list of placeholder node names in the child module
        that were removed.
    """
    if module_id_to_inputs_removed is None:
        module_id_to_inputs_removed = defaultdict(set)

    if id(module) in module_id_to_inputs_removed:
        return {id(module): module_id_to_inputs_removed[id(module)]}

    # We need to use _modules here instead of named_children(), because we
    # explicitly want duplicate modules to show up in the traversal.
    for name, submodule in module._modules.items():
        submod_id_to_inputs_removed = _sink_params(
            cast(torch.nn.Module, submodule),
            inputs_to_state,
            scope + [name],
            module_id_to_inputs_removed,
        )
        for k, v in submod_id_to_inputs_removed.items():
            module_id_to_inputs_removed[k].update(v)

    graph = getattr(module, "graph", None)
    if graph is None or len(graph.nodes) == 0:
        # Not all modules have graphs defined, if they are empty modules with no operations (like ParameterList)
        return module_id_to_inputs_removed

    assert isinstance(graph, torch.fx.Graph)

    inputs = list(filter(lambda n: n.op == "placeholder", graph.nodes))
    the_last_input = inputs[-1]

    # Also remove from call_module nodes
    call_module_nodes = filter(lambda n: n.op == "call_module", graph.nodes)
    for node in call_module_nodes:
        submodule = _get_attr(module, node.target)
        # remove placeholder from call_module node arguments, only if we've
        # erased the placeholder node in the corresponding _sink_params() call
        if submodule is not None and id(submodule) in module_id_to_inputs_removed:
            node.args = tuple(
                filter(
                    lambda n: n.name not in module_id_to_inputs_removed[id(submodule)],
                    node.args,
                )
            )

    # Filter out inputs_to_state corresponding to current scope.
    inputs_to_state_of_scope: Dict[torch.fx.Node, list[str]] = {}
    for node in inputs:
        if node.name not in inputs_to_state:
            continue

        state_name = None
        for sn in inputs_to_state[node.name]:
            sn_split = sn.split(".")
            if sn_split[: len(scope)] == [x.split("@")[0] for x in scope]:
                state_name = sn_split
                break

        # If there's a mismatch between scope name and state name, then
        # there must be multiple scopes pointing to the same state name,
        # meaning some modules are shared. In such case, we can simply skip
        # updating the current node because another later iteration will
        # take care of this input node when the unique match between scope
        # and state name occurs.  To make sure this always happen, we should
        # enforce the invariant that no placeholder node in the unflattened
        # graph appears in inputs_to_state dict, which means all the extra
        # input nodes have been handled.
        if state_name is None:
            continue

        inputs_to_state_of_scope[node] = state_name

    # Record name of remove inputs for return purpose.
    inputs_removed: Set[str] = set()

    for node, state_name in inputs_to_state_of_scope.items():
        if len(node.users) > 0:
            attr_path = state_name[len(scope) :]
            state_attr = _get_attr_via_attr_list(module, attr_path)
            assert isinstance(state_attr, (torch.Tensor, torch.ScriptObject))

            # Make sure the newly created get_attr node is placed after the last placeholder node
            with graph.inserting_after(the_last_input):
                new_node = graph.create_node("get_attr", ".".join(attr_path))

            node.replace_all_uses_with(new_node, propagate_meta=True)

        graph.erase_node(node)
        inputs_removed.add(node.name)

    if isinstance(module, InterpreterModule):
        module.finalize()

    return {id(module): inputs_removed}
