"""
This module implements variable tracking for PyTorch optimizers during Dynamo tracing.

The OptimizerVariable class provides specialized handling for optimizer instances by:
- Optimizing the tracing of expensive optimizer initialization
- Managing optimizer state and parameter group tracking
- Handling tensor sources and guards for optimizer state tensors
- Supporting CUDA graph execution through static tensor address management
- Providing special handling for parameter gradients and optimizer state tensors

Key features include:
- Efficient initialization tracing via _init_group optimization
- Automatic marking of optimizer state tensors as static for CUDA graphs
- Proper source tracking for parameter groups, gradients, and state tensors
- Guard installation for optimizer state structure
- Support for both CPU and GPU tensor handling
- Cleanup of static tensor references via finalizers

The module integrates with Dynamo's broader tracing system while providing
optimizer-specific optimizations and safety guarantees.
"""

import logging
import weakref
from collections.abc import Iterable
from typing import Any, Optional, TYPE_CHECKING

import torch
from torch._dynamo.variables.tensor import TensorVariable
from torch._guards import Source
from torch._logging import getArtifactLogger
from torch.utils._pytree import tree_map_only

from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    ConstDictKeySource,
    DictGetItemSource,
    GetItemSource,
    GlobalWeakRefSource,
    GradSource,
)
from ..utils import GLOBAL_KEY_PREFIX
from .base import VariableTracker
from .constant import ConstantVariable
from .dicts import ConstDictVariable
from .lists import ListVariable
from .misc import GetAttrVariable
from .user_defined import UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


class ArgMappingException(Exception):
    pass


class GuardInstallException(Exception):
    pass


perf_hint_log = getArtifactLogger(__name__, "perf_hints")


def _is_static_for_cudagraphs(x: torch.Tensor) -> bool:
    from torch._inductor.cudagraph_trees import get_manager

    if x.is_cuda:
        manager = get_manager(x.device.index, False)
        is_static_address = torch._dynamo.utils.get_static_address_type(x) is not None
        if manager:
            assert manager.current_node is not None
            return (
                is_static_address
                or manager.current_node._is_cuda_graph_recorded_tensor(x)
            )
        else:
            return is_static_address
    else:
        # Don't print a warning for non-cuda tensors
        return True


class OptimizerVariable(UserDefinedObjectVariable):
    _nonvar_fields = {
        "grad_to_source",
        "tensor_to_source",
        "static_tensor_names",
        *UserDefinedObjectVariable._nonvar_fields,
    }

    def __init__(
        self,
        value: torch.optim.Optimizer,
        grad_to_source: Optional[dict[Any, GradSource]] = None,
        static_tensor_names: Optional[set[str]] = None,
        tensor_to_source: Optional[dict[torch.Tensor, Source]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(value, **kwargs)
        # pyrefly: ignore [bad-override]
        self.value: torch.optim.Optimizer = value
        self.grad_to_source = grad_to_source or {}
        self.tensor_to_source = tensor_to_source or {}
        self.static_tensor_names = static_tensor_names or set()

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker":
        """This is an optimization to avoid tracing the very slow initialization of the optimizer"""
        if name == "_init_group":
            if not hasattr(self.value, "_init_group"):
                # Fallback: if the optimizer does not have _init_group, trace normally
                return super().call_method(tx, name, args, kwargs)
            try:
                self.graph_break_if_pending_mutation(tx)
                self.move_step_if_cpu()
                py_args, py_kwargs = self.get_python_args(*args, **kwargs)
                ret_val = self.value._init_group(*py_args, **py_kwargs)
                self.map_sources_and_install_guards(tx)
                self.update_list_args(tx, args, kwargs, py_args, py_kwargs)
                # stash a weak_ptr to optimizer to invalidate code
                # if the optimizer object dies
                mangled_name = f"__optimizer_{id(self.value)}"
                tx.store_global_weakref_by_id(mangled_name, self.value)
                self.create_finalizer(tx)

                # This is currently safe only because the only actual `ret_val`s returned
                # by the `_init_group` of existing optimizers are properties that are invariant
                # to the input tensors (e.g. dtype, layout). Changing these would trigger a
                # recompilation and hence never result in the wrong specialization of `ret_val`.
                return ConstantVariable.create(ret_val)
            except (ArgMappingException, GuardInstallException) as _:
                # trace normally if we can't map args or install guards correctly
                pass

        return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        # Note: this allows us to intercept the call in call_method
        # in the typical case, we return a UserMethodVariable
        # which will directly inline
        if name in ("_init_group"):
            assert self.source
            return GetAttrVariable(self, name, source=AttrSource(self.source, name))

        if name == "param_groups":
            from ..decorators import mark_static_address

            for group in self.value.param_groups:
                for p in group["params"]:
                    mark_static_address(p, guard=True)

            self._set_capturable(tx)

        return super().var_getattr(tx, name)

    def graph_break_if_pending_mutation(self, tx: "InstructionTranslator") -> None:
        # If there are pending mutations on a parameter (due to using closure)
        # then we need to graph break to allow the python version of the parameter
        # to update, so that running _init_group will initialize the states with
        # the correct values
        for g in self.value.param_groups:
            for p in g["params"]:
                side_effects = tx.output.side_effects
                variable = side_effects.id_to_variable.get(id(p), None)
                if variable and side_effects.has_pending_mutation(variable):
                    from ..exc import unimplemented

                    unimplemented(
                        gb_type="optimizer: pending mutation on parameter",
                        context=f"variable: {variable}, parameter: {p}",
                        explanation="Pending mutations on a parameter (e.g. due to using closure) require a graph break.",
                        hints=[],
                    )

    def _set_capturable(self, tx: "InstructionTranslator") -> None:
        from . import LazyVariableTracker

        # We only set capturable if params are on cuda
        # and the state is not initialized
        def safe_to_set_capturable(group: dict[str, Any]) -> bool:
            all_uninitialized = True
            all_gpu = True

            for p in group.get("params", []):
                all_gpu &= p.is_cuda or p.is_xpu
                all_uninitialized &= p not in self.value.state

            return "capturable" in group and all_uninitialized and all_gpu

        # track indices to not set so we don't need to
        # in the variable tracker realize the whole state
        # we handle guarding the state specially
        for group in self.value.param_groups:
            if safe_to_set_capturable(group):
                group["capturable"] = True

        source = self.source and AttrSource(self.source, "param_groups")
        param_groups_vt = LazyVariableTracker.realize_all(
            VariableTracker.build(tx, self.value.param_groups, source)
        )
        for param_group_vt in param_groups_vt.items:
            key = ConstDictVariable._HashableTracker(
                ConstantVariable.create("capturable")
            )
            param_group_vt.items[key] = ConstantVariable.create(True)

    def get_python_args(
        self, *args: Any, **kwargs: Any
    ) -> tuple[list[Any], dict[str, Any]]:
        """Get python values equivalent to the variable tracker args"""

        def map_arg(arg: Any) -> Any:
            if isinstance(arg, VariableTracker) and arg.is_python_constant():
                return arg.as_python_constant()
            elif isinstance(arg, ListVariable) and not arg.items:
                return []
            elif (
                isinstance(arg, ConstDictVariable)
                and isinstance(arg.source, GetItemSource)
                and isinstance(arg.source.base, AttrSource)
                and arg.source.base.member == "param_groups"
            ):
                return self.value.param_groups[arg.source.index]

            raise ArgMappingException

        new_args = [map_arg(arg) for arg in args]
        new_kwargs = {k: map_arg(v) for k, v in kwargs.items()}

        return new_args, new_kwargs

    # If users load an old state dictionary,
    # it's possible that step could be on the cpu
    # if this is the case, move it to the GPU
    # corresponding to the parameter
    # in most cases this is a no-op because the state is empty
    def move_step_if_cpu(self) -> None:
        for p, state in self.value.state.items():
            if "step" in state and state["step"].is_cpu:
                state["step"] = state["step"].to(p.device)

    def map_sources_and_install_guards(self, tx: "InstructionTranslator") -> None:
        from ..decorators import mark_static_address
        from .lazy import LazyVariableTracker

        self.grad_to_source = {}
        self.tensor_to_source = {}

        def mark_static(x: Any) -> None:
            mark_static_address(x, guard=True)

        tree_map_only(torch.Tensor, mark_static, self.value.state)

        # Recursively realize the variable trackers for optim.state and
        # optim.param_groups, which recursively install the necessary guards.
        params_groups_source = self.source and AttrSource(self.source, "param_groups")
        param_groups_vt = LazyVariableTracker.realize_all(
            VariableTracker.build(tx, self.value.param_groups, params_groups_source)
        )

        state_source = self.source and AttrSource(self.source, "state")
        state_vt = VariableTracker.build(tx, self.value.state, state_source)

        # We need to realize the top level state dict to populate
        # the guard locals
        state_vt.realize()
        assert state_source is not None
        tx.output.guard_on_key_order.add(state_source)

        # Populate self.grad_to_source and self.tensor_to_source so that we can
        # manually update_list_args
        for group, group_vt in zip(self.value.param_groups, param_groups_vt.items):
            # we assume here that all params within a param group
            # are initialized similarly
            if len(group["params"]) > 0:
                for param in group["params"]:
                    if param.grad is not None:
                        key_index = None
                        for i, k in enumerate(self.value.state.keys()):
                            if k is param:
                                key_index = i
                                break
                        if key_index:
                            LazyVariableTracker.realize_all(
                                VariableTracker.build(
                                    tx,
                                    self.value.state[param],
                                    DictGetItemSource(
                                        state_source,
                                        ConstDictKeySource(state_source, key_index),
                                    ),
                                )
                            )
                            break

            params_vt = group_vt.getitem_const(tx, ConstantVariable.create("params"))
            all_static = True
            non_static_grads = []
            for p, p_vt in zip(group["params"], params_vt.unpack_var_sequence(tx)):
                param_source = p_vt.source
                self.tensor_to_source[p] = param_source
                grad_source = GradSource(
                    param_source,
                    "grad",
                )

                if p.grad is not None:
                    self.grad_to_source[p.grad] = grad_source
                    if not _is_static_for_cudagraphs(p.grad):
                        all_static = False
                        non_static_grads.append(grad_source)
                else:
                    install_guard(grad_source.make_guard(GuardBuilder.CONSTANT_MATCH))

            # Note: to avoid spam logs only warn if perf hint artifact is enabled
            # (NB: artifacts are only enabled at the debug or warning level)
            if not all_static and perf_hint_log.isEnabledFor(logging.DEBUG):
                non_static_grad_names = [src.name for src in non_static_grads]
                perf_hint_log.warning(
                    (
                        "Grad tensors %s will be copied during cudagraphs execution."
                        "If using cudagraphs and the grad tensor addresses will be the same across runs,"
                        " use torch._dynamo.decorators.mark_static_address to elide this copy.",
                    ),
                    non_static_grad_names,
                )

        # We have to again iterate over the state dict to collect the
        # tensor_to_source dict. This is used for the finalizer.
        for idx, value in enumerate(self.value.state.values()):
            p_state_source = DictGetItemSource(
                state_source, ConstDictKeySource(state_source, idx)
            )
            tx.output.guard_on_key_order.add(p_state_source)
            for inner_idx, v in enumerate(value.values()):
                if (
                    isinstance(v, torch.Tensor)
                    and v not in self.grad_to_source
                    and v not in self.tensor_to_source
                ):
                    self.tensor_to_source[v] = DictGetItemSource(
                        p_state_source, ConstDictKeySource(p_state_source, inner_idx)
                    )

    def wrap_tensor(
        self, tx: "InstructionTranslator", tensor_value: torch.Tensor
    ) -> TensorVariable:
        """Wrap state tensor in a TensorVariable"""
        from ..decorators import mark_static_address

        # If we have a source for a tensor already use it,
        # if we have not seen a tensor before, stash and use a
        # global weak ref source, since it must be an optimizer tensor
        # that we have missed

        if tensor_value in self.tensor_to_source:
            # mark these tensors as static for cudagraphs
            mark_static_address(tensor_value, guard=True)
            source = self.tensor_to_source[tensor_value]
            self.static_tensor_names.add(tx.output.module_key_name(source.name))
        elif tensor_value in self.grad_to_source:
            source = self.grad_to_source[tensor_value]
        else:
            # mark these tensors as static for cudagraphs
            mark_static_address(tensor_value, guard=True)

            global_name = tx.store_global_weakref_by_id(GLOBAL_KEY_PREFIX, tensor_value)
            source = GlobalWeakRefSource(global_name)
            self.static_tensor_names.add(tx.output.module_key_name(source.name))

        return VariableTracker.build(tx, tensor_value, source)

    def update_list_args(
        self,
        tx: "InstructionTranslator",
        args: Iterable[VariableTracker],
        kwargs: Any,
        py_args: Iterable[Any],
        py_kwargs: Any,
    ) -> None:
        """Update the args and kwargs to the traced optimizer call"""
        for arg, py_arg in zip(args, py_args):
            if isinstance(arg, ListVariable):
                assert isinstance(py_arg, list), (
                    "py_arg should be a list in optimizer variable"
                )
                for i, val in enumerate(py_arg):
                    tx.output.side_effects.mutation(arg)
                    if isinstance(val, torch.Tensor):
                        arg.items.append(self.wrap_tensor(tx, val))
                    else:
                        source = arg.source and GetItemSource(arg.source, i)
                        arg.items.append(VariableTracker.build(tx, val, source))

    def create_finalizer(self, tx: "InstructionTranslator") -> None:
        names_to_delete = self.static_tensor_names
        value = self.value
        tc = tx.output.tracing_context

        def init_finalizer(gm: torch.fx.GraphModule) -> None:
            def clear_static_tensor_refs() -> None:
                for name in names_to_delete:
                    gm._buffers.pop(name, None)
                    gm._parameters.pop(name, None)
                    if tc.params_flat:
                        tc.params_flat.clear()
                    if tc.params_flat_unwrap_subclasses:
                        tc.params_flat_unwrap_subclasses.clear()

            weakref.finalize(value, clear_static_tensor_refs)

        tx.output.add_graph_finalizer(init_finalizer)
