# mypy: ignore-errors

import weakref
from typing import Dict, List, TYPE_CHECKING

import torch
from torch.utils._pytree import tree_map_only

from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    ConstDictKeySource,
    GetItemSource,
    GlobalWeakRefSource,
    GradSource,
)
from ..utils import GLOBAL_KEY_PREFIX
from .constant import ConstantVariable
from .dicts import ConstDictVariable
from .lists import ListVariable
from .misc import GetAttrVariable
from .user_defined import UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from .base import VariableTracker


class ArgMappingException(Exception):
    pass


class GuardInstallException(Exception):
    pass


class OptimizerVariable(UserDefinedObjectVariable):
    _nonvar_fields = {
        "grad_to_source",
        "tensor_to_source",
        "static_tensor_names",
        *UserDefinedObjectVariable._nonvar_fields,
    }

    def __init__(
        self,
        value,
        grad_to_source=None,
        static_tensor_names=None,
        tensor_to_source=None,
        **kwargs,
    ) -> None:
        super().__init__(value, **kwargs)
        self.grad_to_source = grad_to_source or {}
        self.tensor_to_source = tensor_to_source or {}
        self.static_tensor_names = static_tensor_names or set()

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        """This is an optimization to avoid tracing the very slow initialization of the optimizer"""
        if name == "_init_group":
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

    def var_getattr(self, tx: "InstructionTranslator", name):
        # Note: this allows us to intercept the call in call_method
        # in the typical case, we return a UserMethodVariable
        # which will directly inline
        if name in ("_init_group", "step"):
            return GetAttrVariable(self, name, source=AttrSource(self.source, name))

        if name == "param_groups":
            from ..decorators import mark_static_address

            for group in self.value.param_groups:
                for p in group["params"]:
                    mark_static_address(p)

            self._set_capturable(tx)

        return super().var_getattr(tx, name)

    def graph_break_if_pending_mutation(self, tx):
        # If there are pending mutations on a parameter (due to using closure)
        # then we need to graph break to allow the python version of the parameter
        # to update, so that running _init_group will initialize the states with
        # the correct values
        for g in self.value.param_groups:
            for p in g["params"]:
                side_effects = tx.output.side_effects
                variable = side_effects.id_to_variable.get(id(p), None)
                if variable and side_effects.has_pending_mutation(variable):
                    from ..exc import Unsupported

                    raise Unsupported("Pending mutation on parameter")

    def _set_capturable(self, tx):
        from . import LazyVariableTracker
        from .builder import VariableBuilder

        # We only set capturable if params are on cuda
        # and the state is not initialized
        def safe_to_set_capturable(group):
            all_uninitialized = True
            all_gpu = True

            for p in group.get("params", []):
                all_gpu &= p.is_cuda or p.is_xpu
                all_uninitialized &= p not in self.value.state

            return "capturable" in group and all_uninitialized and all_gpu

        # track indices to not set so we don't need to
        # in the variable tracker realize the whole state
        # we handle guarding the state specially
        for ind, group in enumerate(self.value.param_groups):
            if safe_to_set_capturable(group):
                group["capturable"] = True

        param_groups_vt = LazyVariableTracker.realize_all(
            VariableBuilder(tx, AttrSource(self.source, "param_groups"))(
                self.value.param_groups
            )
        )
        for ind, param_group_vt in enumerate(param_groups_vt.items):
            key = ConstDictVariable._HashableTracker(
                ConstantVariable.create("capturable")
            )
            param_group_vt.items[key] = ConstantVariable.create(True)

    def get_python_args(self, *args, **kwargs):
        """Get python values equivalent to the variable tracker args"""

        def map_arg(arg):
            if isinstance(arg, ConstantVariable):
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
    def move_step_if_cpu(self):
        for p, state in self.value.state.items():
            if "step" in state and state["step"].is_cpu:
                state["step"] = state["step"].to(p.device)

    def map_sources_and_install_guards(self, tx):
        from ..decorators import mark_static_address
        from .builder import VariableBuilder
        from .lazy import LazyVariableTracker

        self.grad_to_source = {}
        self.tensor_to_source = {}

        # Tracing the _init_group is expensive. But we still have to insert the
        # necessary guards for _init_group. So, we manually handle insertion of
        # guards. We also want to mark all the tensors inside the state dict to
        # be static address.

        # Mark all the tensors in the state dict to be static address. This has
        # to be done first because the variable builder relies on the static
        # address annotation.
        def mark_static(x):
            mark_static_address(x)

        tree_map_only(torch.Tensor, mark_static, self.value.state)

        # Recursively realize the variable trackers for optim.state and
        # optim.param_groups, which recursively install the necessary guards.
        param_groups_vt = LazyVariableTracker.realize_all(
            VariableBuilder(tx, AttrSource(self.source, "param_groups"))(
                self.value.param_groups
            )
        )

        state_vt = VariableBuilder(tx, AttrSource(self.source, "state"))(
            self.value.state
        )

        # We need to realize the top level state dict to populate
        # the guard locals
        state_vt.realize()

        # Populate self.grad_to_source and self.tensor_to_source so that we can
        # manually update_list_args
        for g_ind, (group, group_vt) in enumerate(
            zip(self.value.param_groups, param_groups_vt.items)
        ):
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
                            state_source = AttrSource(self.source, "state")
                            LazyVariableTracker.realize_all(
                                VariableBuilder(
                                    tx,
                                    GetItemSource(
                                        state_source,
                                        ConstDictKeySource(state_source, key_index),
                                    ),
                                )(self.value.state[param])
                            )
                            break

            group_source = group_vt.source
            params_vt = group_vt.getitem_const(tx, ConstantVariable.create("params"))
            for p_ind, (p, p_vt) in enumerate(
                zip(group["params"], params_vt.unpack_var_sequence(tx))
            ):
                param_source = p_vt.source
                self.tensor_to_source[p] = param_source
                grad_source = GradSource(
                    param_source,
                    "grad",
                )

                if p.grad is not None:
                    self.grad_to_source[p.grad] = grad_source
                else:
                    install_guard(grad_source.make_guard(GuardBuilder.CONSTANT_MATCH))

        # We have to again iterate over the state dict to collect the
        # tensor_to_source dict. This is used for the finalizer.
        state_source = AttrSource(self.source, "state")
        for idx, (p, value) in enumerate(self.value.state.items()):
            p_state_source = GetItemSource(
                state_source, ConstDictKeySource(state_source, idx)
            )
            for k, v in value.items():
                if (
                    isinstance(v, torch.Tensor)
                    and v not in self.grad_to_source
                    and v not in self.tensor_to_source
                ):
                    self.tensor_to_source[v] = GetItemSource(p_state_source, k)

    def wrap_tensor(self, tx: "InstructionTranslator", tensor_value):
        """Wrap state tensor in a TensorVariable"""
        from ..decorators import mark_static_address
        from .builder import VariableBuilder

        # If we have a source for a tensor already use it,
        # if we have not seen a tensor before, stash and use a
        # global weak ref source, since it must be an optimizer tensor
        # that we have missed

        if tensor_value in self.tensor_to_source:
            # mark these tensors as static for cudagraphs
            mark_static_address(tensor_value)
            builder = VariableBuilder(tx, self.tensor_to_source[tensor_value])
            self.static_tensor_names.add(tx.output.module_key_name(builder.name))
        elif tensor_value in self.grad_to_source:
            builder = VariableBuilder(tx, self.grad_to_source[tensor_value])
        else:
            # mark these tensors as static for cudagraphs
            mark_static_address(tensor_value)

            global_name = tx.store_global_weakref_by_id(GLOBAL_KEY_PREFIX, tensor_value)
            builder = VariableBuilder(tx, GlobalWeakRefSource(global_name))
            self.static_tensor_names.add(tx.output.module_key_name(builder.name))

        result = builder(tensor_value)
        return result

    def update_list_args(
        self, tx: "InstructionTranslator", args, kwargs, py_args, py_kwargs
    ):
        """Update the args and kwargs to the traced optimizer call"""
        for arg, py_arg in zip(args, py_args):
            if isinstance(arg, ListVariable):
                assert isinstance(
                    py_arg, list
                ), "py_arg should be a list in optimizer variable"
                for i, val in enumerate(py_arg):
                    tx.output.side_effects.mutation(arg)
                    if isinstance(val, torch.Tensor):
                        arg.items.append(self.wrap_tensor(tx, val))
                    else:
                        from .builder import SourcelessBuilder, VariableBuilder

                        if arg.source:
                            arg.items.append(
                                VariableBuilder(tx, GetItemSource(arg.source, i))(val)
                            )
                        else:
                            arg.items.append(SourcelessBuilder.create(tx, val))

    def create_finalizer(self, tx):
        names_to_delete = self.static_tensor_names
        value = self.value
        tc = tx.output.tracing_context

        def init_finalizer(gm):
            def clear_static_tensor_refs():
                for name in names_to_delete:
                    gm._buffers.pop(name, None)
                    gm._parameters.pop(name, None)
                    if tc.params_flat:
                        tc.params_flat.clear()

            weakref.finalize(value, clear_static_tensor_refs)

        tx.output.add_graph_finalizer(init_finalizer)
