# mypy: ignore-errors

import weakref
from typing import Dict, List

import torch

from ..decorators import mark_static_address

from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, ConstDictKeySource, GetItemSource, GlobalWeakRefSource
from ..utils import GLOBAL_KEY_PREFIX

from .base import VariableTracker
from .constant import ConstantVariable
from .dicts import ConstDictVariable
from .lists import ListVariable
from .misc import GetAttrVariable
from .user_defined import UserDefinedObjectVariable


class ArgMappingException(Exception):
    pass


class GuardInstallException(Exception):
    pass


class OptimizerVariable(UserDefinedObjectVariable):
    def __init__(
        self,
        value,
        grad_to_source=None,
        static_tensor_names=None,
        tensor_to_source=None,
        **kwargs,
    ):
        super().__init__(value, **kwargs)

        for group in self.value.param_groups:
            if "capturable" in group:
                group["capturable"] = True

            for p in group["params"]:
                mark_static_address(p, guard=False)

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

    def var_getattr(self, tx, name):
        if name == "_init_group":
            return GetAttrVariable(self, name)

        return super().var_getattr(tx, name)

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

            raise ArgMappingException()

        new_args = [map_arg(arg) for arg in args]
        new_kwargs = {k: map_arg(v) for k, v in kwargs.items()}

        return new_args, new_kwargs

    def map_sources_and_install_guards(self, tx):
        self.grad_to_source = {}
        self.tensor_to_source = {}

        from .builder import VariableBuilder

        param_groups_vt = VariableBuilder(tx, AttrSource(self.source, "param_groups"))(
            self.value.param_groups
        ).recursive_realize()

        for g_ind, (group, group_vt) in enumerate(
            zip(self.value.param_groups, param_groups_vt.items)
        ):
            group_source = group_vt.source
            params_vt = group_vt.getitem_const(ConstantVariable.create("params"))
            for p_ind, (p, p_vt) in enumerate(
                zip(group["params"], params_vt.unpack_var_sequence(tx))
            ):
                param_source = p_vt.source
                self.tensor_to_source[p] = param_source
                grad_source = AttrSource(
                    param_source,
                    "grad",
                )
                if p.grad is not None:
                    self.grad_to_source[p.grad] = grad_source
                else:
                    install_guard(grad_source.make_guard(GuardBuilder.CONSTANT_MATCH))

        # state guards take a long time to generate
        # so we manually generate them here
        state_source = AttrSource(self.source, "state")
        install_guard(state_source.make_guard(GuardBuilder.DICT_KEYS))
        for idx, (p, value) in enumerate(self.value.state.items()):
            tx.store_global_weakref_by_id(GLOBAL_KEY_PREFIX, p)
            p_state_source = GetItemSource(
                state_source, ConstDictKeySource(state_source, idx)
            )
            install_guard(p_state_source.make_guard(GuardBuilder.DICT_KEYS))
            for k, v in value.items():
                if (
                    isinstance(v, torch.Tensor)
                    and v not in self.grad_to_source
                    and v not in self.tensor_to_source
                ):
                    self.tensor_to_source[v] = GetItemSource(p_state_source, k)
                elif v is None or isinstance(v, (bool, int, float, str)):
                    install_guard(
                        GetItemSource(p_state_source, k).make_guard(
                            GuardBuilder.CONSTANT_MATCH
                        )
                    )
                else:
                    raise GuardInstallException()

    def wrap_tensor(self, tx, tensor_value):
        """Wrap state tensor in a TensorVariable"""
        from .builder import VariableBuilder

        # If we have a source for a tensor already use it,
        # if we have not seen a tensor before, stash and use a
        # global weak ref source, since it must be an optimizer tensor
        # that we have missed

        if tensor_value in self.tensor_to_source:
            # mark these tensors as static for cudagraphs
            mark_static_address(tensor_value, guard=False)
            builder = VariableBuilder(tx, self.tensor_to_source[tensor_value])
            self.static_tensor_names.add(tx.output.module_key_name(builder.name))
        elif tensor_value in self.grad_to_source:
            builder = VariableBuilder(tx, self.grad_to_source[tensor_value])
        else:
            # mark these tensors as static for cudagraphs
            mark_static_address(tensor_value, guard=False)

            global_name = tx.store_global_weakref_by_id(GLOBAL_KEY_PREFIX, tensor_value)
            builder = VariableBuilder(tx, GlobalWeakRefSource(global_name))
            self.static_tensor_names.add(tx.output.module_key_name(builder.name))

        result = builder(tensor_value)
        return result

    def update_list_args(self, tx, args, kwargs, py_args, py_kwargs):
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
                            arg.items.append(SourcelessBuilder()(tx, val))

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
