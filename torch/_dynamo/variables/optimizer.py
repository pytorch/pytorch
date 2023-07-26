from typing import Dict, List

import torch

from ..guards import GuardBuilder
from ..source import AttrSource, GetItemSource, GlobalWeakRefSource
from ..utils import global_key_name

from .base import MutableLocal, VariableTracker
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
    def __init__(self, value, grad_to_source=None, tensor_to_source=None, **kwargs):
        super().__init__(value, **kwargs)

        for group in self.value.param_groups:
            if "capturable" in group:
                group["capturable"] = True

        if grad_to_source is None:
            self.grad_to_source = {}

        if tensor_to_source is None:
            self.tensor_to_source = {}

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        """This is an optimization to avoid tracing the very slow intialization of the optimizer"""
        if name == "_init_group":
            try:
                py_args, py_kwargs = self.get_python_args(*args, **kwargs)
                self.value._init_group(*py_args, **py_kwargs)
                self.map_sources_and_install_guards(tx)
                self.update_list_args(tx, args, kwargs, py_args, py_kwargs)
                return ConstantVariable(None)
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
        from .builder import VariableBuilder

        self.grad_to_source = {}
        self.tensor_to_source = {}

        for g_ind, group in enumerate(self.value.param_groups):
            group_source = GetItemSource(AttrSource(self.source, "param_groups"), g_ind)
            for p_ind, p in enumerate(group["params"]):
                param_source = GetItemSource(
                    GetItemSource(group_source, "params"), p_ind
                )
                self.tensor_to_source[p] = param_source
                if p.grad is not None:
                    self.grad_to_source[p.grad] = AttrSource(
                        param_source,
                        "grad",
                    )

        # state guards take a long time to generate
        # so we manually generate them here
        guards = set()
        state_source = AttrSource(self.source, "state")
        guards.add(state_source.make_guard(GuardBuilder.DICT_KEYS))
        for p, value in self.value.state.items():
            tx.store_dict_key(global_key_name(p), p)
            p_state_source = GetItemSource(state_source, self.tensor_to_source[p])
            guards.add(p_state_source.make_guard(GuardBuilder.DICT_KEYS))
            for k, v in value.items():
                if (
                    isinstance(v, torch.Tensor)
                    and v not in self.grad_to_source
                    and v not in self.tensor_to_source
                ):
                    self.tensor_to_source[v] = GetItemSource(p_state_source, k)
                elif v is None or isinstance(v, (bool, int, float, str)):
                    guards.add(
                        GetItemSource(p_state_source, k).make_guard(
                            GuardBuilder.CONSTANT_MATCH
                        )
                    )
                else:
                    raise GuardInstallException()

        tx.output.guards.update(guards)

        group_guards = VariableBuilder(tx, AttrSource(self.source, "param_groups"))(
            self.value.param_groups
        )
        tx.output.guards.update(group_guards.guards)

    def wrap_tensor(self, tx, tensor_value):
        """Wrap state tensor in a TensorVariable"""
        from .builder import VariableBuilder

        # If we have a source for a tensor already use it,
        # if we have not seen a tensor before, stash and use a
        # global weak ref source, since it must be an optimizer tensor
        # that we have missed
        if tensor_value in self.tensor_to_source:
            return VariableBuilder(tx, self.tensor_to_source[tensor_value])(
                tensor_value
            )
        elif tensor_value in self.grad_to_source:
            return VariableBuilder(tx, self.grad_to_source[tensor_value])(tensor_value)
        else:
            tx.store_dict_key(global_key_name(tensor_value), tensor_value)
            return VariableBuilder(
                tx, GlobalWeakRefSource(global_key_name(tensor_value))
            )(tensor_value)

    def update_list_args(self, tx, args, kwargs, py_args, py_kwargs):
        """Update the args and kwargs to the traced optimizer call"""
        for arg, py_arg in zip(args, py_args):
            if isinstance(arg, ListVariable) and all(
                isinstance(t, torch.Tensor) for t in py_arg
            ):
                tensor_vars = ListVariable(
                    [self.wrap_tensor(tx, t) for t in py_arg],
                    mutable_local=MutableLocal(),
                    recursively_contains={},
                )
                tx.replace_all(arg, tensor_vars)
