# mypy: ignore-errors

"""
Distributed computing variable tracking classes for PyTorch Dynamo.

This module implements variable tracking for distributed computing components:
- Process Groups (for collective communication)
- Device Meshes (for distributed tensor sharding)
- Placement Types (for specifying distribution strategies)
- Distributed Tensors and their operations
- Backward hooks for distributed module operations

These classes are responsible for tracking distributed operations during graph
compilation while maintaining proper guards and handling distributed-specific
behaviors. They ensure correct handling of distributed components like process
groups, device meshes, and placement strategies while preserving proper semantics
for distributed tensor operations in the compiled code.

The implementation provides special handling for distributed package availability
checks and proper tracking of distributed state and operations across processes.
"""

import functools
import inspect
from typing import TYPE_CHECKING

import torch
from torch.fx.experimental._backward_state import BackwardState

from .. import compiled_autograd, variables
from .._trace_wrapped_higher_order_op import trace_wrapped
from ..exc import unimplemented_v2
from ..external_utils import call_module_hooks_from_backward_state
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource
from ..utils import istype
from .base import VariableTracker
from .constant import ConstantVariable, EnumVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


class DistributedVariable(VariableTracker):
    """
    The base distributed variable that encapsulates common methods
    for the distributed objects (i.e. ProcessGroup, DeviceMesh, etc.).
    Concrete distributed objects could inherit this class and add object
    specific logic.

    i.e. It provides the check on the distributed package existence
    and hold the tracking value for the corresponding distributed object.
    """

    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        if not DistributedVariable.is_available():
            unimplemented_v2(
                gb_type="torch.distributed package is not available!",
                context="",
                explanation="The PyTorch package doesn't include torch.distributed when building from source.",
                hints=[
                    "Set USE_DISTRIBUTED=1 to enable it when building PyTorch from source."
                ],
            )
        self.value = value

    def python_type(self):
        return type(self.value)

    @staticmethod
    def is_available():
        # check if the distributed package is available or not
        return torch.distributed.is_available()


def is_from_local(value):
    if not DistributedVariable.is_available():
        return False
    from torch.distributed.tensor import DTensor

    return inspect.isfunction(value) and value is DTensor.from_local


def is_constant_pg_functions(value):
    if not DistributedVariable.is_available():
        return False

    from torch.distributed.distributed_c10d import (
        _get_group_size_by_name,
        _get_group_tag,
        _rank_not_in_group,
        _resolve_group_name_by_ranks_and_tag,
        get_process_group_ranks,
    )

    constant_processgroup_functions = [
        _get_group_size_by_name,
        _get_group_tag,
        _rank_not_in_group,
        get_process_group_ranks,
        _resolve_group_name_by_ranks_and_tag,
    ]

    return inspect.isfunction(value) and value in constant_processgroup_functions


class WorldMetaClassVariable(DistributedVariable):
    """
    Tracks torch.distributed.GroupMember and torch.distributed.group, which are
    instances of the metaclass _WorldMeta.
    """

    @classmethod
    def is_group_member_type(cls, value):
        if not cls.is_available():
            return False

        from torch.distributed.distributed_c10d import _WorldMeta

        return type(value) is _WorldMeta

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "WORLD":
            source = AttrSource(base=self.source, member="WORLD")
            install_guard(source.make_guard(GuardBuilder.ID_MATCH))
            return ProcessGroupVariable(self.value.WORLD)
        elif name == "NON_GROUP_MEMBER":
            source = AttrSource(base=self.source, member="NON_GROUP_MEMBER")
            install_guard(source.make_guard(GuardBuilder.ID_MATCH))
            return EnumVariable(self.value.NON_GROUP_MEMBER)
        return super().var_getattr(tx, name)


class PlacementClassVariable(DistributedVariable):
    @staticmethod
    def is_placement_type(value):
        # we can't rely on importing/accessing torch distributed, it is not always built.
        if not DistributedVariable.is_available():
            return False

        from torch.distributed.tensor.placement_types import Placement

        return type(value) is type and issubclass(value, Placement)

    def as_python_constant(self):
        return self.value

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if (
            inspect.getattr_static(self.value, "__new__", None) in (object.__new__,)
            and self.source
        ):
            # NOTE: we don't need to track mutations to the placement class as they
            # suppose to be immutable.
            new_obj = object.__new__(self.value)
            var = PlacementVariable(new_obj)
            if inspect.getattr_static(self.value, "__init__", None):
                var.call_method(tx, "__init__", args, kwargs)
                return var

        return super().call_function(tx, args, kwargs)


class PlacementVariable(DistributedVariable):
    @staticmethod
    def is_placement(value):
        # we can't rely on importing/accessing torch distributed, it is not always built.
        if not DistributedVariable.is_available():
            return False

        from torch.distributed.tensor.placement_types import Placement

        return isinstance(value, Placement)

    def as_python_constant(self):
        return self.value

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "dim":
            return ConstantVariable.create(self.value.dim)
        return super().var_getattr(tx, name)

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import ConstantVariable

        # Placement types dynamo tracking only allows following methods
        # and __setattr__  is for case like `Shard(dim)` and methods.
        # Methods in the list must satisfy:
        #    1. Input arguments are constants and do not need to be guarded on;
        #    2. Output is constant with respect to their inputs
        constant_fold_functions = [
            "__init__",
            "__setattr__",
            "is_shard",
            "is_partial",
            "is_replicate",
        ]

        if name in constant_fold_functions:
            try:
                value_type = type(self.value)
                assert (
                    inspect.getattr_static(value_type, "__getattr__", None) is None
                ), "no custom getattr allowed!"
                method = inspect.getattr_static(value_type, name)
            except AttributeError:
                method = None
            if method is object.__init__:
                return ConstantVariable.create(None)

            args = [x.as_python_constant() for x in args]
            kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
            if name == "__setattr__":
                method(self.value, *args, **kwargs)
                return self
            constant_val = method(self.value, *args, **kwargs)
            return ConstantVariable.create(constant_val)

        return super().call_method(tx, name, args, kwargs)


class DeviceMeshVariable(DistributedVariable):
    @staticmethod
    def is_device_mesh(value):
        # we can't rely on importing/accessing torch distributed, it is not always built.
        if not DistributedVariable.is_available():
            return False

        from torch.distributed.device_mesh import DeviceMesh

        return istype(value, DeviceMesh)

    def as_python_constant(self):
        return self.value

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "ndim":
            return ConstantVariable.create(self.value.ndim)
        if name == "device_type":
            return ConstantVariable.create(self.value.device_type)
        return super().var_getattr(tx, name)

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "size":
            const_args = [x.as_python_constant() for x in args]
            const_kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
            return ConstantVariable.create(self.value.size(*const_args, **const_kwargs))
        if name == "get_coordinate":
            return ConstantVariable.create(self.value.get_coordinate())
        if name == "get_group":
            const_args = [x.as_python_constant() for x in args]
            const_kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
            return ProcessGroupVariable(
                self.value.get_group(*const_args, **const_kwargs)
            )
        if name == "_get_or_create_default_group":
            return ProcessGroupVariable(self.value._get_or_create_default_group())
        return super().call_method(tx, name, args, kwargs)


class ProcessGroupVariable(DistributedVariable):
    """
    We don't want a ProcessGroup object to end up in our output graph.

    But it's common for dynamo to intercept a PG that is then used to get info like
    rank() or world_size(), as well as passed to utility functions in distributed_c10d
    which desugar it into plain types like a ranklist and tag.

    For convenience and proper guarding, we construct a variable type.

    TODO: make it possible to use ProcessGroupVariable as input to simple functions
          like _expand_group without dynamo complaining about making a proxy for it.
          It is not a tensor-like type, and we don't want a proxy- but dynamo assumes
          torch library functions are dealing with tensor-like types and would have proxies
          for their args.
    TODO: should we make this inherit VT instead of UDOV? Do we want any of the default behaviors
          or just graph-break whenever one of our special cases is not hit?
    """

    def as_python_constant(self):
        return self.value

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "rank":
            return variables.ConstantVariable.create(self.value.rank())
        if name == "size":
            return variables.ConstantVariable.create(self.value.size())
        if name == "_get_backend_name":
            return variables.ConstantVariable.create(self.value._get_backend_name())

        return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx: "InstructionTranslator", name):
        if name == "group_name":
            return variables.ConstantVariable.create(self.value.group_name)
        if name in ["rank", "size"]:
            return variables.LambdaVariable(
                lambda *args, **kwargs: self.call_method(tx, name, args, kwargs)
            )
        # TODO should this just raise unimplemented?
        return super().var_getattr(tx, name)

    @staticmethod
    def is_process_group(value):
        # we can't rely on importing/accessing torch distributed, it is not always built.
        if not DistributedVariable.is_available():
            return False
        from torch._C._distributed_c10d import ProcessGroup
        from torch.testing._internal.distributed.fake_pg import FakeProcessGroup

        return istype(value, (ProcessGroup, FakeProcessGroup))


class BackwardHookVariable(VariableTracker):
    """
    Handles torch.utils.hooks.BackwardHook for module-level backward
    hooks.
    """

    @staticmethod
    def create(
        tx,
        module: VariableTracker,
        user_hooks: VariableTracker,
        user_pre_hooks: VariableTracker,
    ):
        if not compiled_autograd.compiled_autograd_enabled:
            unimplemented_v2(
                gb_type="Module-level backwards hooks require compiled autograd.",
                context="",
                explanation="",
                hints=[
                    "Enable compiled autograd by setting torch._dynamo.config.compiled_autograd = True."
                ],
            )

        def _in_graph_bw_hooks(bw_state: BackwardState):
            """
            Rather than installing the user hooks in the graph (which
            don't survive AotAutograd), we install hooks that will call
            trace_wrapped in the backward pass that CompiledAutograd
            can turn into actual hook calls.
            """
            return torch.utils.hooks.BackwardHook(
                None,
                (
                    functools.partial(
                        trace_wrapped,
                        fn=call_module_hooks_from_backward_state,
                        bw_state=bw_state,
                        hooks_name=user_hooks_name,
                        module_name=module_name,
                    ),
                ),
                (
                    functools.partial(
                        trace_wrapped,
                        fn=call_module_hooks_from_backward_state,
                        bw_state=bw_state,
                        hooks_name=user_pre_hooks_name,
                        module_name=module_name,
                    ),
                ),
            )

        module_name, bw_state_proxy = tx.output.add_backward_state_hook(module, "mod")
        user_pre_hooks_name, _ = tx.output.add_backward_state_hook(user_pre_hooks)
        user_hooks_name, _ = tx.output.add_backward_state_hook(user_hooks)
        proxy = tx.output.create_proxy(
            "call_function",
            _in_graph_bw_hooks,
            (bw_state_proxy,),
            {},
        )
        proxy.node.meta["example_value"] = torch.utils.hooks.BackwardHook(None, (), ())
        return BackwardHookVariable(proxy, module, user_hooks, user_pre_hooks)

    def __init__(
        self,
        proxy: torch.fx.Proxy,
        module: VariableTracker,
        user_hooks: VariableTracker,
        user_pre_hooks: VariableTracker,
        **options,
    ) -> None:
        super().__init__(**options)
        self.proxy = proxy
        self.module = module
        self.user_hooks = user_hooks
        self.user_pre_hooks = user_pre_hooks

    def as_proxy(self):
        return self.proxy

    def call_method(
        self,
        tx,
        name,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name in ("setup_input_hook", "setup_output_hook"):
            return self._setup_hook(tx, name, *args, **kwargs)
        return super().call_method(tx, name, args, kwargs)

    def _setup_hook(self, tx: "InstructionTranslator", hook_method_name, args):
        from .builder import wrap_fx_proxy

        return wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_method",
                hook_method_name,
                (self.as_proxy(), args.as_proxy()),
                {},
            ),
        )
