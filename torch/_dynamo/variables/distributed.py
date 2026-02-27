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
from typing import Any, Literal, TYPE_CHECKING

import torch
from torch.fx.experimental._backward_state import BackwardState

from .. import compiled_autograd
from .._trace_wrapped_higher_order_op import trace_wrapped
from ..exc import unimplemented
from ..external_utils import call_module_hooks_from_backward_state
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource
from .base import VariableTracker
from .constant import ConstantVariable


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

    def __init__(self, value: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not DistributedVariable.is_available():
            unimplemented(
                gb_type="torch.distributed package is not available!",
                context="",
                explanation="The PyTorch package doesn't include torch.distributed when building from source.",
                hints=[
                    "Set USE_DISTRIBUTED=1 to enable it when building PyTorch from source."
                ],
            )
        self.value = value

    def python_type(self) -> type:
        return type(self.value)

    @staticmethod
    def is_available() -> bool:
        # check if the distributed package is available or not
        return torch.distributed.is_available()

    def is_python_hashable(self) -> Literal[True]:
        return True

    def get_python_hash(self) -> int:
        return hash(self.value)

    def is_python_equal(self, other: object) -> bool:
        return (
            isinstance(other, VariableTracker)
            and self.as_python_constant() == other.as_python_constant()
        )


def is_from_local(value: object) -> bool:
    if not DistributedVariable.is_available():
        return False
    from torch.distributed.tensor import DTensor

    return inspect.isfunction(value) and value is DTensor.from_local


def is_constant_pg_functions(value: object) -> bool:
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
    def is_group_member_type(cls, value: object) -> bool:
        if not cls.is_available():
            return False

        from torch.distributed.distributed_c10d import _WorldMeta

        return type(value) is _WorldMeta

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "WORLD":
            from .builder import SourcelessBuilder

            assert self.source
            source = AttrSource(base=self.source, member="WORLD")
            install_guard(source.make_guard(GuardBuilder.ID_MATCH))
            return SourcelessBuilder.create(tx, self.value.WORLD)
        elif name == "NON_GROUP_MEMBER":
            assert self.source
            source = AttrSource(base=self.source, member="NON_GROUP_MEMBER")
            install_guard(source.make_guard(GuardBuilder.ID_MATCH))
            return VariableTracker.build(tx, self.value.NON_GROUP_MEMBER)
        return super().var_getattr(tx, name)


class BackwardHookVariable(VariableTracker):
    """
    Handles torch.utils.hooks.BackwardHook for module-level backward
    hooks.
    """

    @staticmethod
    def create(
        tx: "InstructionTranslator",
        module: VariableTracker,
        user_hooks: VariableTracker,
        user_pre_hooks: VariableTracker,
    ) -> "BackwardHookVariable":
        if not compiled_autograd.compiled_autograd_enabled:
            unimplemented(
                gb_type="Module-level backwards hooks require compiled autograd.",
                context="",
                explanation="",
                hints=[
                    "Enable compiled autograd by setting torch._dynamo.config.compiled_autograd = True."
                ],
            )

        def _in_graph_bw_hooks(
            bw_state: BackwardState,
        ) -> torch.utils.hooks.BackwardHook:
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
        **options: Any,
    ) -> None:
        super().__init__(**options)
        self.proxy = proxy
        self.module = module
        self.user_hooks = user_hooks
        self.user_pre_hooks = user_pre_hooks

    def as_proxy(self) -> torch.fx.Proxy:
        return self.proxy

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name in ("setup_input_hook", "setup_output_hook"):
            return self._setup_hook(tx, name, *args, **kwargs)
        return super().call_method(tx, name, args, kwargs)

    def _setup_hook(
        self, tx: "InstructionTranslator", hook_method_name: str, args: VariableTracker
    ) -> VariableTracker:
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


class P2POpVariable(VariableTracker):
    @staticmethod
    def can_rewrite(variable):
        return isinstance(variable, dist.P2POp)

    def __init__(
        self,
        op: VariableTracker,
        peer: VariableTracker,
        tag: VariableTracker,
        tensor: VariableTracker,
        pg: VariableTracker,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.op = op
        self.peer = peer
        self.tag = tag
        self.tensor = tensor
        self.group = pg

    @staticmethod
    def create(tx, value, args, kwargs, source):
        def get_param(name, pos, default=None, transform=None):
            if name in kwargs:
                val = kwargs[name]
            elif pos < len(args):
                val = args[pos]
            else:
                val = default

            return transform(val) if transform and val is not None else val

        op_var = get_param(
            "op", 0, transform=lambda x: ConstantVariable.create(x.get_name())
        )
        tensor_var = get_param("tensor", 1, transform=lambda x: x.realize())
        peer_var = get_param("peer", 2)
        group_var = get_param("group", 3, default=ConstantVariable.create(""))
        tag_var = get_param("tag", 4, default=ConstantVariable.create(0))

        return P2POpVariable(
            op=op_var, tensor=tensor_var, peer=peer_var, tag=tag_var, pg=group_var
        )

    def python_type(self):
        return torch.distributed.P2POp

    def as_proxy(self):
        return self.tensor.as_proxy()

    def reconstruct(self, codegen):
        unimplemented_v2("Cannot reconstruct P2POpVariable")

    def var_getattr(self, tx, name):
        if name == "op":
            return self.op
        elif name == "tensor":
            return self.tensor
        elif name == "tag":
            return self.tag
        elif name == "group":
            return self.group
        elif name == "peer":
            return self.peer
        else:
            return super().var_getattr(tx, name)
