import dataclasses
from typing import Iterator, Optional, Tuple

import torch
from torch._C._profiler import (
    _EventType,
    _ProfilerEvent,
    _TensorMetadata,
    RecordScope,
)


@dataclasses.dataclass(eq=True, unsafe_hash=True, frozen=True)
class Key:
    id: int
    storage_ptr: int
    device: torch.device

    def __repr__(self) -> str:
        return f"id={self.id}: {hex(self.storage_ptr)} ({self.device})"

    @staticmethod
    def _make(
        id: Optional[int], ptr: Optional[int], device: torch.device
    ) -> Optional["Key"]:
        return Key(id, ptr, device) if id is not None and ptr is not None else None

    @classmethod
    def from_tensor(cls, t: Optional[_TensorMetadata]) -> Optional["Key"]:
        return cls._make(t.id, t.storage_data_ptr, t.device) if t else None


def extract_gradients(node: _ProfilerEvent) -> Iterator[Tuple[Optional[Key], Key]]:
    tag, typed_fields = node.typed
    children = node.children

    # AccumulateGrad is used in the Autograd engine to handle gradient updates.
    # There are two possible cases:
    # 1) This is a newly created gradient Tensor. In that case there is nothing
    #    to accumulate, so autograd simply detaches the Tensor.
    #
    # 2) There is a preexisting gradient Tensor and we need to add the newly
    #    computed update. This is done with an in-place add (aten::add_) op.
    #    (The underscore suffix denotes "in-place".)
    if (
        tag == _EventType.TorchOp
        and typed_fields.scope == RecordScope.BACKWARD_FUNCTION
        # TODO: Move away from load bearing names
        and node.name == "torch::autograd::AccumulateGrad"
        and children
        and children[0].typed[0] == _EventType.TorchOp
        and children[0].name in ("aten::detach", "aten::add_")
        and children[0].typed[1].inputs.tensor_metadata
        and children[0].typed[1].inputs.tensor_metadata[0].id is not None
    ):
        key = Key.from_tensor(children[0].extra_fields.inputs.tensor_metadata[0])
        yield None, key

    # We directly instrument `torch.nn.Module` and `torch.optim.Optimizer`
    # NOTE: The values captured by the python tracer are cached; they can be
    #       used to build up labels but do not imply that a Tensor was live at
    #       a particular time.
    elif tag == _EventType.PyCall:
        assert typed_fields.module is None or typed_fields.optimizer is None
        if typed_fields.module is not None:
            for _, p, p_grad in typed_fields.module.parameters:
                p_grad_key = Key.from_tensor(p_grad)
                if p_grad_key is not None:
                    yield Key.from_tensor(p), p_grad_key

        if typed_fields.optimizer is not None:
            for p, p_grad, _ in typed_fields.optimizer.parameters:
                p_grad_key = Key.from_tensor(p_grad)
                if p_grad_key is not None:
                    yield Key.from_tensor(p), p_grad_key
