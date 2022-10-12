import dataclasses
from typing import Iterator, Optional, Tuple

import torch
from torch._C._profiler import _EventType, _ProfilerEvent, _TensorMetadata, RecordScope


@dataclasses.dataclass(eq=True, unsafe_hash=True, frozen=True)
class TensorKey:
    """Hashable identifier for a storage which has been asigned an ID.

    A detailed description of Tensor IDs and why they are needed is given in
    `torch/csrc/profiler/collection.h` when `TensorID` is declared. To
    summarize, multiple Storage buffers can map to the same logical Tensor.
    This dataclass is used to refer to a concrete in-memory StorageImpl of
    a Tensor.
    """

    id: int
    storage_ptr: int
    device: torch.device

    def __repr__(self) -> str:
        return f"id={self.id}: {hex(self.storage_ptr):>18} ({self.device})"

    @staticmethod
    def _make(
        tensor_id: Optional[int], ptr: Optional[int], device: torch.device
    ) -> Optional["TensorKey"]:
        if tensor_id is not None and ptr is not None:
            return TensorKey(tensor_id, ptr, device)
        return None

    @classmethod
    def from_tensor(cls, t: Optional[_TensorMetadata]) -> Optional["TensorKey"]:
        return cls._make(t.id, t.storage_data_ptr, t.device) if t else None


def extract_gradients(
    node: _ProfilerEvent,
) -> Iterator[Tuple[Optional[TensorKey], TensorKey]]:
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
        node.typed[0] == _EventType.TorchOp
        and node.typed[1].scope == RecordScope.BACKWARD_FUNCTION
        # TODO(robieta): Move away from load bearing names
        and node.name == "torch::autograd::AccumulateGrad"
        and children
        and children[0].typed[0] == _EventType.TorchOp
        and children[0].name in ("aten::detach", "aten::add_")
        and children[0].typed[1].inputs.tensor_metadata
    ):
        key = TensorKey.from_tensor(children[0].typed[1].inputs.tensor_metadata[0])
        if key:
            yield None, key

    # We directly instrument `torch.nn.Module` and `torch.optim.Optimizer`
    # NOTE: The values captured by the python tracer are cached; they can be
    #       used to build up labels but do not imply that a Tensor was live at
    #       a particular time.
    elif node.typed[0] == _EventType.PyCall:
        typed_fields = node.typed[1]
        assert typed_fields.module is None or typed_fields.optimizer is None
        if typed_fields.module is not None:
            for _, p, p_grad in typed_fields.module.parameters:
                p_grad_key = TensorKey.from_tensor(p_grad)
                if p_grad_key is not None:
                    yield TensorKey.from_tensor(p), p_grad_key

        if typed_fields.optimizer is not None:
            for p, p_grad, _ in typed_fields.optimizer.parameters:
                p_grad_key = TensorKey.from_tensor(p_grad)
                if p_grad_key is not None:
                    yield TensorKey.from_tensor(p), p_grad_key
