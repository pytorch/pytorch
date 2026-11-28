"""Iterator-owned shared input slots for default collation with CUDA pinning."""

from typing import NamedTuple

import torch

from .collate import collate_tensor_fn, default_collate, default_collate_fn_map
from .pin_memory import pin_memory


class _TensorSlice(NamedTuple):
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]

    def view(self, buffer):
        contents = buffer.narrow(0, self.offset, self.nbytes)
        return contents.view(self.dtype).view(self.shape)


class _SharedBatch(NamedTuple):
    slot: int
    data: object


class SharedBatchPool:
    def __init__(self, slots, capacity, context):
        self.capacity = capacity
        self.buffers = [
            torch.empty(0, dtype=torch.uint8, device="cpu").set_(
                torch.UntypedStorage._new_shared(capacity, device="cpu")
            )
            for _ in range(slots)
        ]
        # A worker acquires the slot; the pin thread releases it after copying.
        self.locks = [context.Lock() for _ in range(slots)]

    def collate(self, batch):
        first = batch[0]
        mapping = type(first) is dict
        if (
            default_collate_fn_map.get(torch.Tensor) is not collate_tensor_fn
            or (type(first) is not torch.Tensor and not mapping)
            or (mapping and any(isinstance(first, t) for t in default_collate_fn_map))
        ):
            return default_collate(batch)
        for slot, lock in enumerate(self.locks):
            if lock.acquire(False):
                break
        else:
            # A small pool must not block the worker or depend on delivery order.
            return default_collate(batch)

        offset = 0
        handed_off = False

        def stack(samples):
            nonlocal offset
            elem = samples[0]
            if type(elem) is torch.Tensor and all(
                type(t) is torch.Tensor
                and t.device.type == "cpu"
                and t.layout == torch.strided
                and not t.is_nested
                and not t.is_quantized
                and not t.requires_grad
                and t.dtype == elem.dtype
                and t.shape == elem.shape
                for t in samples
            ):
                size = elem.element_size()
                start = (offset + size - 1) // size * size
                nbytes = len(samples) * elem.numel() * size
                if nbytes and start + nbytes <= self.capacity:
                    leaf = _TensorSlice(
                        start, nbytes, elem.dtype, (len(samples), *elem.shape)
                    )
                    torch.stack(samples, out=leaf.view(self.buffers[slot]))
                    offset = start + nbytes
                    return leaf
            return default_collate(samples)

        try:
            if mapping:
                data = {key: stack([sample[key] for sample in batch]) for key in first}
            else:
                data = stack(batch)
            if not offset:
                return data
            result = _SharedBatch(slot, data)
            handed_off = True
            return result
        finally:
            if not handed_off:
                lock.release()

    def pin(self, batch, device):
        if not isinstance(batch, _SharedBatch):
            return pin_memory(batch, device)

        def pin_leaf(value):
            if isinstance(value, _TensorSlice):
                source = value.view(self.buffers[batch.slot])
                # The delivered tensor must outlive reuse of this slot.
                return torch.empty_like(source, pin_memory=True).copy_(source)
            return pin_memory(value, device)

        try:
            if type(batch.data) is dict:
                return {key: pin_leaf(value) for key, value in batch.data.items()}
            return pin_leaf(batch.data)
        finally:
            self.locks[batch.slot].release()
