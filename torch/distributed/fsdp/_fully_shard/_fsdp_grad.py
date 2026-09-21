from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Replicate
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._utils import (
    compute_local_shape_and_global_offset,
    compute_local_stride,
)
from torch.utils import _pytree as pytree


_SYNC_MESSAGE = (
    "Call synchronize_gradients() on every rank before modifying the complete "
    "gradient with this operation."
)


class FSDPGrad(DTensor):
    """A gradient with separately stored reduced and pending contributions.

    The public dtype is the reduced gradient dtype. ``unreduced`` and ``partial``
    retain the autograd accumulation dtype; neither includes ``reduced``.
    Reading values may run collectives. Zeroing and scalar scaling act on the
    components without communication and must be called consistently on all ranks.
    Synchronize gradients before calling an optimizer step. Pending gradients
    cannot participate in mutable parameter or optimizer-state operations.
    """

    reduced: DTensor | None
    unreduced: DTensor | None
    partial: DTensor | None
    sharded_spec: DTensorSpec
    _materialize_fn: Callable[["FSDPGrad"], DTensor]

    @staticmethod
    def __new__(
        cls,
        reduced: DTensor | None,
        unreduced: DTensor | None,
        partial: DTensor | None,
        sharded_spec: DTensorSpec,
        materialize_fn: Callable[["FSDPGrad"], DTensor],
    ) -> "FSDPGrad":
        component = next(
            (t for t in (reduced, unreduced, partial) if t is not None), None
        )
        if component is None:
            raise ValueError("FSDPGrad requires at least one gradient component")
        if sharded_spec.tensor_meta is None:
            raise ValueError("FSDPGrad requires sharded tensor metadata")
        return torch.Tensor._make_wrapper_subclass(
            cls,
            sharded_spec.shape,
            strides=sharded_spec.stride,
            dtype=sharded_spec.tensor_meta.dtype,
            device=component.device,
            requires_grad=False,
        )

    def __init__(
        self,
        reduced: DTensor | None,
        unreduced: DTensor | None,
        partial: DTensor | None,
        sharded_spec: DTensorSpec,
        materialize_fn: Callable[["FSDPGrad"], DTensor],
    ) -> None:
        for component in (reduced, unreduced, partial):
            if component is None:
                continue
            if type(component) is not DTensor:
                raise TypeError("FSDPGrad components must be ordinary DTensors")
            if component.shape != sharded_spec.shape:
                raise ValueError("FSDPGrad components must have the gradient shape")
        if reduced is not None and reduced.dtype != self.dtype:
            raise ValueError(
                "The reduced component must have the public gradient dtype"
            )
        if (
            unreduced is not None
            and partial is not None
            and unreduced.dtype != partial.dtype
        ):
            raise ValueError("Pending gradient components must have the same dtype")
        self.reduced = reduced
        self.unreduced = unreduced
        self.partial = partial
        self.sharded_spec = sharded_spec
        self._materialize_fn = materialize_fn

    def __repr__(self) -> str:
        return (
            f"FSDPGrad(reduced={self.reduced}, unreduced={self.unreduced}, "
            f"partial={self.partial}, dtype={self.dtype})"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "data":
            raise RuntimeError(
                f"Replacing FSDPGrad storage is unsupported. {_SYNC_MESSAGE}"
            )
        super().__setattr__(name, value)

    @property
    def device_mesh(self) -> DeviceMesh:
        return self.sharded_spec.mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        raise RuntimeError(
            "FSDPGrad has no single placement; inspect reduced, unreduced, and partial, "
            "or call synchronize_gradients() on every rank."
        )

    def local_components(self) -> dict[str, torch.Tensor]:
        """Return physical local contributions, each with its own dtype and shape.

        These are writable aliases of the individual contributions, not of their
        logical sum. They do not require communication.
        """
        return {
            name: component._local_tensor
            for name in ("reduced", "unreduced", "partial")
            if (component := getattr(self, name)) is not None
        }

    def materialize(self) -> DTensor:
        """Collectively return an independent, reduced, sharded gradient snapshot."""
        with torch.no_grad():
            result = self._materialize_fn(self)
        if type(result) is not DTensor or result._spec != self.sharded_spec:
            raise RuntimeError(
                "FSDPGrad materialization must return an ordinary DTensor with "
                "the sharded gradient spec"
            )
        return result

    def to_local(
        self, *, grad_placements: Sequence[Placement] | None = None
    ) -> torch.Tensor:
        """Return a storage-less view of the canonical local gradient shard.

        Creating this view does not communicate. Reads materialize a snapshot;
        whole-view zeroing and scalar scaling modify all components and require
        the same operation on every rank. Subset writes and raw storage exports
        require synchronizing the gradients first.
        """
        if grad_placements is not None:
            raise RuntimeError("FSDPGrad does not support differentiable local views")
        return _FSDPGradLocalTensor(self)

    def full_tensor(
        self, *, grad_placements: Sequence[Placement] | None = None
    ) -> torch.Tensor:
        return self.materialize().full_tensor(grad_placements=grad_placements)

    def redistribute(
        self,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        *,
        async_op: bool = False,
        forward_dtype: torch.dtype | None = None,
        backward_dtype: torch.dtype | None = None,
    ) -> DTensor:
        return self.materialize().redistribute(
            device_mesh=device_mesh,
            placements=placements,
            async_op=async_op,
            forward_dtype=forward_dtype,
            backward_dtype=backward_dtype,
        )

    def __tensor_flatten__(self):
        names = [
            name
            for name in ("reduced", "unreduced", "partial")
            if getattr(self, name) is not None
        ]
        return names, (self.sharded_spec, self._materialize_fn)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        raise RuntimeError(f"Tracing pending FSDPGrad is unsupported. {_SYNC_MESSAGE}")

    def __reduce_ex__(self, protocol):
        raise RuntimeError(
            f"Serializing pending FSDPGrad is unsupported. {_SYNC_MESSAGE}"
        )

    def numpy(self, *, force=False):
        raise RuntimeError(f"FSDPGrad has no NumPy storage alias. {_SYNC_MESSAGE}")

    def data_ptr(self):
        raise RuntimeError(f"FSDPGrad has no single storage. {_SYNC_MESSAGE}")

    def storage(self):
        raise RuntimeError(f"FSDPGrad has no single storage. {_SYNC_MESSAGE}")

    def untyped_storage(self):
        raise RuntimeError(f"FSDPGrad has no single storage. {_SYNC_MESSAGE}")

    def __dlpack__(self, *args, **kwargs):
        raise RuntimeError(f"FSDPGrad has no DLPack storage alias. {_SYNC_MESSAGE}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return _dispatch_grad(func, args, kwargs or {})


class _FSDPGradLocalTensor(torch.Tensor):
    grad_owner: FSDPGrad
    views: tuple
    meta: torch.Tensor

    @staticmethod
    def __new__(cls, grad, views=(), meta=None):
        if meta is None:
            shape, _ = compute_local_shape_and_global_offset(
                grad.shape,
                grad.sharded_spec.mesh,
                grad.sharded_spec.placements,
                skip_offset=True,
            )
            stride = compute_local_stride(grad.stride(), shape)
            meta = torch.empty_strided(shape, stride, dtype=grad.dtype, device="meta")
        result = torch.Tensor._make_wrapper_subclass(
            cls,
            meta.shape,
            strides=meta.stride(),
            storage_offset=meta.storage_offset(),
            dtype=grad.dtype,
            device=grad.device,
            requires_grad=False,
        )
        result.grad_owner = grad
        result.views = views
        result.meta = meta
        return result

    def __repr__(self, *, tensor_contents=None):
        return f"FSDPGradLocalTensor(shape={tuple(self.shape)}, dtype={self.dtype})"

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "data":
            raise RuntimeError(
                f"Replacing FSDPGrad storage is unsupported. {_SYNC_MESSAGE}"
            )
        super().__setattr__(name, value)

    def materialize(self):
        value = self.grad_owner.materialize().to_local()
        for func, args, kwargs in self.views:
            value = func(value, *args, **kwargs)
        return value

    numpy = FSDPGrad.numpy
    data_ptr = FSDPGrad.data_ptr
    storage = FSDPGrad.storage
    untyped_storage = FSDPGrad.untyped_storage
    __dlpack__ = FSDPGrad.__dlpack__
    __reduce_ex__ = FSDPGrad.__reduce_ex__

    def __tensor_flatten__(self):
        return ["grad_owner"], (self.views, self.meta)

    __tensor_unflatten__ = staticmethod(FSDPGrad.__tensor_unflatten__)

    @staticmethod
    def __torch_dispatch__(func, types, args=(), kwargs=None):
        return _dispatch_grad(func, args, kwargs or {})


_GRAD_TYPES = (FSDPGrad, _FSDPGradLocalTensor)
_LOCAL_VIEW_OPS = {
    torch.ops.aten.alias.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten._reshape_alias.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.unsqueeze.default,
}


def _scalar(value):
    if isinstance(value, DTensor):
        value = (
            value.to_local()
            if all(isinstance(p, Replicate) for p in value.placements)
            else value.full_tensor()
        )
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or isinstance(value, _GRAD_TYPES):
            raise RuntimeError(
                f"FSDPGrad only supports scalar scaling. {_SYNC_MESSAGE}"
            )
    elif not isinstance(value, (int, float, complex)):
        raise RuntimeError(f"FSDPGrad only supports scalar scaling. {_SYNC_MESSAGE}")
    return value


def _mutate(grad, operation, scalar=None):
    if isinstance(grad, _FSDPGradLocalTensor):
        grad = grad.grad_owner
    for local in grad.local_components().values():
        if operation == "zero":
            local.zero_()
        else:
            value = (
                scalar.to(local.device) if isinstance(scalar, torch.Tensor) else scalar
            )
            if operation == "mul":
                local.mul_(value)
            else:
                local.div_(value)


def _dispatch_grad(func, args, kwargs):
    name = func._schema.name
    if name in ("aten::detach", "aten::alias", "aten::clone"):
        grad = args[0]
        if isinstance(grad, FSDPGrad):
            components = [
                None if c is None else func(c, **kwargs)
                for c in (grad.reduced, grad.unreduced, grad.partial)
            ]
            return FSDPGrad(
                components[0],
                components[1],
                components[2],
                grad.sharded_spec,
                grad._materialize_fn,
            )
        if name != "aten::clone":
            return _FSDPGradLocalTensor(grad.grad_owner, grad.views, grad.meta)
    if name == "aten::detach_":
        return args[0]
    if name == "aten::zero_" or (
        name == "aten::fill_" and not isinstance(args[1], torch.Tensor) and args[1] == 0
    ):
        _mutate(args[0], "zero")
        return args[0]
    if name in ("aten::mul_", "aten::div_") and isinstance(args[0], _GRAD_TYPES):
        if kwargs.get("rounding_mode") is not None:
            raise RuntimeError(
                f"FSDPGrad does not support rounded division. {_SYNC_MESSAGE}"
            )
        _mutate(args[0], "mul" if name == "aten::mul_" else "div", _scalar(args[1]))
        return args[0]
    if name in (
        "aten::_foreach_mul_",
        "aten::_foreach_div_",
        "aten::_foreach_zero_",
    ) and any(isinstance(value, _GRAD_TYPES) for value in args[0]):
        values = args[0]
        operation = name.removeprefix("aten::_foreach_").removesuffix("_")
        scalars = (
            [None] * len(values)
            if operation == "zero"
            else list(args[1])
            if isinstance(args[1], (tuple, list))
            else [args[1]] * len(values)
        )
        if len(scalars) != len(values):
            raise RuntimeError("foreach scaling requires one scalar per gradient")
        scalars = [None if operation == "zero" else _scalar(s) for s in scalars]
        for value, scalar in zip(values, scalars):
            if isinstance(value, _GRAD_TYPES):
                _mutate(value, operation, scalar)
            elif operation == "zero":
                value.zero_()
            elif operation == "mul":
                value.mul_(scalar)
            else:
                value.div_(scalar)
        return None
    if name == "aten::_amp_foreach_non_finite_check_and_unscale_":
        grads, found_inf, inv_scale = args
        amp_snapshots = [
            g.materialize() if isinstance(g, _GRAD_TYPES) else g for g in grads
        ]
        func(amp_snapshots, found_inf, inv_scale, **kwargs)
        for grad in grads:
            if isinstance(grad, _GRAD_TYPES):
                _mutate(grad, "mul", inv_scale)
        return None
    if args and isinstance(args[0], _FSDPGradLocalTensor) and func in _LOCAL_VIEW_OPS:
        grad = args[0]
        meta = func(grad.meta, *args[1:], **kwargs)
        return _FSDPGradLocalTensor(
            grad.grad_owner, (*grad.views, (func, args[1:], kwargs)), meta
        )
    if func._schema.is_mutable:
        raise RuntimeError(f"FSDPGrad does not support {func}. {_SYNC_MESSAGE}")
    if any(result.alias_info is not None for result in func._schema.returns):
        raise RuntimeError(
            f"FSDPGrad does not support this view: {func}. {_SYNC_MESSAGE}"
        )

    snapshots: dict[int, Any] = {}

    def unwrap(value):
        if isinstance(value, _GRAD_TYPES):
            key = id(value)
            if key not in snapshots:
                snapshots[key] = value.materialize()
            return snapshots[key]
        return value

    return func(*pytree.tree_map(unwrap, args), **pytree.tree_map(unwrap, kwargs))
