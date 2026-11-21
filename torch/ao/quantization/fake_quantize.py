# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
"""Implements modules  used to perform fake quantization."""

import re
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch._compile import _assume_constant_result
from torch.ao.quantization.observer import (
    _with_args,
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer,
    FixedQParamsObserver,
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
)
from torch.nn import Module


__all__ = [
    "FakeQuantizeBase",
    "FakeQuantize",
    "FixedQParamsFakeQuantize",
    "FusedMovingAvgObsFakeQuantize",
    "disable_fake_quant",
    "disable_observer",
    "enable_fake_quant",
    "enable_observer",
    "default_fake_quant",
    "default_weight_fake_quant",
    "default_dynamic_fake_quant",
    "default_fixed_qparams_range_neg1to1_fake_quant",
    "default_fixed_qparams_range_0to1_fake_quant",
    "default_symmetric_fixed_qparams_fake_quant",
    "default_affine_fixed_qparams_fake_quant",
    "default_per_channel_weight_fake_quant",
    "default_embedding_fake_quant",
    "default_embedding_fake_quant_4bit",
    "default_histogram_fake_quant",
    "default_fused_act_fake_quant",
    "default_fused_wt_fake_quant",
    "default_fused_per_channel_wt_fake_quant",
    "fused_wt_fake_quant_range_neg_127_to_127",
    "fused_per_channel_wt_fake_quant_range_neg_127_to_127",
]


def _is_per_channel(qscheme: "torch.qscheme") -> bool:
    return qscheme in [
        torch.per_channel_symmetric,
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
    ]


def _is_per_tensor(qscheme: "torch.qscheme") -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]


def _is_symmetric_quant(qscheme: "torch.qscheme") -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]


def _is_float_qparams(qscheme: "torch.qscheme") -> bool:
    return qscheme == torch.per_channel_affine_float_qparams


# Dynamo calls this while tracing, with the device of the input, which it guards on, and
# does not put the call into the graph.
@_assume_constant_result
def _device_has_fake_quantize_kernels(device_type):
    # Whether the kernels that FakeQuantize._forward_compiled calls have an implementation
    # for this device. They are looked up in the dispatcher, and not in a list of device
    # types, so that a backend that registers them is traced. A fallback of the backend
    # for all ops is not a kernel.
    try:
        key = torch._C._dispatch_key_for_device(device_type)
    except RuntimeError:
        # not a device type with a dispatch key
        return False
    return all(
        torch._C._dispatch_has_kernel_for_dispatch_key(op, key)
        for op in (
            "aten::fake_quantize_per_channel_affine_cachemask",
            "aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
        )
    )


class FakeQuantizeBase(ABC, Module):
    r"""Base fake quantize module.

    Base fake quantize module
    Any fake quantize implementation should derive from this class.

    Concrete fake quantize module should follow the same API. In forward, they will update
    the statistics of the observed Tensor and fake quantize the input. They should also provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    """

    fake_quant_enabled: torch.Tensor
    observer_enabled: torch.Tensor

    def __init__(self) -> None:
        """Set fake_quant_enabled and observer_enabled."""
        super().__init__()
        # fake_quant_enabled and observer_enabled are buffers to support their
        # replication in DDP. Data type is uint8 because NCCL does not support
        # bool tensors.
        self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.uint8))
        self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.uint8))

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    @torch.jit.export
    def enable_fake_quant(self, enabled: bool = True) -> None:
        self.fake_quant_enabled[0] = 1 if enabled else 0

    @torch.jit.export
    def disable_fake_quant(self):
        self.enable_fake_quant(False)

    @torch.jit.export
    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled[0] = 1 if enabled else 0

    @torch.jit.export
    def disable_observer(self):
        self.enable_observer(False)

    @classmethod
    def with_args(cls, **kwargs):
        fake_quant_constructor = _with_args(cls, **kwargs)
        # need to assign the correct module to fake_quantize
        # constructors to satisfy public v private requirements
        fake_quant_constructor.__module__ = "torch.ao.quantization.fake_quantize"
        return fake_quant_constructor


# Why FakeQuantize._forward_compiled cannot be used. Each reason is also the reason of the
# disabled eager forward that the call graph-breaks into.
_COMPILED_FORWARD_BLOCKERS = {
    "hooks": (
        "FakeQuantize cannot be traced when its observer has forward hooks (its own or "
        "global ones, for example from register_module_forward_hook): the traced forward "
        "also runs the observer while observer_enabled is 0, where eager does not, so the "
        "hooks would see calls that eager never makes. Remove the hooks, or run the module "
        "eagerly: torch.compile without fullgraph=True does that for this call, "
        "torch.export cannot trace it (torch.export.draft_export traces the eager forward "
        "instead)."
    ),
    "unsupported_type": (
        "FakeQuantize is only traced for FakeQuantize and FixedQParamsFakeQuantize with a "
        "MinMaxObserver, MovingAverageMinMaxObserver, PerChannelMinMaxObserver, "
        "MovingAveragePerChannelMinMaxObserver or FixedQParamsObserver (the exact classes, "
        "not subclasses): the traced forward restores the state that the observer and "
        "calculate_qparams update when observer_enabled is 0, which needs all of it to be "
        "known. Use one of these classes, or run the module eagerly: torch.compile without "
        "fullgraph=True does that for this call, torch.export cannot trace it "
        "(torch.export.draft_export traces the eager forward instead)."
    ),
    "device": (
        "FakeQuantize cannot be traced for an input on a device that has no kernel for the "
        "fake-quantize ops (for example mps): the traced forward calls the kernel also while "
        "fake_quant_enabled is 0, where eager does not. Run the module eagerly: "
        "torch.compile without fullgraph=True does that for this call, torch.export cannot "
        "trace it (torch.export.draft_export traces the eager forward instead)."
    ),
    "statistics": (
        "A per-channel FakeQuantize cannot be traced before its statistics exist: run one "
        "eager forward first, with the observer enabled (for example a calibration step), "
        "and compile or export after that. torch.compile without fullgraph=True runs this "
        "call eagerly instead."
    ),
    "shape": (
        "FakeQuantize cannot be traced when its scale and zero_point do not have the shape "
        "that observing this input gives them (for example after the number of channels "
        "changed): the eager forward resizes them, which a traced region cannot do. Run one "
        "eager forward with this input first, with the observer enabled (after "
        "reset_min_max_vals() on the observer if the number of channels changed). "
        "torch.compile without fullgraph=True runs this call eagerly instead."
    ),
}


class FakeQuantize(FakeQuantizeBase):
    r"""Simulate the quantize and dequantize operations in training time.

    The output of this module is given by::

        x_out = (
            clamp(round(x / scale + zero_point), quant_min, quant_max) - zero_point
        ) * scale

    * :attr:`is_dynamic` indicates whether the fake quantize is a placeholder for dynamic quantization
      operators (choose_qparams -> q -> dq) or static quantization operators (q -> dq)

    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to

    * :attr:`fake_quant_enabled` controls the application of fake quantization on tensors, note that
      statistics can still be updated.

    * :attr:`observer_enabled` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
        allowable values are torch.qint8 and torch.quint8.

    .. note::
        Under :func:`torch.compile` and :func:`torch.export` this module is traced with a
        forward that differs from the eager one, because ``observer_enabled`` and
        ``fake_quant_enabled`` are tensors. It runs the observer and ``calculate_qparams``
        on every step, also when ``observer_enabled`` is 0 (the statistics of the observer
        and the qparams are restored afterwards), and the fake-quantize kernel also runs
        when ``fake_quant_enabled`` is 0, and the module returns the input then. A
        per-channel module gives the kernel a clamped ``zero_point`` then, so that its range
        check cannot fail; the check still synchronizes with the device for an integer ``zero_point`` (a
        floating-point ``zero_point``, as with ``torch.per_channel_affine_float_qparams``,
        is not checked), which CUDA graphs cannot capture. With Inductor's
        ``graph_partition`` (on by default outside fbcode) only the partition of that op is
        run without CUDA graphs and the rest of the graph is captured; with
        ``graph_partition=False``, ``cpp_wrapper`` or the ``cudagraphs`` backend the whole
        compiled graph is not captured, and that is the whole traced region, which can be
        most of the model. A negative ``ch_axis``, which the observers accept, is
        normalized before the kernel gets it, so the traced forward returns a result where
        the eager forward raises an error once ``fake_quant_enabled`` is 1 (the kernel
        rejects a negative axis). The
        ``min_val > max_val`` assertion and the "must run observer" warning of
        ``calculate_qparams`` are skipped, and an observer that has not observed anything
        returns a ``zero_point`` with the dtype of the computed one (``torch.int32`` for
        ``torch.per_tensor_affine``) instead of ``torch.int64``. The module is only traced for
        ``FakeQuantize`` and ``FixedQParamsFakeQuantize`` themselves (not subclasses) with
        ``MinMaxObserver``, ``MovingAverageMinMaxObserver``, ``PerChannelMinMaxObserver``,
        ``MovingAveragePerChannelMinMaxObserver`` or ``FixedQParamsObserver`` (not
        subclasses), without forward hooks or forward pre-hooks on the observer (its own, or
        global ones from ``register_module_forward_hook`` and
        ``register_module_forward_pre_hook``), with a ``scale`` and ``zero_point`` that
        already have the shape that observing the input gives them, for an input on a device
        that has kernels for the fake-quantize ops (for example CPU, CUDA and XPU, or another
        backend that registers kernels for these two ops; a fallback of the backend for all
        ops does not count, so with ``PYTORCH_ENABLE_MPS_FALLBACK=1`` the module runs
        eagerly on MPS), and, for a per-channel module, after one eager forward (for example a calibration step, with the observer
        enabled) has created its statistics. Otherwise :func:`torch.compile` graph-breaks
        and runs it eagerly, and :func:`torch.export` raises an error that says why.
        :func:`torch.export.draft_export` traces the eager forward with real tensors
        instead, as before, so the branches on ``observer_enabled`` and
        ``fake_quant_enabled`` that it takes are fixed in the exported program.

    Args:

        observer (module): Module for observing statistics on input tensors and calculating scale
          and zero-point.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        activation_post_process (Module): User provided module that collects statistics on the input tensor and
          provides a method to calculate scale and zero-point.

    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
        self,
        observer=MovingAverageMinMaxObserver,
        quant_min=None,
        quant_max=None,
        is_dynamic=False,
        **observer_kwargs,
    ):
        super().__init__()
        # Populate quant_min/quant_max to observer_kwargs if valid
        if quant_min is not None and quant_max is not None:
            if quant_min > quant_max:
                raise AssertionError(
                    "quant_min must be less than or equal to quant_max"
                )
            dtype = observer_kwargs.get("dtype", torch.quint8)
            if hasattr(observer, "p"):
                # In case observer is _PartialWrapper, dtype can be stored in
                # observer.p.keywords["dtype"]
                dtype = getattr(getattr(observer, "p", {}), "keywords", {}).get(
                    "dtype", dtype
                )

            if torch.iinfo(dtype).min > quant_min:
                raise AssertionError("quant_min out of bound")

            if quant_max > torch.iinfo(dtype).max:
                raise AssertionError("quant_max out of bound")
            observer_kwargs.update({"quant_min": quant_min, "quant_max": quant_max})
        observer_kwargs["is_dynamic"] = is_dynamic
        self.activation_post_process = observer(**observer_kwargs)
        # TODO: keeping self.quant_min/max for BC; remove after a couple releases
        # Users should use self.activation_post_process.quant_min
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.is_dynamic = self.activation_post_process.is_dynamic
        if _is_float_qparams(self.activation_post_process.qscheme):
            zero_point_dtype = torch.float
        else:
            zero_point_dtype = torch.int
        self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([0], dtype=zero_point_dtype))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = (
            self.activation_post_process.ch_axis
            if hasattr(self.activation_post_process, "ch_axis")
            else -1
        )
        if not (_is_per_channel(self.qscheme) or _is_per_tensor(self.qscheme)):
            raise AssertionError(
                "Only per channel and per tensor quantization are supported in fake quantize"
                + " got qscheme: "
                + str(self.qscheme)
            )
        self.is_per_channel = _is_per_channel(self.qscheme)

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        return self.activation_post_process.calculate_qparams()

    def _fake_quantize(self, X):
        if self.is_per_channel:
            return torch.fake_quantize_per_channel_affine(
                X,
                self.scale,
                self.zero_point,
                self.ch_axis,
                self.activation_post_process.quant_min,
                self.activation_post_process.quant_max,
            )
        return torch.fake_quantize_per_tensor_affine(
            X,
            self.scale,
            self.zero_point,
            self.activation_post_process.quant_min,
            self.activation_post_process.quant_max,
        )

    def _compiled_forward_blocker(self, X):
        # The key of _COMPILED_FORWARD_BLOCKERS that says why _forward_compiled cannot be
        # used for this module and input, or None when it can. It observes
        # unconditionally and then restores the observer's state when the observer is
        # disabled, so it has to know all of that state: it is only used for the classes
        # below (a subclass may update more) and when nothing else runs with the
        # observer, and the qparams must already have the shape that observing X gives
        # them, because a buffer cannot be resized in a traced region and a new shape
        # cannot be selected against the old state.
        observer = self.activation_post_process
        if (
            observer._forward_hooks
            or observer._forward_pre_hooks
            or torch.nn.modules.module._global_forward_hooks
            or torch.nn.modules.module._global_forward_pre_hooks
        ):
            return "hooks"
        if type(self) not in (FakeQuantize, FixedQParamsFakeQuantize) or type(
            observer
        ) not in (
            FixedQParamsObserver,
            MinMaxObserver,
            MovingAverageMinMaxObserver,
            PerChannelMinMaxObserver,
            MovingAveragePerChannelMinMaxObserver,
        ):
            return "unsupported_type"
        # the traced forward calls the kernel on every step, eager only when fake-quant is
        # enabled
        if not _device_has_fake_quantize_kernels(X.device.type):
            return "device"
        if self.is_per_channel and observer.min_val.numel() == 0:
            return "statistics"
        shape = (X.size(self.ch_axis),) if self.is_per_channel else (1,)
        if self.scale.shape != shape or self.zero_point.shape != shape:
            return "shape"
        return None

    def _forward_compiled(self, X):
        # ``observer_enabled`` and ``fake_quant_enabled`` are tensors, so branching
        # on them is data-dependent control flow that torch.compile cannot trace.
        # Run both steps unconditionally and select the result with torch.where.
        observer = self.activation_post_process
        observer_enabled = self.observer_enabled[0] == 1
        observer_state = (
            []
            if type(observer) is FixedQParamsObserver
            else [observer.min_val, observer.max_val]
        )
        saved = [t.clone() for t in observer_state]

        observer(X.detach())
        _scale, _zero_point = self.calculate_qparams()
        self.scale.copy_(
            torch.where(observer_enabled, _scale.to(self.scale.device), self.scale)
        )
        self.zero_point.copy_(
            torch.where(
                observer_enabled,
                _zero_point.to(self.zero_point.device),
                self.zero_point,
            )
        )
        for t, old in zip(observer_state, saved):
            t.copy_(torch.where(observer_enabled, t, old))

        if self.is_per_channel:
            fake_quant_enabled = self.fake_quant_enabled[0] == 1
            quant_min = self.activation_post_process.quant_min
            quant_max = self.activation_post_process.quant_max
            # the observer accepts a negative ch_axis and the kernel does not, which eager
            # only calls when fake-quant is enabled, and this path calls on every step
            ch_axis = self.ch_axis + X.dim() if self.ch_axis < 0 else self.ch_axis
            # The kernel runs on every step. It checks (and synchronizes with the device
            # to do so) that an integer zero_point lies in [quant_min, quant_max], which
            # eager only does when fake-quant is enabled, so it gets one that passes while
            # fake-quant is disabled.
            zero_point = torch.where(
                fake_quant_enabled,
                self.zero_point,
                self.zero_point.clamp(quant_min, quant_max),
            )
            return torch.where(
                fake_quant_enabled,
                torch.fake_quantize_per_channel_affine(
                    X, self.scale, zero_point, ch_axis, quant_min, quant_max
                ),
                X,
            )
        # this op takes the flag itself (and passes X through when it is 0), so there is
        # no extra pass over X; it treats any nonzero value as enabled, and eager only 1
        return torch._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
            X,
            self.scale,
            self.zero_point,
            (self.fake_quant_enabled == 1).long(),
            self.activation_post_process.quant_min,
            self.activation_post_process.quant_max,
        )[0]

    def forward(self, X):
        if not torch.jit.is_scripting():
            if torch.compiler.is_compiling():
                blocker = self._compiled_forward_blocker(X)
                if blocker is None:
                    return self._forward_compiled(X)
                # one graph break, with the reason of the blocker
                return _EAGER_FORWARDS[blocker](self, X)
        return self._forward_eager(X)

    def _forward_eager(self, X):
        if not torch.jit.is_scripting():
            if (
                torch.compiler.is_exporting()
                and not torch._functorch.config.fake_tensor_propagate_real_tensors
            ):
                # Non-strict export has no graph break to report the reason with, and
                # would fail on the branch on observer_enabled below. draft_export
                # propagates real tensors, which can take that branch.
                blocker = self._compiled_forward_blocker(X)
                if blocker is not None:
                    raise RuntimeError(
                        f"{type(self).__name__} with "
                        f"{type(self.activation_post_process).__name__}: "
                        f"{_COMPILED_FORWARD_BLOCKERS[blocker]}"
                    )
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = (
                _scale.to(self.scale.device),
                _zero_point.to(self.zero_point.device),
            )
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            X = self._fake_quantize(X)
        return X

    @torch.jit.export
    def extra_repr(self):
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, "
            f"dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}, "
            f"scale={self.scale}, zero_point={self.zero_point}"
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "scale"] = self.scale
        destination[prefix + "zero_point"] = self.zero_point

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Removing this function throws an error that the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ["scale", "zero_point"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == "scale":
                    self.scale.resize_(val.shape)
                else:
                    if name != "zero_point":
                        raise AssertionError(
                            "Expected 'zero_point' but got different state key"
                        )
                    self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == "scale":
                        self.scale.copy_(val)
                    else:
                        if name != "zero_point":
                            raise AssertionError(
                                "Expected 'zero_point' but got different state key"
                            )
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class FixedQParamsFakeQuantize(FakeQuantize):
    """Simulate quantize and dequantize in training time.

    Simulate quantize and dequantize with fixed quantization
    parameters in training time. Only per tensor quantization
    is supported.
    """

    # TODO: rename observer to observer_ctr
    def __init__(self, observer):
        super().__init__(observer=observer)
        if type(self.activation_post_process) is not FixedQParamsObserver:
            raise AssertionError(
                f"{self.__class__.__name__}'s observer must be a {FixedQParamsObserver.__name__}"
            )
        self._observer_ctr = observer
        self.scale = self.activation_post_process.scale
        self.zero_point = self.activation_post_process.zero_point
        if not _is_per_tensor(self.qscheme):
            raise AssertionError(
                "Only per tensor quantization is supported"
                + " FixedQParamsFakeQuantize module, got qscheme:"
                + str(self.qscheme)
            )

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        return self.scale, self.zero_point

    @torch.jit.export
    def extra_repr(self):
        """Define a string representation of the object's attributes."""
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"scale={self.scale}, zero_point={self.zero_point}, "
            f"dtype={self.dtype}, quant_min={self.activation_post_process.quant_min}, "
            f"quant_max={self.activation_post_process.quant_max}, qscheme={self.qscheme}"
        )


class FusedMovingAvgObsFakeQuantize(FakeQuantize):
    r"""Define a fused module to observe the tensor.

    Fused module that is used to observe the input tensor (compute min/max), compute
    scale/zero_point and fake_quantize the tensor.
    This module uses calculation similar MovingAverageMinMaxObserver for the inputs,
    to compute the min/max values in order to compute the scale/zero_point.
    The qscheme input in the observer is used to differentiate between symmetric/affine
    quantization scheme.

    The output of this module is given by
    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale

    Similar to :class:`~torch.ao.quantization.FakeQuantize`, and accepts the same attributes as the
    base class.

    .. note::
        Under :func:`torch.compile` and :func:`torch.export` a per-channel module is only
        traced after one eager forward (for example a calibration step, with the observer
        enabled) has created its statistics, because the fake kernel of the fused op does
        not create them. Before that :func:`torch.compile` graph-breaks and runs the op
        eagerly, strict :func:`torch.export` raises an error that says why, and non-strict
        :func:`torch.export` runs the op with fake tensors as before.

        The fused op is in the traced graph of a default QAT model, per-tensor as well as
        per-channel. CUDA graphs cannot capture it (it is in Inductor's
        ``FORBIDDEN_CUDAGRAPH_OPS``): with ``graph_partition`` (on by default outside fbcode)
        its partition runs without CUDA graphs, and with ``graph_partition=False``,
        ``cpp_wrapper`` or the ``cudagraphs`` backend the whole compiled graph is not
        captured.

    """

    def __init__(
        self,
        observer: Any = MovingAverageMinMaxObserver,
        quant_min: int = 0,
        quant_max: int = 255,
        **observer_kwargs: Any,
    ) -> None:
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)
        if not isinstance(
            self.activation_post_process,
            (MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver),
        ):
            raise AssertionError(
                "Fused observer+fake_quant module only works with MovingAverageMinMaxObserver"
            )
        self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.long))
        self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.long))
        self.is_symmetric_quant = _is_symmetric_quant(
            self.activation_post_process.qscheme
        )

    @torch.jit.export
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.activation_post_process.calculate_qparams()

    @torch.jit.export
    def extra_repr(self) -> str:
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"scale={self.scale}, zero_point={self.zero_point}, dtype={self.dtype}, "
            f"quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, "
            f"qscheme={self.qscheme}, reduce_range={self.activation_post_process.reduce_range}"
        )

    def _fused_forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.fused_moving_avg_obs_fake_quant(
            X,
            self.observer_enabled,
            self.fake_quant_enabled,
            self.activation_post_process.min_val,
            self.activation_post_process.max_val,
            self.scale,
            self.zero_point,
            self.activation_post_process.averaging_constant,
            self.activation_post_process.quant_min,
            self.activation_post_process.quant_max,
            self.ch_axis,
            self.is_per_channel,
            self.is_symmetric_quant,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting():
            if (
                torch.compiler.is_compiling()
                and self.is_per_channel
                and self.activation_post_process.min_val.numel() == 0
            ):
                # TODO: remove this guard once the fake kernel of the fused op creates the
                # per-channel statistics (#199920, #199938)
                return _fused_fake_quantize_forward_before_statistics_exist(self, X)
        return self._fused_forward(X)


@torch._disable_dynamo(library_fallback_reason=_COMPILED_FORWARD_BLOCKERS["hooks"])
def _fake_quantize_forward_with_observer_hooks(module, X):
    return FakeQuantize._forward_eager(module, X)


@torch._disable_dynamo(
    library_fallback_reason=_COMPILED_FORWARD_BLOCKERS["unsupported_type"]
)
def _fake_quantize_forward_with_unsupported_classes(module, X):
    return FakeQuantize._forward_eager(module, X)


@torch._disable_dynamo(library_fallback_reason=_COMPILED_FORWARD_BLOCKERS["device"])
def _fake_quantize_forward_on_a_device_without_the_kernel(module, X):
    return FakeQuantize._forward_eager(module, X)


@torch._disable_dynamo(library_fallback_reason=_COMPILED_FORWARD_BLOCKERS["statistics"])
def _fake_quantize_forward_before_statistics_exist(module, X):
    return FakeQuantize._forward_eager(module, X)


@torch._disable_dynamo(library_fallback_reason=_COMPILED_FORWARD_BLOCKERS["shape"])
def _fake_quantize_forward_with_changed_qparams_shape(module, X):
    return FakeQuantize._forward_eager(module, X)


# The calls that a traced FakeQuantize graph-breaks into when it cannot be traced (one per
# key of _COMPILED_FORWARD_BLOCKERS), so that the break is reported once, with a reason, and
# what follows it does not trace the call again.
_EAGER_FORWARDS = {
    "hooks": _fake_quantize_forward_with_observer_hooks,
    "unsupported_type": _fake_quantize_forward_with_unsupported_classes,
    "device": _fake_quantize_forward_on_a_device_without_the_kernel,
    "statistics": _fake_quantize_forward_before_statistics_exist,
    "shape": _fake_quantize_forward_with_changed_qparams_shape,
}


@torch._disable_dynamo(
    library_fallback_reason=(
        "A per-channel FusedMovingAvgObsFakeQuantize cannot be traced before its "
        "statistics exist, because the fake kernel of the fused op does not create them: "
        "run one eager forward first, with the observer enabled (for example a calibration "
        "step), and compile or export after that. torch.compile without fullgraph=True "
        "runs this call eagerly instead."
    )
)
def _fused_fake_quantize_forward_before_statistics_exist(module, X):
    return FusedMovingAvgObsFakeQuantize._fused_forward(module, X)


default_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    reduce_range=True,
)
"""
Default fake_quant for activations.
"""

default_weight_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
    reduce_range=False,
)
"""
Default fake_quant for weights.
Observer is memoryless since averaging_constant is 1.
"""

default_dynamic_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    is_dynamic=True,
    dtype=torch.quint8,
    averaging_constant=1,
)
"""
Default dynamic fake_quant for activations.
"""

default_fixed_qparams_range_neg1to1_fake_quant = FixedQParamsFakeQuantize.with_args(
    observer=default_fixed_qparams_range_neg1to1_observer
)
default_fixed_qparams_range_0to1_fake_quant = FixedQParamsFakeQuantize.with_args(
    observer=default_fixed_qparams_range_0to1_observer
)
# TODO: the following 2 variables are kept for backwards compatibility; remove after a few releases
default_symmetric_fixed_qparams_fake_quant = (
    default_fixed_qparams_range_neg1to1_fake_quant
)
default_affine_fixed_qparams_fake_quant = default_fixed_qparams_range_0to1_fake_quant

default_per_channel_weight_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
    reduce_range=False,
    ch_axis=0,
)
"""
Default fake_quant for per-channel weights.
Observer is memoryless since averaging_constant is 1.
"""
default_embedding_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    qscheme=torch.per_channel_affine_float_qparams,
    dtype=torch.quint8,
    quant_min=0,
    quant_max=255,
    ch_axis=0,
    averaging_constant=1,
)
"""
Default fake_quant for embeddings.
Observer is memoryless since averaging_constant is 1.
"""

default_embedding_fake_quant_4bit = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    qscheme=torch.per_channel_affine_float_qparams,
    ch_axis=0,
    dtype=torch.quint4x2,
    averaging_constant=1,
)

default_histogram_fake_quant = FakeQuantize.with_args(
    observer=HistogramObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    reduce_range=True,
)
"""
Fake_quant for activations using a histogram..
"""


default_fused_act_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
)

"""
Fused version of `default_fake_quant`, with improved performance.
"""


default_fused_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
)
"""
Fused version of `default_weight_fake_quant`, with improved performance.
"""

default_fused_per_channel_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
)
"""
Fused version of `default_per_channel_weight_fake_quant`, with improved performance.
"""

fused_wt_fake_quant_range_neg_127_to_127 = FusedMovingAvgObsFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-127,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
    eps=2**-12,
)
"""
Fused version of `default_weight_fake_quant`, with the 8-bit values restricted to [-127, +127], excluding -128.
"""

fused_per_channel_wt_fake_quant_range_neg_127_to_127 = (
    FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-127,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        eps=2**-12,
    )
)

"""
Fused version of `default_per_channel_weight_fake_quant`, with the 8-bit values restricted to [-127, +127], excluding -128.
"""


def _is_fake_quant_script_module(mod):
    """Return true if given mod is an instance of FakeQuantize script module."""
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        # qualified name looks like '__torch__.torch.ao.quantization.fake_quantize.___torch_mangle_2.FakeQuantize'
        suffix = mod._c.qualified_name.split(".", 1)[1]
        name = re.sub(r"\.___torch_mangle_\d+", "", suffix)
        return (
            name == "torch.ao.quantization.fake_quantize.FakeQuantize"
            or name
            == "torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize"
        )
    return False


def disable_fake_quant(mod):
    """Disable fake quantization for the module.

    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.disable_fake_quant()


def enable_fake_quant(mod):
    """Enable fake quantization for the module.

    Enable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_fake_quant)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.enable_fake_quant()


def disable_observer(mod):
    """Disable observation for this module.

    Disable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_observer)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.disable_observer()


def enable_observer(mod):
    """Enable observation for this module.

    Enable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_observer)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.enable_observer()
