import torch
from torch.ao.quantization._pt2e.quantizer.quantizer import (
    get_observer_kwargs,
    QuantizationConfig,
    QuantizationSpec,
)
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)

def create_observer(observer_type, quantization_spec: QuantizationSpec, **extra_kwargs):
    if quantization_spec is None:
        return None
    kwargs = get_observer_kwargs(quantization_spec)
    # we will remove is_dynamic from QuantizationSpec because
    # it seems that dynamic range quantization
    if observer_type != PlaceholderObserver:
        kwargs.pop("is_dynamic")
    if "PerChannel" not in observer_type.__name__:
        kwargs.pop("ch_axis")
    return observer_type.with_args(**kwargs, **extra_kwargs)


def get_act_obs_or_fq_ctr(quantization_config: QuantizationConfig):
    if quantization_config is None:
        return None
    if quantization_config.activation is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.activation
    assert quantization_spec.qscheme in [
        torch.per_tensor_affine,
        torch.per_tensor_symmetric,
    ]
    if quantization_spec.is_dynamic:
        # TODO: extend this helper function to support dynamic quantization
        raise Exception(
            "Unsupported quantization_spec for activation: {}".format(quantization_spec)
        )
    if quantization_config.is_qat:
        return create_observer(
            FusedMovingAvgObsFakeQuantize,
            quantization_spec,
            reduce_range=False,
            eps=2**-12,
        )
    else:  # ptq
        return create_observer(
            HistogramObserver, quantization_spec, reduce_range=False, eps=2**-12
        )


def get_weight_obs_or_fq_ctr(quantization_config: QuantizationConfig):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    if quantization_config.weight is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.weight
    if quantization_spec.qscheme not in [
        torch.per_tensor_symmetric,
        torch.per_channel_symmetric,
    ]:
        raise ValueError(
            f"Unsupported quantization_spec {quantization_spec} for weight"
        )
    observer_type = MinMaxObserver
    extra_args = {}
    if quantization_config.is_qat:
        observer_type = FusedMovingAvgObsFakeQuantize  # type: ignore[assignment]
        if quantization_spec.qscheme == torch.per_tensor_symmetric:
            extra_args = {"observer": MovingAverageMinMaxObserver}
        else:
            extra_args = {"observer": MovingAveragePerChannelMinMaxObserver}  # type: ignore[dict-item]
    elif quantization_spec.qscheme == torch.per_channel_symmetric:
        observer_type = PerChannelMinMaxObserver  # type: ignore[assignment]
    return create_observer(observer_type, quantization_spec, eps=2**-12, **extra_args)


def get_bias_obs_or_fq_ctr(quantization_config: QuantizationConfig):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    if quantization_config.bias is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.bias
    assert (
        quantization_spec.dtype == torch.float
    ), "Only float dtype for bias is supported for bias right now"
    return PlaceholderObserver.with_args(dtype=quantization_spec.dtype)
