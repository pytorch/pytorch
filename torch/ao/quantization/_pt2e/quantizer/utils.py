import torch
from torch.ao.quantization._pt2e.quantizer.quantizer import (
    get_observer_kwargs,
    QuantizationConfig,
    QuantizationSpec,
)
from torch.ao.quantization.observer import (
    _PartialWrapper,
    PlaceholderObserver,
)
from torch.ao.quantization.qconfig import _obs_or_fq_ctr_equals

def create_observer(quantization_spec: QuantizationSpec, **extra_kwargs):
    if quantization_spec is None:
        return None
    observer_or_fake_quant_ctr = quantization_spec.observer_or_fake_quant_ctr
    kwargs = get_observer_kwargs(quantization_spec)
    kwargs.pop("observer_or_fake_quant_ctr")
    # we will remove is_dynamic from QuantizationSpec because
    # it seems that dynamic range quantization
    if not _obs_or_fq_ctr_equals(observer_or_fake_quant_ctr, PlaceholderObserver):
        kwargs.pop("is_dynamic")
    obs_or_fq_class = observer_or_fake_quant_ctr
    if isinstance(observer_or_fake_quant_ctr, _PartialWrapper):
        obs_or_fq_class = observer_or_fake_quant_ctr.p.func  # type: ignore[union-attr, assignment]
    if "PerChannel" not in obs_or_fq_class.__name__:  # type: ignore[operator, union-attr]
        kwargs.pop("ch_axis")
    return observer_or_fake_quant_ctr.with_args(**kwargs, **extra_kwargs)


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
    return create_observer(quantization_spec)

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
    return create_observer(quantization_spec)

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
    return create_observer(quantization_spec)
