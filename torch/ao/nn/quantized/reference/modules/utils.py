import torch
import typing

class ReferenceQuantizedModule(torch.nn.Module):
    def _init_weight_qparams(self, weight_qparams, device):
        if weight_qparams is None:
            weight_qparams = {
                "qscheme": torch.per_tensor_affine,
                "dtype": torch.quint8,
                "scale": 1.0,
                "zero_point": 0
            }
        self.weight_qscheme: torch.qscheme = weight_qparams["qscheme"]
        self.weight_dtype = weight_qparams["dtype"]
        assert self.weight_qscheme in [
            None, torch.per_tensor_affine, torch.per_channel_affine,
            torch.per_channel_affine_float_qparams], \
            Exception(f"qscheme: {self.weight_qscheme} is not support in reference quantized {self._get_name()}")
        if self.weight_dtype in [torch.quint8, torch.qint8, torch.quint4x2, torch.qint32]:
            zero_point_dtype = weight_qparams["zero_point"].dtype if \
                isinstance(weight_qparams["zero_point"], torch.Tensor) else \
                torch.int
            w_scale = weight_qparams["scale"]
            w_scale_tensor = w_scale.clone().detach() \
                if isinstance(w_scale, torch.Tensor) \
                else torch.tensor(w_scale, dtype=torch.float, device=device)
            self.register_buffer("weight_scale", w_scale_tensor)
            w_zp = weight_qparams["zero_point"]
            w_zp_tensor = w_zp.clone().detach() \
                if isinstance(w_zp, torch.Tensor) \
                else torch.tensor(w_zp, dtype=zero_point_dtype, device=device)
            self.register_buffer("weight_zero_point", w_zp_tensor)
            if self.weight_qscheme in [torch.per_channel_affine, torch.per_channel_affine_float_qparams]:
                w_axis = weight_qparams["axis"]
                w_axis_tensor = w_axis.clone().detach() \
                    if isinstance(w_axis, torch.Tensor) \
                    else torch.tensor(w_axis, dtype=torch.int, device=device)
                self.register_buffer("weight_axis", w_axis_tensor)
            else:
                # added for TorchScriptability, not used
                self.register_buffer(
                    "weight_axis", torch.tensor(0, dtype=torch.int, device=device))
        else:
            # added for TorchScriptability, and for torch.float
            self.register_buffer("weight_scale", torch.tensor(1.0, dtype=torch.float, device=device))
            self.register_buffer("weight_zero_point", torch.tensor(0, dtype=torch.int, device=device))
            self.register_buffer(
                "weight_axis", torch.tensor(0, dtype=torch.int, device=device))

    def get_weight(self):
        """
        Fake quantize (quantize and dequantize) the weight with
        the quantization parameters for weight, this is used to
        simulate the numerics for the quantized weight in a quantized
        model
        """
        # suppress mypy warning
        assert isinstance(self.weight_scale, torch.Tensor)
        assert isinstance(self.weight_zero_point, torch.Tensor)
        assert isinstance(self.weight_axis, torch.Tensor)
        return _quantize_and_dequantize_weight(
            self.weight,  # type: ignore[arg-type]
            self.weight_qscheme,
            self.weight_dtype,
            self.weight_scale,
            self.weight_zero_point, self.weight_axis)

    def get_quantized_weight(self):
        # suppress mypy warning
        assert isinstance(self.weight_scale, torch.Tensor)
        assert isinstance(self.weight_zero_point, torch.Tensor)
        assert isinstance(self.weight_axis, torch.Tensor)
        return _quantize_weight(
            self.weight,  # type: ignore[arg-type]
            self.weight_qscheme,
            self.weight_dtype,
            self.weight_scale,
            self.weight_zero_point,
            self.weight_axis)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        _save_weight_qparams(
            destination, prefix, self.weight_qscheme, self.weight_dtype,
            self.weight_scale, self.weight_zero_point, self.weight_axis)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for key in _get_weight_qparam_keys(state_dict, prefix):
            setattr(self, key, state_dict[prefix + key])
            state_dict.pop(prefix + key)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False,
            missing_keys, unexpected_keys, error_msgs)

def _quantize_weight(
        weight: torch.Tensor,
        weight_qscheme: torch.qscheme,
        weight_dtype: torch.dtype,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor,
        weight_axis: torch.Tensor):
    if weight_dtype == torch.float16:
        weight = weight.to(weight_dtype)
        return weight

    if weight_qscheme == torch.per_tensor_affine:
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight = torch.quantize_per_tensor(weight, weight_scale, weight_zero_point, weight_dtype)
            return weight
    elif weight_qscheme in [torch.per_channel_affine, torch.per_channel_affine_float_qparams]:
        if weight_dtype in [torch.quint8, torch.qint8, torch.quint4x2, torch.qint32]:
            weight = torch.quantize_per_channel(
                weight, weight_scale,
                weight_zero_point, weight_axis.item(), weight_dtype)  # type: ignore[arg-type]
            return weight
    raise Exception(f"Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}")

def _quantize_and_dequantize_weight(
        weight: torch.Tensor,
        weight_qscheme: torch.qscheme,
        weight_dtype: torch.dtype,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor,
        weight_axis: torch.Tensor):
    """ Quantize and then dequantize the weight based on
    the quantization parameters
    """
    if weight_qscheme in [
            torch.per_tensor_affine,
            torch.per_channel_affine,
            torch.per_channel_affine_float_qparams]:
        weight_quant = _quantize_weight(
            weight, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis)
        weight_dequant = weight_quant.dequantize()
    else:
        weight_dequant = weight
    return weight_dequant

def _save_weight_qparams(destination, prefix, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis):
    destination[prefix + "weight_qscheme"] = weight_qscheme
    destination[prefix + "weight_dtype"] = weight_dtype
    if weight_qscheme is not None:
        destination[prefix + "weight_scale"] = weight_scale
        destination[prefix + "weight_zero_point"] = weight_zero_point
        if weight_qscheme == torch.per_channel_affine:
            destination[prefix + "weight_axis"] = weight_axis

def _get_weight_qparam_keys(
        state_dict: typing.Dict[str, typing.Any],
        prefix: str):
    keys = ["weight_qscheme", "weight_dtype"]
    weight_qscheme = state_dict[prefix + "weight_qscheme"]
    if weight_qscheme is not None:
        keys.append("weight_scale")
        keys.append("weight_zero_point")
        if weight_qscheme == torch.quantize_per_channel:
            keys.append("weight_axis")
    return keys
