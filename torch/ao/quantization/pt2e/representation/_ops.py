from torch.library import Library, impl
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax

quant_lib = Library("quant", "DEF")

quan_lib.define(
    "choose_qparams.tensor(Tensor input, int quant_min, int quant_max, "
    "float eps, ScalarType dtype) -> (Tensor, Tensor)")

@impl(quant_lib, "choose_qparams", "CompositeExplicitAutograd")
def choose_qparams(
    input: torch.Tensor,
    qmin: int,
    qmax: int,
    eps: float,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (torch.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (torch.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert dtype == torch.int8 or dtype == torch.uint8 or dtype == torch.int32, \
        f"Expecting target dtype to be int8 uint8 or int32, but got: {dtype}"
    validate_qmin_qmax(qmin, qmax)

    min_val, max_val = torch.aminmax(input)

    return determine_qparams(
        min_val, max_val, qmin, qmax, dtype, torch.Tensor([eps]), has_customized_qrange=False)

quant_lib.define(
    "choose_qparams_symmetric(Tensor input, int quant_min, int quant_max, "
    "float eps, ScalarType dtype) -> (Tensor, Tensor)")

@impl(quant_lib, "choose_qparams_symmetric", "CompositeExplicitAutograd")
def choose_qparams_symmetric(
    input: torch.Tensor,
    qmin: int,
    qmax: int,
    eps: float,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Given an input Tensor, derive the per tensor affine quantization parameter
    (scale and zero_point) for target quantized Tensor from the Tensor

    Args:
       input (torch.Tensor): floating point input Tensor
       quant_min (int): minimum quantized value for target quantized Tensor
       quant_max (int): maximum quantized value for target quantized Tensor
       dtype (torch.dtype): dtype for target quantized Tensor

    Returns:
       scale (float): quantization parameter for the target quantized Tensor
       zero_point (int): quantization parameter for the target quantized Tensor
    """
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert dtype == torch.int8 or dtype == torch.uint8 or dtype == torch.int32, \
        f"Expecting target dtype to be int8 uint8 or int32, but got: {dtype}"
    validate_qmin_qmax(qmin, qmax)

    min_val, max_val = torch.aminmax(input)
    return determine_qparams(
        min_val,
        max_val,
        qmin,
        qmax,
        dtype,
        torch.Tensor([eps]),
        has_customized_qrange=False,
        qscheme=torch.per_tensor_symmetric
    )

@impl(quant_lib, "choose_qparams", "Meta")
def choose_qparams_meta(
    input: torch.Tensor,
    quant_min: int,
    quant_max: int,
    eps: float,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    assert quant_min < quant_max, f"Expecting quant_min to be smaller than quant_max but received min: \
        {quant_min} max: {quant_max}"
    return torch.empty(1, dtype=torch.double, device=input.device), torch.empty(1, dtype=torch.int64, device=input.device)

@impl(quant_lib, "choose_qparams_symmetric", "Meta")
def choose_qparams_symmetric_meta(
    input: torch.Tensor,
    quant_min: int,
    quant_max: int,
    eps: float,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty(1, dtype=torch.double, device=input.device), torch.empty(1, dtype=torch.int64, device=input.device)
