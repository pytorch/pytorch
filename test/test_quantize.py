import torch
import unittest
import math
from typing import Tuple

class TestQuantizeOp(unittest.TestCase):
    
    def test_basic_fp8_quantization(self):
        input_tensor = torch.randn(128, dtype=torch.float32)
        output, scale = torch.quantize_mx(
            input_tensor,
            block_size=32,
            dtype=torch.float8_e4m3fn,
            scale_calculation_mode=0
        )
        
        self.assertEqual(output.dtype, torch.float8_e4m3fn)
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertEqual(scale.shape[0], 128//32)

        golden_output, golden_scale = quantize_python(input_tensor,
            block_size=32,
            dtype=torch.float8_e4m3fn,
            scale_calculation_mode=0
        )

        # print(output)
        # print(golden_output)
        self.assertEqual(torch.equal(output, golden_output), True)
        self.assertEqual(torch.equal(scale, golden_scale), True)
    
    def test_mxfp8_quantization(self):
        input_tensor = torch.randn(1024, 1024 , dtype=torch.float32)
        output, scale = torch.quantize_mx(
            input_tensor,
            block_size=32,
            dtype=torch.float8_e4m3fn,
            scale_calculation_mode=0
        )
        
        self.assertEqual(output.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape[0], 1024)

        golden_output, golden_scale = quantize_python(
            input_tensor,
            block_size=32,
            dtype=torch.float8_e4m3fn,
            scale_calculation_mode=0
        )

        # print(output)
        # print(golden_output)
        self.assertEqual(torch.equal(output, golden_output), True)
        self.assertEqual(torch.equal(scale, golden_scale), True)
    
    def test_per_tensor_quantization(self):
        input_tensor = torch.randn(64, 64, dtype=torch.float32)
        output, scale = torch.quantize_mx(
            input_tensor,
            block_size=64*64,
            dtype=torch.float8_e4m3fn,
            scale_calculation_mode=0
        )
        
        self.assertEqual(scale.shape[0], 64)

        golden_output, golden_scale = quantize_python(input_tensor,
            block_size=64*64,
            dtype=torch.float8_e4m3fn,
            scale_calculation_mode=0
        )

        # print(output)
        # print(golden_output)
        self.assertEqual(torch.equal(output, golden_output), True)
        self.assertEqual(torch.equal(scale, golden_scale), True)
    
    def test_torch_compile(self):

        block_size = 32
        @torch.compile
        def quantize_model(x):
            output, scale = torch.quantize_mx(
                x, 
                block_size=block_size,
                dtype=torch.float8_e4m3fn,
                scale_calculation_mode=0
            )
            return output, scale
        
        x = torch.randn(1024, 1024)
        output, scale = quantize_model(x)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(scale.shape[0], 1024)

        golden_output, golden_scale = quantize_python(
            x,
            block_size=block_size,
            dtype=torch.float8_e4m3fn,
            scale_calculation_mode=0
        )

        # print(output)
        # print(golden_output)
        self.assertEqual(torch.equal(output, golden_output), True)
        self.assertEqual(torch.equal(scale, golden_scale), True)

E8M0_EXPONENT_BIAS = 127
E8M0_EXPONENT_NAN_VAL = 255
MBITS_F32 = 23
F32_MIN_NORMAL = 1.17549435e-38
F32_EXP_BIAS = 127

def get_fp8_max(dtype: torch.dtype) -> float:
    """Get maximum representable value for FP8 dtype."""
    if dtype == torch.float8_e4m3fn:
        return 448.0
    elif dtype == torch.float8_e5m2:
        return 57344.0
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def get_fp8_max_pow2(dtype: torch.dtype) -> int:
    """Get maximum power of 2 for FP8 dtype."""
    if dtype == torch.float8_e4m3fn:
        return 8  # 2^8 = 256
    elif dtype == torch.float8_e5m2:
        return 15  # 2^15 = 32768
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def quantize_mxfp8_python(
    input: torch.Tensor,
    block_size: int,
    dtype: torch.dtype,
    scale_calculation_mode: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure Python implementation of MXFP8 quantization.
    
    Args:
        input: 2D tensor [rows, cols]
        block_size: Block size for quantization
        dtype: Target FP8 dtype
        scale_calculation_mode: 0=floor, 1=round_even
    
    Returns:
        output: Quantized tensor [rows, cols]
        scale: Scale tensor [rows, num_blocks] in uint8 E8M0 format
    """
    assert input.dim() == 2, "Input must be 2D"
    
    rows, cols = input.shape
    num_blocks_per_row = (cols + block_size - 1) // block_size
    
    # Create output tensors
    output = torch.empty_like(input, dtype=dtype)
    scale = torch.empty((rows, num_blocks_per_row), dtype=torch.uint8, device=input.device)
    
    fp8_max = get_fp8_max(dtype)
    target_max_pow2 = get_fp8_max_pow2(dtype)
    
    # Process each row
    for row in range(rows):
        # Process each block in the row
        for block_idx in range(num_blocks_per_row):
            col_start = block_idx * block_size
            col_end = min(col_start + block_size, cols)
            
            # Extract block
            block = input[row, col_start:col_end]
            
            # Find max absolute value
            amax = block.abs().max().item()
            # print("golden amaxes are", amax)
            
            if amax == 0 or math.isnan(amax):
                # Handle zero or NaN
                scale_e8m0_biased = E8M0_EXPONENT_NAN_VAL if math.isnan(amax) else 0
                scale_fp32 = F32_MIN_NORMAL
            else:
                # Extract exponent using bit manipulation
                max_abs_int32 = torch.tensor(amax, dtype=torch.float32).view(torch.int32).item()
                extracted_pow2 = ((max_abs_int32 >> MBITS_F32) & 0xFF) - F32_EXP_BIAS
                
                # Calculate E8M0 scale
                if scale_calculation_mode ==0:
                    scale_e8m0_unbiased = extracted_pow2 - target_max_pow2
                elif scale_calculation_mode ==1:
                    mantissa_gt_one = (max_abs_int32 & 0x7FFFFF) > 0
                    extracted_pow2 += mantissa_gt_one
                    scale_e8m0_unbiased = extracted_pow2 - target_max_pow2 + 1

                scale_e8m0_unbiased = max(-E8M0_EXPONENT_BIAS, 
                                          min(scale_e8m0_unbiased, E8M0_EXPONENT_BIAS + 1))
                
                # Create biased E8M0 representation
                scale_e8m0_biased = scale_e8m0_unbiased + E8M0_EXPONENT_BIAS
                
                # Convert E8M0 to float32
                scale_bits = (scale_e8m0_biased << MBITS_F32)
                scale_fp32 = torch.tensor(scale_bits, dtype=torch.int32).view(torch.float32).item()
                scale_fp32 = max(scale_fp32, F32_MIN_NORMAL)
            
            
            
            # Store E8M0 scale
            scale[row, block_idx] = scale_e8m0_biased
            
            # Quantize block
            # quant_scale = 1.0 / scale_fp32 if scale_fp32 > 0 else 1.0
            quantized = block / scale_fp32
            
            # print("golden before round are", quantized)
            # Apply rounding
            # if rounding_mode == 1:  # round_even
            quantized = torch.round(quantized)
            # elif rounding_mode == 0:  # floor
            #     quantized = torch.floor(quantized)
            
            # Clamp to FP8 range
            quantized = torch.clamp(quantized, -fp8_max, fp8_max)
            
            # Store quantized values
            output[row, col_start:col_end] = quantized.to(dtype)
    
    return output, scale


def quantize_python(
    input: torch.Tensor,
    block_size: int = 32,
    dtype: torch.dtype = torch.float8_e4m3fn,
    scale_calculation_mode: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure Python quantization (matches C++ behavior).
    """
    # Handle 1D input
    is_1d = (input.dim() == 1)
    input_2d = input.unsqueeze(0) if is_1d else input
    
    # Validate
    assert input_2d.dim() == 2, "Input must be 2D"
    assert block_size > 0, "Block size must be positive"
    assert scale_calculation_mode in (0, 1), "scale_calculation_mode must be 0 or 1"
    
    # Quantize
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        output_2d, scale_2d = quantize_mxfp8_python(input_2d, block_size, dtype, scale_calculation_mode)
    else:
        raise NotImplementedError("quant_type=1 not implemented in Python reference")
    
    # Reshape back to 1D if needed
    output = output_2d.squeeze(0) if is_1d else output_2d
    scale = scale_2d.squeeze(0) if is_1d else scale_2d
    
    return output, scale


if __name__ == '__main__':
    unittest.main()