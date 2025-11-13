#include <ATen/ATen.h>
#include <ATen/native/Quantize.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <cmath>
#include <tuple>
#include <iostream>

namespace at {
namespace native {

namespace {

// // Helper function for rounding
// template <typename scalar_t>
// inline scalar_t apply_rounding(scalar_t value, int64_t rounding_mode, uint32_t* rand_state = nullptr) {
//   switch (rounding_mode) {
//     case 0: // floor
//       return std::floor(value);
//     case 1: // round to nearest even
//       return std::rint(value);
//     case 2: // stochastic (simplified)
//       if (rand_state) {
//         float frac = value - std::floor(value);
//         // Simple LCG random number generator
//         *rand_state = (*rand_state * 1103515245 + 12345) & 0x7fffffff;
//         float rand_val = static_cast<float>(*rand_state) / 0x7fffffff;
//         return (rand_val < frac) ? std::ceil(value) : std::floor(value);
//       }
//       return std::rint(value);
//     default:
//       return std::rint(value);
//   }
// }

// FP8 quantization parameters
constexpr float FP8_E4M3_MAX = 448.0f;
constexpr float FP8_E5M2_MAX = 57344.0f;
constexpr int F32_EXP_BIAS = 127;
constexpr int EBITS_F32 = 8;
constexpr int MBITS_F32 = 23;
constexpr int E8M0_EXPONENT_BIAS = 127;
constexpr int E8M0_EXPONENT_NAN_VAL = 255;
const float F32_MIN_NORMAL = std::pow(2.0f, -126.0f);

float get_fp8_max(at::ScalarType dtype) {
  if (dtype == at::kFloat8_e4m3fn) {
    return FP8_E4M3_MAX;
  } else if (dtype == at::kFloat8_e5m2) {
    return FP8_E5M2_MAX;
  }
  return FP8_E4M3_MAX;
}

// Get target max power of 2 for FP8 dtypes
inline int get_fp8_max_pow2(at::ScalarType dtype) {
  if (dtype == at::kFloat8_e4m3fn) {
    // Max value for e4m3fn is 448 = 1.75 * 2^8
    // But the largest power of 2 representable is 2^8 = 256
    return 8;
  } else if (dtype == at::kFloat8_e5m2) {
    // Max value for e5m2 is 57344 = 1.75 * 2^15
    // But the largest power of 2 representable is 2^15 = 32768
    return 15;
  }
  return 0;
}

// template <typename input_t, typename output_t>
// void quantize_fp8_impl(
//     const input_t* input_data,
//     output_t* output_data,
//     uint8_t* scale_data,
//     int64_t dim0_size,
//     int64_t dim1_size,
//     int64_t block_size,
//     float fp8_max,
//     int64_t rounding_mode) {
  
//   int64_t num_blocks = (dim1_size + block_size - 1) / block_size;
  
//   at::parallel_for(0, num_blocks, 0, [&](int64_t begin, int64_t end) {
//     uint32_t rand_state = begin + 12345; // Simple seed
    
//     for (int64_t block_idx = begin; block_idx < end; ++block_idx) {
//       int64_t start = block_idx * block_size;
//       int64_t block_end = std::min(start + block_size, dim1_size);
      
//       // Find max absolute value in block
//       float amax = 0.0f;
//       for (int64_t i = start; i < block_end; ++i) {
//         float abs_val = std::abs(static_cast<float>(input_data[i]));
//         amax = std::max(amax, abs_val);
//       }
      
//       // Compute scale
//       float scale = (amax == 0.0f) ? 1.0f : (fp8_max / amax);
//       scale_data[block_idx] = 1.0f / scale; // Store inverse scale for dequant
      
//       // Quantize block
//       for (int64_t i = start; i < block_end; ++i) {
//         float val = static_cast<float>(input_data[i]) * scale;
//         val = std::max(-fp8_max, std::min(val, fp8_max));
//         output_data[i] = static_cast<output_t>(val);
//       }
//     }
//   });
// }

template <typename input_t, typename output_t>
void quantize_mxfp8_impl(
    const input_t* input_data,
    output_t* output_data,
    uint8_t* scale_data,
    int64_t dim0_size,
    int64_t dim1_size,
    int64_t block_size,
    float fp8_max,
    int64_t rounding_mode, 
    at::ScalarType fp8_dtype
  ) {
  
  int64_t num_blocks_per_row = (dim1_size + block_size - 1) / block_size;
  int target_max_pow2 = get_fp8_max_pow2(fp8_dtype);

  // Parallel over rows
  at::parallel_for(0, dim0_size, 0, [&](int64_t begin_row, int64_t end_row) {
    for (int64_t row = begin_row; row < end_row; ++row) {
      
      // Process each block in this row
      for (int64_t block_idx = 0; block_idx < num_blocks_per_row; ++block_idx) {
        int64_t col_start = block_idx * block_size;
        int64_t col_end = std::min(col_start + block_size, dim1_size);
        
        // Step 1: Find max absolute value in block
        float amax = 0.0f;
        for (int64_t col = col_start; col < col_end; ++col) {
          int64_t idx = row * dim1_size + col;
          float abs_val = std::abs(static_cast<float>(input_data[idx]));
          amax = std::max(amax, abs_val);
        }
        
        // std::cout<<"the amax is"<<std::setprecision(10)<<amax<<std::endl;
        // Step 2: Extract exponent using int32 bit manipulation
        // Equivalent to: max_abs_int32 = max_abs.view(torch.int32)
        int32_t max_abs_int32;
        std::memcpy(&max_abs_int32, &amax, sizeof(float));
        
        // Step 3: Extract power of 2
        // extracted_pow2 = ((max_abs_int32 >> hp_mbits) & 0b11111111) - hp_exp_bias
        // hp_mbits = 23 (mantissa bits for float32)
        // hp_exp_bias = 127 (exponent bias for float32)
        int32_t extracted_pow2 = ((max_abs_int32 >> MBITS_F32) & 0xFF) - F32_EXP_BIAS;

        // Step 4: Calculate scale exponent (E8M0 format)
        // scale_e8m0_unbiased = extracted_pow2 - target_max_pow2
        int32_t scale_e8m0_unbiased = 0;
        
        
        if (rounding_mode ==0){
          // Floor the scale 
          scale_e8m0_unbiased = extracted_pow2 - target_max_pow2;
        }
        else if(rounding_mode == 1){
          // Ceil the scale
          int32_t mantissa_gt_one = (max_abs_int32 & 0x7FFFFF) > 0 ;
          extracted_pow2 += mantissa_gt_one;
          scale_e8m0_unbiased = extracted_pow2 - target_max_pow2 + 1;
        }else{
          TORCH_CHECK(false, "Unsupported rounding mode for scale calculation");
        }
        
        // Step 5: Clamp to valid E8M0 range
        scale_e8m0_unbiased = std::max(-E8M0_EXPONENT_BIAS, std::min(scale_e8m0_unbiased, E8M0_EXPONENT_BIAS + 1));

        int32_t scale_e8m0_biased_int = scale_e8m0_unbiased + E8M0_EXPONENT_BIAS;
        uint8_t scale_e8m0_biased = static_cast<uint8_t>(scale_e8m0_biased_int);
        
        if (std::isnan(amax)) {
          scale_e8m0_biased = E8M0_EXPONENT_NAN_VAL;
        }

        int32_t scale_bits = static_cast<int32_t>(scale_e8m0_biased) << MBITS_F32;
        float scale_fp32;
        std::memcpy(&scale_fp32, &scale_bits, sizeof(float));

        scale_fp32 = std::max(scale_fp32, F32_MIN_NORMAL);

        // std::cout<<"the scale_fp32 is"<<std::setprecision(10)<<scale_fp32<<std::endl;
        // Store scale
        int64_t scale_idx = row * num_blocks_per_row + block_idx;
        scale_data[scale_idx] = scale_e8m0_biased;
        
        // Quantize this block
        for (int64_t col = col_start; col < col_end; ++col) {
          int64_t idx = row * dim1_size + col;
          float val = static_cast<float>(input_data[idx]) / scale_fp32;
          
          // Apply rounding
          
          val = std::nearbyint(val);
          val = std::max(-fp8_max, std::min(val, fp8_max));
          output_data[idx] = static_cast<output_t>(val);
        }
      }
    }
  });
}

} // anonymous namespace

std::tuple<Tensor, Tensor> quantize_cpu(
    const Tensor& self,
    int64_t block_size,
    at::ScalarType dtype,
    int64_t quant_type,
    int64_t rounding_mode) {
  
  bool is_1d = (self.dim() == 1);
  const Tensor input = is_1d ? self.unsqueeze(0) : self;

  TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
  TORCH_CHECK(block_size > 0, "Block size must be positive");
  TORCH_CHECK(quant_type >= 0 && quant_type <= 1, "quant_type must be 0 (FP8) or 1 (MXFP8)");
  TORCH_CHECK(rounding_mode >= 0 && rounding_mode <= 1, "rounding_mode must be 0, 1");
  
  
  // Get dimensions
  int64_t dim0_size = input.size(0);  // Number of rows
  int64_t dim1_size = input.size(1);  // Number of columns
  
  // Create output with same shape but different dtype
  auto output = at::empty_like(input, input.options().dtype(dtype));
  
  // Create scale tensor: [dim0, num_blocks_per_row]
  int64_t num_blocks_per_row = (dim1_size + block_size - 1) / block_size;
  auto scale = at::empty({dim0_size, num_blocks_per_row}, input.options().dtype(at::kByte));
  
  float fp8_max = get_fp8_max(dtype);
  
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, input.scalar_type(), "quantize_cpu", [&] {
    const scalar_t* input_data = input.data_ptr<scalar_t>();
    uint8_t* scale_data = scale.data_ptr<uint8_t>();
    
    if (dtype == at::kFloat8_e4m3fn) {
      auto* output_data = output.data_ptr<at::Float8_e4m3fn>();
      if (quant_type == 0) {
        quantize_mxfp8_impl(input_data, output_data, scale_data, dim0_size, dim1_size, block_size, fp8_max, rounding_mode, dtype);
      }
        // quantize_fp8_impl(input_data, output_data, scale_data, dim0_size, dim1_size, block_size, fp8_max, rounding_mode);
      // } else {
      //   quantize_mxfp8_impl(input_data, output_data, scale_data, dim0_size, dim1_size, block_size, fp8_max, rounding_mode);
      // }
    } else if (dtype == at::kFloat8_e5m2) {
      auto* output_data = output.data_ptr<at::Float8_e5m2>();
      if (quant_type == 0) {
        quantize_mxfp8_impl(input_data, output_data, scale_data, dim0_size, dim1_size, block_size, fp8_max, rounding_mode, dtype);
      }
        // quantize_fp8_impl(input_data, output_data, scale_data, dim0_size, dim1_size, block_size, fp8_max, rounding_mode);
      // } else {
      //   quantize_mxfp8_impl(input_data, output_data, scale_data, dim0_size, dim1_size, block_size, fp8_max, rounding_mode);
      // }
    } else {
      TORCH_CHECK(false, "Unsupported dtype for quantization");
    }
  });
  
  output = is_1d ? output.squeeze(0) : output;
  scale = is_1d ? scale.squeeze(0) : scale;

  return std::make_tuple(output, scale);
}

// Meta kernel: Only computes shapes, no actual computation
std::tuple<Tensor, Tensor> quantize_meta(
    const Tensor& self,
    int64_t block_size,
    at::ScalarType dtype_opt,
    int64_t quant_type,
    int64_t rounding_mode) {
  
  // Apply default
  at::ScalarType dtype = dtype_opt;
  
  bool is_1d = (self.dim() == 1);
  const Tensor input = is_1d ? self.unsqueeze(0) : self;

  // Validation (same as other kernels)
  TORCH_CHECK(input.dim() >= 1, "Input tensor must have at least 1 dimension");
  TORCH_CHECK(block_size > 0, "Block size must be positive");
  
  // Create output tensors with correct shapes and dtypes
  // but on Meta device (no actual memory allocation)
  auto output = at::empty_like(input, self.options().dtype(dtype));

  int64_t dim0_size = input.size(0);  // Number of rows
  int64_t dim1_size = input.size(1);  // Number of columns

  int64_t num_blocks = (dim1_size + block_size - 1) / block_size;
  auto scale = at::empty({dim0_size, num_blocks}, input.options().dtype(at::kByte));
  
  output = is_1d ? output.squeeze(0) : output;
  scale = is_1d ? scale.squeeze(0) : scale;

  return std::make_tuple(output, scale);
}

} // namespace native
} // namespace at