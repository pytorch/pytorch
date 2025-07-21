#pragma once
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#if AT_KLEIDIAI_ENABLED()

namespace at::native::kleidiai {

/**
 * @brief Rearranges the quantized weight to support kleidiai inference
 * @param bl Groupsize for quantization should be multiple of 32
 */
void kai_pack_int4_rhs(
    const Tensor& weight_packed,
    const Tensor& weight,
    const Tensor& scales,
    const std::optional<Tensor>& bias,
    const int64_t n,
    const int64_t k,
    const int64_t bl);

/**
 * @brief Outputs the buffer size for the packed weights
 * @param bl Groupsize for quantization. 32 for groupwise , 0 for channelwise
 */
size_t kai_pack_rhs_int4_size(
    const int64_t n,
    const int64_t k,
    const int64_t bl);

/**
 * @brief Run 2 operations ( Input quantize and pack -> 4 bit Matmul )
 */
void kai_quant_pack_lhs_int4_mm(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const int64_t bl);
} // namespace at::native::kleidiai
#endif
