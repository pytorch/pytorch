// elementwise_add.cu
// CUDA device functions for elementwise tensor addition
//
// This file contains device functions that can be compiled to LLVM bitcode
// (.bc) and linked with Triton kernels via extern_libs.
//
// Compile to bitcode:
//   clang++ -x cuda --cuda-device-only -emit-llvm -c elementwise_add.cu \
//           -o elementwise_add.bc --cuda-gpu-arch=sm_80 -O3

#include <cuda_fp16.h>

// Mark functions as extern "C" to avoid C++ name mangling
extern "C" {

// =============================================================================
// SCALAR IMPLEMENTATIONS
// These process one element at a time and are used with Triton's
// extern_elementwise mechanism
// =============================================================================

/**
 * Scalar addition for float32 values.
 *
 * @param a First operand
 * @param b Second operand
 * @return Sum of a and b
 */
__device__ float scalar_add_f32(float a, float b) {
  return a + b;
}

/**
 * Scalar addition for float16 (half precision) values.
 *
 * @param a First operand
 * @param b Second operand
 * @return Sum of a and b using native half-precision arithmetic
 */
__device__ __half scalar_add_f16(__half a, __half b) {
  return __hadd(a, b);
}

/**
 * Scalar addition for float64 (double precision) values.
 *
 * @param a First operand
 * @param b Second operand
 * @return Sum of a and b
 */
__device__ double scalar_add_f64(double a, double b) {
  return a + b;
}

// =============================================================================
// VECTORIZED IMPLEMENTATIONS
// These use vector types for better memory throughput
// =============================================================================

/**
 * Vectorized addition for float4 (4 floats packed together).
 * More efficient as it can utilize wider memory transactions.
 *
 * @param a First operand (4 floats packed)
 * @param b Second operand (4 floats packed)
 * @return Elementwise sum of a and b
 */
__device__ float4 vectorized_add_f32x4(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

/**
 * Vectorized addition for half2 (2 half-precision floats packed).
 * Uses native half2 arithmetic for better throughput.
 *
 * @param a First operand (2 halfs packed)
 * @param b Second operand (2 halfs packed)
 * @return Elementwise sum using native half2 operations
 */
__device__ __half2 vectorized_add_f16x2(__half2 a, __half2 b) {
  return __hadd2(a, b);
}

/**
 * Vectorized addition for float2 (2 floats packed together).
 *
 * @param a First operand (2 floats packed)
 * @param b Second operand (2 floats packed)
 * @return Elementwise sum of a and b
 */
__device__ float2 vectorized_add_f32x2(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

// =============================================================================
// BIT-REPRESENTATION VERSIONS
// These take raw bits and return raw bits, useful for PTX integration
// =============================================================================

/**
 * Scalar addition using bit representation for float32.
 * Takes two uint32_t representing floats, returns their sum as uint32_t.
 *
 * @param a_bits Bit representation of first float32 operand
 * @param b_bits Bit representation of second float32 operand
 * @return Bit representation of the sum
 */
__device__ unsigned int scalar_add_f32_bits(
    unsigned int a_bits,
    unsigned int b_bits) {
  float a = __uint_as_float(a_bits);
  float b = __uint_as_float(b_bits);
  return __float_as_uint(a + b);
}

/**
 * Scalar addition using bit representation for float16.
 * Takes two uint16_t representing halfs, returns their sum as uint16_t.
 *
 * @param a_bits Bit representation of first half operand
 * @param b_bits Bit representation of second half operand
 * @return Bit representation of the sum
 */
__device__ unsigned short scalar_add_f16_bits(
    unsigned short a_bits,
    unsigned short b_bits) {
  __half a = *reinterpret_cast<__half*>(&a_bits);
  __half b = *reinterpret_cast<__half*>(&b_bits);
  __half result = __hadd(a, b);
  return *reinterpret_cast<unsigned short*>(&result);
}

// =============================================================================
// FUSED MULTIPLY-ADD VARIANTS
// Bonus operations commonly needed in neural network computations
// =============================================================================

/**
 * Fused multiply-add for float32: a * b + c
 *
 * @param a First multiplicand
 * @param b Second multiplicand
 * @param c Addend
 * @return Result of a * b + c computed with single rounding
 */
__device__ float scalar_fma_f32(float a, float b, float c) {
  return fmaf(a, b, c);
}

/**
 * Vectorized fused multiply-add for float4.
 *
 * @param a First multiplicand (4 floats packed)
 * @param b Second multiplicand (4 floats packed)
 * @param c Addend (4 floats packed)
 * @return Elementwise fma results
 */
__device__ float4 vectorized_fma_f32x4(float4 a, float4 b, float4 c) {
  return make_float4(
      fmaf(a.x, b.x, c.x),
      fmaf(a.y, b.y, c.y),
      fmaf(a.z, b.z, c.z),
      fmaf(a.w, b.w, c.w));
}

} // extern "C"
