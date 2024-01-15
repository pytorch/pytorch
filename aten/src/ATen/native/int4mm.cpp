#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/cpu/int4mm_kernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_convert_weight_to_int4pack_native.h>
#include <ATen/ops/_weight_int4pack_mm_native.h>
#include <ATen/ops/empty.h>
#endif

namespace at::native {

DEFINE_DISPATCH(weight_to_int4pack_stub);
DEFINE_DISPATCH(int4pack_mm_stub);

Tensor _convert_weight_to_int4pack_cpu(
    const Tensor& in,
    int64_t innerKTiles) {

  TORCH_CHECK(in.dim() == 2,
      "_convert_weight_to_int4pack: expect weight to be 2D tensor.");
  TORCH_CHECK(in.dtype() == at::kInt,
      "_convert_weight_to_int4pack: expect weight to be kInt.");

  auto weight = in.contiguous();
  auto N = weight.size(0);
  auto K = weight.size(1);

  TORCH_CHECK(N % 16 == 0,
      "_convert_weight_to_int4pack: expect N to be dividable by 16");
  TORCH_CHECK(K % 2 == 0,
      "_convert_weight_to_int4pack: expect K to be dividable by 2");

  auto weight_packed = at::empty({N, K / 2}, weight.options().dtype(kByte));
  weight_to_int4pack_stub(kCPU, weight_packed, weight);
  return weight_packed;
}

Tensor _weight_int4pack_mm_cpu(
    const Tensor& A,
    const Tensor& B,
    int64_t qGroupSize,
    const Tensor& qScaleAndZeros) {

  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  TORCH_CHECK(A.dtype() == kBFloat16,
      "_weight_int4pack_mm: expect A to be bfloat16 tensor.");
  TORCH_CHECK(A.is_contiguous(),
      "_weight_int4pack_mm: expect A to be contiguous.");
  TORCH_CHECK(A.dim() == 2,
      "_weight_int4pack_mm: expect A to be 2D tensor.");

  TORCH_CHECK(B.dtype() == kByte,
      "_weight_int4pack_mm: expect B to be uint8 tensor.");
  TORCH_CHECK(B.is_contiguous(),
      "_weight_int4pack_mm: expect B to be contiguous.");
  TORCH_CHECK(B.size(1) == K / 2);

  TORCH_CHECK(
      qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128 ||
      qGroupSize == 256);

  TORCH_CHECK(qScaleAndZeros.dim() == 3);
  TORCH_CHECK(qScaleAndZeros.size(1) == N);
  TORCH_CHECK(qScaleAndZeros.size(2) == 2);

  auto C = at::empty({M, N}, A.options());
  int4pack_mm_stub(kCPU, C, A, B, qGroupSize, qScaleAndZeros);

  return C;
}

} // at::native
