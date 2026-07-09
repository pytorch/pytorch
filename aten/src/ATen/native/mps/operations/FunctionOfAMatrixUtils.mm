#include <ATen/native/FunctionOfAMatrixUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/util/irange.h>

#include <array>
#include <cstdlib>
#include <limits>

namespace at::native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/FunctionOfAMatrixUtils_metallib.h>
#endif

// Largest byte offset the kernel dereferences for operand `arg`, including the
// manually walked summation dim. That dim is collapsed out of the iterator (a
// size-1 broadcast), so can_use_32bit_indexing() does not account for its
// i*stride reach; a summation stride > INT32_MAX must still force 64-bit math.
static int64_t clc_max_byte_offset(const TensorIteratorBase& iter,
                                   int arg,
                                   int64_t sum_stride,
                                   int64_t num_summations) {
  int64_t off = 0;
  for (const auto d : c10::irange(iter.ndim())) {
    off += (iter.shape()[d] - 1) * std::abs(iter.strides(arg)[d]);
  }
  return off + (num_summations - 1) * std::abs(sum_stride) * iter.element_size(arg);
}

static void _compute_linear_combination_mps_kernel(TensorIterator& iter,
                                                   int64_t in_stride,
                                                   int64_t coeff_stride,
                                                   int64_t num_summations) {
  using namespace mps;
  if (iter.numel() == 0) {
    return;
  }
  const auto dtype = iter.dtype();
  TORCH_CHECK_TYPE(supportedFloatingType(dtype) || dtype == kComplexFloat,
                   "_compute_linear_combination: unsupported dtype ",
                   dtype,
                   " on MPS, expected float32, float16, bfloat16 or complex64");

  // uint thread ids cap one dispatch at UINT32_MAX threads; split so numel (and
  // every operand's base offset) fits 32 bits.
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _compute_linear_combination_mps_kernel(sub_iter, in_stride, coeff_stride, num_summations);
    }
    return;
  }

  // Base offsets fit int32 now; widen to 64-bit only when the summation reach
  // pushes an input/coeff operand past it (out has no summation reach).
  const int64_t kMax = std::numeric_limits<int32_t>::max();
  const bool use32 = clc_max_byte_offset(iter, 1, in_stride, num_summations) <= kMax &&
      clc_max_byte_offset(iter, 2, coeff_stride, num_summations) <= kMax;

  const auto type_str = scalarToMetalTypeString(dtype);
  const auto ndim = static_cast<uint32_t>(iter.ndim());
  auto pso = lib.getPipelineStateForFunc("compute_linear_combination_" + type_str + std::string(mtlIdxSuffix(use32)));
  dispatch_sync_with_rethrow(getCurrentMPSStream()->queue(), ^() {
    auto computeEncoder = getCurrentMPSStream()->commandEncoder();
    [computeEncoder setComputePipelineState:pso];
    bind_iter_tensors(computeEncoder, iter);
    mtlDispatchByIndexWidth<int32_t, int64_t>(use32, [&](auto idx_tag) {
      using idx_t = typename decltype(idx_tag)::type;
      std::array<idx_t, 4> params{idx_t(in_stride), idx_t(coeff_stride), idx_t(num_summations), idx_t(ndim)};
      mtl_setArgs<3>(computeEncoder, iter.shape(), iter.strides(0), iter.strides(1), iter.strides(2), params);
    });
    mtl_dispatch1DJob(computeEncoder, pso, static_cast<uint32_t>(iter.numel()));
  });
}

REGISTER_MPS_DISPATCH(_compute_linear_combination_stub, &_compute_linear_combination_mps_kernel)

} // namespace at::native
