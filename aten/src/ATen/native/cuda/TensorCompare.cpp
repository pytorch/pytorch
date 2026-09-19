#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorCompare.h>
#include <cmath>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/isin_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/searchsorted.h>
#include <ATen/ops/where.h>
#endif

namespace at::native {

namespace {

// Composite op implementation for simplicity. This materializes the cross product of elements and test elements,
// so it is not very memory efficient, but it is fast on CUDA.
void isin_default_kernel_gpu(
    const Tensor& elements, const Tensor& test_elements, bool invert, const Tensor& out) {
  std::vector<int64_t> bc_shape(elements.dim(), 1);
  bc_shape.push_back(-1);
  out.copy_(invert ? elements.unsqueeze(-1).ne(test_elements.view(bc_shape)).all(-1)
            : elements.unsqueeze(-1).eq(test_elements.view(bc_shape)).any(-1));
}

// Sorts test_elements, then binary-searches each element into it and checks
// the value it lands on for an exact match.
void isin_sorting(
    const Tensor& elements,
    const Tensor& test_elements,
    bool invert,
    const Tensor& out) {
  // Empty test_elements is routed to the brute-force kernel.
  TORCH_INTERNAL_ASSERT(test_elements.numel() > 0);
  const ScalarType common_dtype = at::result_type(elements, test_elements);
  Tensor elements_flat = elements.to(common_dtype).ravel();
  Tensor test_elements_flat = test_elements.to(common_dtype).ravel();
  // NaNs in the haystack derail searchsorted's binary search, as every
  // comparison against a NaN is false. Overwriting each with an existing
  // non-NaN element is sound, since a NaN never compares equal to anything.
  if (isFloatingType(common_dtype)) {
    Tensor nan_mask = test_elements_flat.isnan();
    Tensor replacement_index = nan_mask.to(kByte).argmin();
    test_elements_flat = at::where(
        nan_mask,
        test_elements_flat.index_select(0, replacement_index),
        test_elements_flat);
  }
  Tensor sorted_test_elements = std::get<0>(test_elements_flat.sort());
  Tensor indices = at::searchsorted(sorted_test_elements, elements_flat);
  indices.clamp_(0, sorted_test_elements.numel() - 1);
  Tensor candidates = sorted_test_elements.index_select(0, indices);
  Tensor mask =
      invert ? candidates.ne(elements_flat) : candidates.eq(elements_flat);
  out.copy_(mask.view_as(out));
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(isin_default_stub, &isin_default_kernel_gpu)

// assume_unique is unused because neither path relies on uniqueness.
TORCH_IMPL_FUNC(isin_Tensor_Tensor_out_cuda)
(const Tensor& elements,
 const Tensor& test_elements,
 bool /*assume_unique*/,
 bool invert,
 const Tensor& out) {
  if (elements.numel() == 0) {
    return;
  }

  // Heuristic taken from numpy's implementation. Kept separate from the CPU
  // copy so the two sorting paths can be tuned independently.
  // See
  // https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/arraysetops.py#L575
  if (test_elements.numel() <
      static_cast<int64_t>(
          10.0f * std::pow(static_cast<double>(elements.numel()), 0.145))) {
    isin_default_stub(kCUDA, elements, test_elements, invert, out);
  } else {
    isin_sorting(elements, test_elements, invert, out);
  }
}

} // namespace at::native
