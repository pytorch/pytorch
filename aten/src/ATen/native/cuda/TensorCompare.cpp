#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorCompare.h>

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

} // anonymous namespace

REGISTER_CUDA_DISPATCH(isin_default_stub, &isin_default_kernel_gpu)

} // namespace at::native
