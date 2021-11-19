#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

namespace at { namespace native {

namespace {

struct InputMeta {
  void* data_ptr;
  int64_t inner_size;

  InputMeta(const Tensor& t, int64_t dim, int64_t inner)
    : data_ptr(t.data_ptr())
    , inner_size(t.sizes()[dim] * inner) {}
};

template <typename scalar_t>
void cat_serial_kernel_impl(const Tensor& result, ITensorList tensors, int64_t dim) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dim >= 0 && dim < result.dim(), "dim out of range in cat_serial_kernel_impl");
  int64_t outer = result.numel() / (result.sizes()[dim] * result.strides()[dim]);
  scalar_t* result_data = result.data_ptr<scalar_t>();
  int64_t ninputs = static_cast<int64_t>(tensors.size());
  std::vector<InputMeta> inputs;
  inputs.reserve(ninputs);
  for (auto const &tensor : tensors) {
    inputs.emplace_back(tensor, dim, result.strides()[dim]);
  }

  using Vec = vec::Vectorized<scalar_t>;
  scalar_t* result_ptr = result_data;
  for (const auto i : c10::irange(outer)) {
    for (const auto j : c10::irange(ninputs)) {
      int64_t local_inner = inputs[j].inner_size;
      scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr) + i * local_inner;
      int64_t d = 0;
      for (; d < local_inner - (local_inner % Vec::size()); d += Vec::size()) {
        Vec in_vec = Vec::loadu(input_ptr + d);
        in_vec.store(result_ptr + d);
      }
      #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      for (; d < local_inner; d++) {
        result_ptr[d] = input_ptr[d];
      }
      result_ptr += local_inner;
    }
  }
}

void cat_serial_kernel(const Tensor& result, ITensorList tensors, int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, result.scalar_type(), "cat_serial_kernel", [&]() {
    cat_serial_kernel_impl<scalar_t>(result, tensors, dim);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(cat_serial_stub, &cat_serial_kernel);

}} // at::native
