#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

namespace at::native {

namespace {

struct InputMeta {
  void* data_ptr;
  int64_t slice_size;

  InputMeta(const Tensor& t, int64_t dim, int64_t inner_size)
    : data_ptr(t.data_ptr())
    , slice_size(t.sizes()[dim] * inner_size) {}
};

template <typename scalar_t>
static inline void copy_stub(scalar_t* result, scalar_t* self, int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(self + d);
    data_vec.store(result + d);
  }
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (; d < size; ++d) {
    result[d] = self[d];
  }
}

template <typename scalar_t>
void cat_serial_kernel_impl(
    scalar_t* result_data,
    const std::vector<InputMeta>& inputs,
    int64_t outer_size) {

  int64_t ninputs = static_cast<int64_t>(inputs.size());

  scalar_t* result_ptr = result_data;
  for (const auto i : c10::irange(outer_size)) {
    for (const auto j : c10::irange(ninputs)) {
      int64_t input_slice_size = inputs[j].slice_size;
      scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr) + i * input_slice_size;
      copy_stub(result_ptr, input_ptr, input_slice_size);
      result_ptr += input_slice_size;
    }
  }
}

template <typename scalar_t>
void cat_parallel_firstdim_kernel_impl(
    scalar_t* result_data,
    const std::vector<InputMeta>& inputs,
    int64_t dim_size,
    int64_t inner_size) {

  int64_t ninputs = static_cast<int64_t>(inputs.size());
  int64_t input_dim_size = dim_size / ninputs;

  // parallel on dim_size (dim_size == ninputs * input_dim_size).
  //
  // note that prallel on ninputs may not have enough parallelism (e.g. inputs == 2), also
  // parallel on input_dim_size would trigger multiple omp sessions, which has additional overhead.
  int64_t grain_size = internal::GRAIN_SIZE / inner_size;
  at::parallel_for(0, dim_size, grain_size, [&](int64_t begin, int64_t end) {
    int64_t n{0}, i{0};
    data_index_init(begin, n, ninputs, i, input_dim_size);
    for (const auto ii : c10::irange(begin, end)) {
      scalar_t* result_ptr = result_data + ii * inner_size;
      scalar_t* input_ptr = (scalar_t*)(inputs[n].data_ptr) + i * inner_size;
      copy_stub(result_ptr, input_ptr, inner_size);
      data_index_step(n, ninputs, i, input_dim_size);
    }
  });
}

template <typename scalar_t>
void cat_parallel_non_firstdim_kernel_impl(
    scalar_t* result_data,
    const std::vector<InputMeta>& inputs,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

  int64_t ninputs = static_cast<int64_t>(inputs.size());

  // parallel on outer_size.
  int64_t result_slice_size = dim_size * inner_size;
  int64_t grain_size = internal::GRAIN_SIZE / result_slice_size;
  at::parallel_for(0, outer_size, grain_size, [&](int64_t begin, int64_t end) {
    scalar_t* result_ptr = result_data + begin * result_slice_size;
    for (const auto i : c10::irange(begin, end)) {
      for (const auto j : c10::irange(ninputs)) {
        int64_t input_slice_size = inputs[j].slice_size;
        scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr) + i * input_slice_size;
        copy_stub(result_ptr, input_ptr, input_slice_size);
        result_ptr += input_slice_size;
      }
    }
  });
}

template <typename scalar_t, bool parallel>
void cat_kernel_dispatch(const Tensor& result, const MaterializedITensorListRef& tensors, int64_t dim) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dim >= 0 && dim < result.dim(), "dim out of range in cat_kernel_impl");

  // normalize self and result shape as:
  //   input: [outer_size, input_dim_size, inner_size]
  //   result: [outer_size, dim_size, inner_size]
  int64_t inner_size = result.strides()[dim];
  int64_t dim_size = result.sizes()[dim];
  int64_t outer_size = result.numel() / (dim_size * inner_size);

  // construct meta struct for input list, so that we can skip
  // `.data_ptr()` and `.size()` callings in performance critical loops.
  int64_t ninputs = static_cast<int64_t>(tensors.size());
  std::vector<InputMeta> inputs;
  inputs.reserve(ninputs);
  for (const Tensor& tensor : tensors) {
    inputs.emplace_back(tensor, dim, result.strides()[dim]);
  }
  scalar_t* result_data = result.data_ptr<scalar_t>();

  if (parallel) {
    if (outer_size == 1) {
      cat_parallel_firstdim_kernel_impl(result_data, inputs, dim_size, inner_size);
    } else {
      cat_parallel_non_firstdim_kernel_impl(result_data, inputs, outer_size, dim_size, inner_size);
    }
  } else {
    cat_serial_kernel_impl(result_data, inputs, outer_size);
  }
}

void cat_serial_kernel(const Tensor& result, const MaterializedITensorListRef& tensors, int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, result.scalar_type(), "cat_serial_kernel", [&]() {
    cat_kernel_dispatch<scalar_t, /*parallel*/false>(result, tensors, dim);
  });
}

void cat_parallel_kernel(const Tensor& result, const MaterializedITensorListRef& tensors, int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, result.scalar_type(), "cat_parallel_kernel", [&]() {
    cat_kernel_dispatch<scalar_t, /*parallel*/true>(result, tensors, dim);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(cat_serial_stub, &cat_serial_kernel);
REGISTER_DISPATCH(cat_parallel_stub, &cat_parallel_kernel);

} // at::native
