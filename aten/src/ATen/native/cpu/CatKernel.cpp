#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

namespace at::native {

namespace {

struct InputMeta {
  void* data_ptr;
  int64_t inner_size;

  InputMeta(const Tensor& t, int64_t dim, int64_t inner)
    : data_ptr(t.data_ptr())
    , inner_size(t.sizes()[dim] * inner) {}
};

template <typename scalar_t>
void cat_serial_kernel_impl(const Tensor& result, const MaterializedITensorListRef& tensors, int64_t dim) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dim >= 0 && dim < result.dim(), "dim out of range in cat_serial_kernel_impl");
  int64_t outer = result.numel() / (result.sizes()[dim] * result.strides()[dim]);
  scalar_t* result_data = result.data_ptr<scalar_t>();
  int64_t ninputs = static_cast<int64_t>(tensors.size());
  std::vector<InputMeta> inputs;
  inputs.reserve(ninputs);
  for (const Tensor& tensor : tensors) {
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

void cat_serial_kernel(const Tensor& result, const MaterializedITensorListRef& tensors, int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, result.scalar_type(), "cat_serial_kernel", [&]() {
    cat_serial_kernel_impl<scalar_t>(result, tensors, dim);
  });
}

template <typename scalar_t>
void cat_fast_dim0_kernel_impl(const Tensor& result, const MaterializedITensorListRef& tensors) {
  auto outBytes = result.nbytes();
  char* dataPtr = reinterpret_cast<char*>(result.data_ptr());
  size_t totalBytes = 0;
  for (const Tensor& input : tensors) {
    const auto inputNbytes = input.nbytes();
    TORCH_CHECK(outBytes >= (totalBytes + inputNbytes));
    if (inputNbytes == 0) {
      continue;
    }
    std::memcpy(dataPtr + totalBytes, input.data_ptr(), input.nbytes());
    totalBytes += input.nbytes();
  }
  TORCH_CHECK(outBytes == totalBytes);
}

template <typename scalar_t>
void cat_fast_dim1_kernel_impl(const Tensor& result, const MaterializedITensorListRef& tensors) {
  auto outBytes = result.nbytes();
  char* outputDataPtr = reinterpret_cast<char*>(result.data_ptr());
  size_t sliceBytes = 0;
  size_t offsetInSlice = 0;
  for (const Tensor& input : tensors) {
    sliceBytes += input.nbytes() / input.size(0);
  }
  for (const Tensor& input : tensors) {
    size_t inputBytes = input.nbytes();
    char* inputDataPtr = reinterpret_cast<char*>(input.data_ptr());
    size_t inputSliceBytes = input.nbytes() / input.size(0);
    for (auto s = 0; s < input.size(0); ++s) {
      auto destOffset = sliceBytes * s + offsetInSlice;
      auto srcOffset = inputSliceBytes * s;
      TORCH_CHECK(destOffset + inputSliceBytes <= outBytes);
      TORCH_CHECK(srcOffset + inputSliceBytes <= inputBytes);
      std::memcpy(
          outputDataPtr + destOffset,
          inputDataPtr + srcOffset,
          inputSliceBytes);
    }
    offsetInSlice += inputSliceBytes;
  }
  TORCH_CHECK(offsetInSlice == sliceBytes);
}

void cat_fast_kernel(const Tensor& result, const MaterializedITensorListRef& tensors, int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, result.scalar_type(), "cat_fast_kernel", [&]() {
    if (dim == 0) {
      cat_fast_dim0_kernel_impl<scalar_t>(result, tensors);
    } else if (dim == 1) {
      cat_fast_dim1_kernel_impl<scalar_t>(result, tensors);
    } else {
      TORCH_CHECK(false, "Only dim = 0, 1 is supported for cat_fast now");
    }
  });
}

} // anonymous namespace

REGISTER_DISPATCH(cat_serial_stub, &cat_serial_kernel);
REGISTER_DISPATCH(cat_fast_stub, &cat_fast_kernel);

} // at::native
