#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/BucketizationUtils.h>
#include <ATen/native/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_torch_cuda_cu_linker_symbol_op_native.h>
#include <ATen/ops/bucketize_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/searchsorted_native.h>
#endif

namespace at {
namespace native {

// Implement a numpy like searchsorted and a TF like bucketize function running on cuda
// See details in ATen/nativate/Bucketization.cpp

namespace {

template<typename input_t>
__device__ int64_t lower_bound(const input_t *data_ss, int64_t start, int64_t end, const input_t val, const int64_t *data_sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
  // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_sort ? data_ss[orig_start + data_sort[mid]] : data_ss[mid];
    if (!(mid_val >= val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t>
__device__ int64_t upper_bound(const input_t *data_ss, int64_t start, int64_t end, const input_t val, const int64_t *data_sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
  // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_sort ? data_ss[orig_start + data_sort[mid]] : data_ss[mid];
    if (!(mid_val > val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t, typename output_t>
__global__ void searchsorted_cuda_kernel(
  output_t *data_out,
  const input_t *data_in,
  const input_t *data_bd,
  const int64_t *data_sort,
  int64_t idim_in,
  int64_t idim_bd,
  int64_t numel_in,
  bool right,
  bool is_1d_boundaries) {

  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < numel_in; tid += blockDim.x * gridDim.x) {
    // If boundaries tensor is 1d, we always search the entire boundary tensor
    int64_t start_bd = is_1d_boundaries ? 0 : tid / idim_in * idim_bd;
    int64_t end_bd = start_bd + idim_bd;

    int64_t pos = !right ?
      lower_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid], data_sort) - start_bd :
      upper_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid], data_sort) - start_bd;

    // type conversion might happen here
    data_out[tid] = pos;
  }
}

template<typename input_t, typename output_t>
void searchsorted_cuda_contiguous(Tensor& result, const Tensor& input, const Tensor& boundaries, const bool& right, const Tensor& sorter) {
  int64_t numel_in = input.numel();
  bool is_scalar_input = input.dim() == 0 && numel_in == 1;
  // inner most dim size of input and boundaries
  int64_t idim_in = is_scalar_input ? 1 : input.sizes().back();
  int64_t idim_bd = boundaries.sizes().back();

  const input_t *data_in = input.data_ptr<input_t>();
  const input_t *data_bd = boundaries.data_ptr<input_t>();
  const int64_t *data_sort = sorter.defined() ? sorter.data_ptr<int64_t>() : nullptr;
  output_t *data_out = result.data_ptr<output_t>();

  int64_t maxThread = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t maxGrid = 1024;
  dim3 block = dim3(std::min(maxThread, numel_in));
  dim3 grid  = dim3(std::min(maxGrid, ceil_div<int64_t>(numel_in, block.x)));
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  searchsorted_cuda_kernel<<<grid, block, 0, stream>>>(
    data_out, data_in, data_bd, data_sort, idim_in, idim_bd, numel_in, right, boundaries.dim() == 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch(
    Tensor& result,
    const Tensor& input,
    const Tensor& boundaries,
    bool out_int32,
    bool right,
    const Tensor& sorter) {
  if (!out_int32) {
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "searchsorted_out_cuda", [&] {
      searchsorted_cuda_contiguous<scalar_t, int64_t>(result, input, boundaries, right, sorter);
    });
  }
  else {
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "searchsorted_out_cuda", [&] {
      searchsorted_cuda_contiguous<scalar_t, int>(result, input, boundaries, right, sorter);
    });
  }
}

}

Tensor& searchsorted_out_cuda(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<Tensor>& sorter_opt,
    Tensor& result) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> sorter_maybe_owned = at::borrow_from_optional_tensor(sorter_opt);
  const Tensor& sorter = *sorter_maybe_owned;
  searchsorted_pre_check(sorted_sequence, self, result, out_int32, right, side_opt, sorter);
  resize_output(result, self.sizes());

  // we have two inputs to set right, pre_check checks that they aren't set to opposites
  bool is_right = (side_opt && *side_opt == "right") || right;
  if (self.numel() == 0) {
    return result;
  }

  // for non-contiguous result tensors, we write the output to a contiguous copy so we can later copy back, maintaing the original result tensor
  Tensor out = result;
  if (!result.is_contiguous()) {
    out = result.contiguous();
  }
  if (sorted_sequence.is_contiguous() && self.is_contiguous() && sorted_sequence.dtype() == self.dtype() && sorter.is_contiguous()) {
   dispatch(out, self, sorted_sequence, out_int32, is_right, sorter);
  }
  else {
    Tensor trimmed_input;
    Tensor trimmed_boundaries;
    Tensor trimmed_sorter;
    searchsorted_maybe_trim_input_tensors(trimmed_input, trimmed_boundaries, trimmed_sorter, self, sorted_sequence, sorter);
    const Tensor& final_input = trimmed_input.defined() ? trimmed_input : self;
    const Tensor& final_boundaries = trimmed_boundaries.defined() ? trimmed_boundaries : sorted_sequence;
    const Tensor& final_sorter = trimmed_sorter.defined() ? trimmed_sorter : sorter;
    dispatch(out, final_input, final_boundaries, out_int32, is_right, final_sorter);
  }

  // if result is non-contiguous, we wrote the answer to a copied version, so we copy back to the original result tensor
  if (!result.is_contiguous()) {
    result.copy_(out);
  }
  return result;
}

Tensor searchsorted_cuda(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<Tensor>& sorter) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::native::searchsorted_out_cuda(sorted_sequence, self, out_int32, right, side_opt, sorter, result);
  return result;
}

// See [Note about _torch_cuda_cu_linker_symbol_op and torch_cuda_cu] in native_functions.yaml
Tensor _torch_cuda_cu_linker_symbol_op_cuda(const Tensor& self) {
  return self;
}

Tensor searchsorted_cuda(
    const Tensor& sorted_sequence,
    const Scalar& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<Tensor>& sorter) {
  const Tensor& scalar_tensor = searchsorted_scalar_tensor(self, sorted_sequence.device());
  return searchsorted_cuda(sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter);
}

Tensor& bucketize_out_cuda(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right, Tensor& result) {
  TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  at::native::searchsorted_out_cuda(boundaries, self, out_int32, right, nullopt, nullopt, result);
  return result;
}

Tensor bucketize_cuda(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::native::bucketize_out_cuda(self, boundaries, out_int32, right, result);
  return result;
}

Tensor bucketize_cuda(const Scalar& self, const Tensor& boundaries, bool out_int32, bool right) {
  return bucketize_cuda(searchsorted_scalar_tensor(self, boundaries.device()), boundaries, out_int32, right);
}

}} // namespace at::native
