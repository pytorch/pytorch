#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/BucketizationUtils.h>

/* Implement a TF like searchsorted and a bucketize function running on cpu
 *
 * - torch.searchsorted(sorted_sequence, values, right=False, out_int32=False)
 *   sorted_sequence - N*D or 1D (apply to all values) tensor containing sorted sequences in last dimension
 *   values          - N*D tensor or a Scalar (when sorted_sequence is 1D) containing the search values
 *   right           - corresponding to lower bound if False and upper bound if True
 *   out_int32       - the output tensor is int64_t type if False and int(32bit normally) type if True.
 *
 * - torch.bucketize(values, boundaries, right=False, out_int32=False)
 *   values     - N*D tensor or a Scalar containing the search value
 *   boundaries - 1D tensor containing a sorted sequences
 *   right      - corresponding to lower bound if False and upper bound if True
 *   out_int32  - the output tensor is int64_t type if False and int(32bit normally) type if True.
 *
 * - Restrictions are defined in searchsorted_pre_check()
 */

namespace at {
namespace native {

namespace {

// minimal size for searchsorted_cpu_contiguous to run parallel (multithread)
constexpr int64_t SEARCHSORTED_GRAIN_SIZE = 200;

// customized lower_bound func to ensure the low bound of 'nan', 'inf' etc. be the end of boundary
// std::lower_bound can not be used here since its customized comparator need strict weak ordering
template<typename input_t>
const input_t* cus_lower_bound(const input_t* start, const input_t* end, input_t val) {
  while (start < end) {
    const input_t* mid = start + ((end - start) >> 1);
    if (!(*mid >= val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t, typename output_t>
void searchsorted_cpu_contiguous(Tensor& result, const Tensor& input, const Tensor& boundaries, const bool& right) {
  int64_t numel_in = input.numel();
  bool is_scalar_input = input.dim() == 0 && numel_in == 1;
  // inner most dim size of input and boundaries
  int64_t idim_in = is_scalar_input ? 1 : input.sizes().back();
  int64_t idim_bd = boundaries.sizes().back();

  const input_t *data_in = input.data_ptr<input_t>();
  const input_t *data_bd = boundaries.data_ptr<input_t>();
  output_t *data_out = result.data_ptr<output_t>();

  bool is_1d_boundaries = boundaries.dim() == 1;
  at::parallel_for(0, numel_in, SEARCHSORTED_GRAIN_SIZE, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      // If boundaries tensor is 1d, we always search the entire boundary tensor
      int64_t start_bd = is_1d_boundaries ? 0 : i / idim_in * idim_bd;
      const input_t *data_bd_start = &data_bd[start_bd];

      int64_t pos = !right ?
        cus_lower_bound(data_bd_start, data_bd_start + idim_bd, data_in[i]) - data_bd_start :
        std::upper_bound(data_bd_start, data_bd_start + idim_bd, data_in[i]) - data_bd_start;

      // type conversion might happen here
      data_out[i] = pos;
    }
  });
}

void dispatch(Tensor& result, const Tensor& input, const Tensor& boundaries, bool out_int32, bool right) {
  if (!out_int32) {
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "searchsorted_out_cpu", [&] {
      searchsorted_cpu_contiguous<scalar_t, int64_t>(result, input, boundaries, right);
    });
  }
  else {
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "searchsorted_out_cpu", [&] {
      searchsorted_cpu_contiguous<scalar_t, int>(result, input, boundaries, right);
    });
  }
}

}

Tensor& searchsorted_out_cpu(Tensor& result, const Tensor& sorted_sequence, const Tensor& self, bool out_int32, bool right) {
  searchsorted_pre_check(sorted_sequence, self, result, out_int32);
  if (result.numel() == 0) {
    result.resize_(self.sizes());
  }
  if (self.numel() == 0) {
    return result;
  }
  if (sorted_sequence.is_contiguous() && self.is_contiguous() && sorted_sequence.dtype() == self.dtype()) {
    dispatch(result, self, sorted_sequence, out_int32, right);
    return result;
  }

  Tensor trimmed_input;
  Tensor trimmed_boundaries;
  searchsorted_maybe_trim_input_tensors(trimmed_input, trimmed_boundaries, self, sorted_sequence);
  const Tensor& final_input = trimmed_input.defined() ? trimmed_input : self;
  const Tensor& final_boundaries = trimmed_boundaries.defined() ? trimmed_boundaries : sorted_sequence;
  dispatch(result, final_input, final_boundaries, out_int32, right);
  return result;
}

Tensor searchsorted_cpu(const Tensor& sorted_sequence, const Tensor& self, bool out_int32, bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  searchsorted_out_cpu(result, sorted_sequence, self, out_int32, right);
  return result;
}

Tensor searchsorted_cpu(const Tensor& sorted_sequence, const Scalar& self, bool out_int32, bool right) {
  return searchsorted_cpu(sorted_sequence, searchsorted_scalar_tensor(self, sorted_sequence.device()), out_int32, right);
}

Tensor& bucketize_out_cpu(Tensor& result, const Tensor& self, const Tensor& boundaries, bool out_int32, bool right) {
  TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  searchsorted_out_cpu(result, boundaries, self, out_int32, right);
  return result;
}

Tensor bucketize_cpu(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  bucketize_out_cpu(result, self, boundaries, out_int32, right);
  return result;
}

Tensor bucketize_cpu(const Scalar& self, const Tensor& boundaries, bool out_int32, bool right) {
  return bucketize_cpu(searchsorted_scalar_tensor(self, boundaries.device()), boundaries, out_int32, right);
}

}} // namespace at::native
