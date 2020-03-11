#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/BucketizationUtils.h>

/* Implement a TF like searchsorted and a bucketize function running on cpu
 *
 * - torch.searchsorted(sorted_sequence, values, right=False, out_int32=False)
 *   sorted_sequence - N*D tensor containing sorted sequences in last dimension
 *   values          - N*D tensor containing the search values
 *   right           - corresponding to lower bound if False and upper bound if True
 *   out_int32       - the output tensor is int64_t type if False and int(32bit normally) type if True.
 *
 * - torch.bucketlize(values, boundaries, right=False, out_int32=False)
 *   values     - N*D tensor containing the search values
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

template<typename input_t, typename output_t>
void searchsorted_cpu_contiguous(Tensor& result, const Tensor& input, const Tensor& boundaries, const bool& right) {
  bool is_1d_boundaries = boundaries.dim() == 1;
  // inner most dim size of input and boundaries
  int64_t idim_in = input.sizes().back();
  int64_t idim_bd = boundaries.sizes().back();

  const input_t *data_in = input.data_ptr<input_t>();
  const input_t *data_bd = boundaries.data_ptr<input_t>();
  output_t *data_out = result.data_ptr<output_t>();

  int64_t numel_in = input.numel();
  at::parallel_for(0, numel_in, SEARCHSORTED_GRAIN_SIZE, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      // If boundaries tensor is 1d, we always search the entire boundary tensor
      int64_t start_bd = is_1d_boundaries ? 0 : i / idim_in * idim_bd;
      const input_t *data_bd_start = &data_bd[start_bd];

      int64_t pos = !right ?
        std::lower_bound(data_bd_start, data_bd_start + idim_bd, data_in[i]) - data_bd_start :
        std::upper_bound(data_bd_start, data_bd_start + idim_bd, data_in[i]) - data_bd_start;

      // type conversion might happen here
      data_out[i] = pos;
    }
  });
}

}

Tensor& searchsorted_out_cpu(Tensor& result, const Tensor& sorted_sequence, const Tensor& self, bool out_int32, bool right) {
  searchsorted_pre_check(sorted_sequence, self, out_int32);
  if (result.numel() == 0) {
    result.resize_(self.sizes());
  }
  if (self.numel() == 0) {
    return result;
  }

  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "searchsorted_out_cpu", [&] {
    if (out_int32) {
      searchsorted_generic_template(result, self, sorted_sequence, right, searchsorted_cpu_contiguous<scalar_t, int>);
    }
    else {
      searchsorted_generic_template(result, self, sorted_sequence, right, searchsorted_cpu_contiguous<scalar_t, int64_t>);
    }
  });
  return result;
}

Tensor searchsorted_cpu(const Tensor& sorted_sequence, const Tensor& self, bool out_int32, bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  searchsorted_out_cpu(result, sorted_sequence, self, out_int32, right);
  return result;
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

}} // namespace at::native
