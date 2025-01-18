#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/BucketizationUtils.h>
#include <ATen/native/Resize.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/bucketize_native.h>
#include <ATen/ops/searchsorted_native.h>
#endif

/* Implement a numpy like searchsorted and a TF like bucketize function running on cpu
 *
 * - torch.searchsorted(sorted_sequence, values, right=False, side=None, out_int32=False, sorter=None)
 *   sorted_sequence - N*D or 1D (apply to all values) tensor containing sorted sequences in last dimension
 *   values          - N*D tensor or a Scalar (when sorted_sequence is 1D) containing the search values
 *   right           - corresponding to lower bound if False and upper bound if True
 *   side            - (preferred to right) corresponding to lower bound if 'left' and upper bound if 'right'
 *   out_int32       - the output tensor is int64_t type if False and int(32bit normally) type if True.
 *   sorter          - if provided, sorted_sequence may not be sorted and the sorted order is given by this tensor
 *
 * - torch.bucketize(values, boundaries, right=False, out_int32=False)
 *   values     - N*D tensor or a Scalar containing the search value
 *   boundaries - 1D tensor containing a sorted sequences
 *   right      - corresponding to lower bound if False and upper bound if True
 *   out_int32  - the output tensor is int64_t type if False and int(32bit normally) type if True.
 *
 * - Restrictions are defined in searchsorted_pre_check()
 */

namespace at::native {

namespace {

// minimal size for searchsorted_cpu_contiguous to run parallel (multithread)
constexpr int64_t SEARCHSORTED_GRAIN_SIZE = 200;

// customized lower_bound func to ensure the low bound of 'nan', 'inf' etc. be the end of boundary
// and we can properly handle a sorter argument
// std::lower_bound can not be used here since its customized comparator need strict weak ordering
// and the customized comparators require both arguments to have the same type, which wouldn't
// happen when comparing val of input_t to an indexer value from sorter of int64
template<typename input_t>
int64_t cus_lower_bound(int64_t start, int64_t end, const input_t val, const input_t* bd, const int64_t* sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
  // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = sort ? bd[sort[mid] + orig_start] : bd[mid];
    if (!(mid_val >= val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

// customized upper_bound func to ensure we can properly handle a sorter argument
// std::upper_bound can not be used here since its customized comparator requires both arguments to have the
// same type, which wouldn't happen when comparing val of input_t to an indexer value from sorter of int64
template<typename input_t>
int64_t cus_upper_bound(int64_t start, int64_t end, const input_t val, const input_t* bd, const int64_t* sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
  // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = sort ? bd[sort[mid] + orig_start] : bd[mid];
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
void searchsorted_cpu_contiguous(Tensor& result, const Tensor& input, const Tensor& boundaries, const bool& right, const Tensor& sorter) {
  int64_t numel_in = input.numel();
  bool is_scalar_input = input.dim() == 0 && numel_in == 1;
  // inner most dim size of input and boundaries
  int64_t idim_in = is_scalar_input ? 1 : input.sizes().back();
  int64_t idim_bd = boundaries.sizes().back();

  const input_t *data_in = input.const_data_ptr<input_t>();
  const input_t *data_bd = boundaries.const_data_ptr<input_t>();
  const int64_t *data_st = sorter.defined() ? sorter.const_data_ptr<int64_t>() : nullptr;
  output_t *data_out = result.data_ptr<output_t>();

  bool is_1d_boundaries = boundaries.dim() == 1;
  at::parallel_for(0, numel_in, SEARCHSORTED_GRAIN_SIZE, [&](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      // If boundaries tensor is 1d, we always search the entire boundary tensor
      int64_t start_bd = is_1d_boundaries ? 0 : i / idim_in * idim_bd;
      int64_t end_bd = start_bd + idim_bd;

      int64_t pos = !right ?
        cus_lower_bound(start_bd, end_bd, data_in[i], data_bd, data_st) - start_bd :
        cus_upper_bound(start_bd, end_bd, data_in[i], data_bd, data_st) - start_bd;

      // type conversion might happen here
      data_out[i] = pos;
    }
  });
}

void dispatch(Tensor& result, const Tensor& input, const Tensor& boundaries, bool out_int32, bool right, const Tensor& sorter) {
  if (!out_int32) {
    AT_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        input.scalar_type(),
        "searchsorted_out_cpu",
        [&] {
          searchsorted_cpu_contiguous<scalar_t, int64_t>(
              result, input, boundaries, right, sorter);
        });
  }
  else {
    AT_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        input.scalar_type(),
        "searchsorted_out_cpu",
        [&] {
          searchsorted_cpu_contiguous<scalar_t, int>(
              result, input, boundaries, right, sorter);
        });
  }
}

}

Tensor& searchsorted_out_cpu(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    const std::optional<std::string_view> side_opt,
    const std::optional<Tensor>& sorter_opt,
    Tensor& result) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> sorter_maybe_owned = at::borrow_from_optional_tensor(sorter_opt);
  const Tensor& sorter = *sorter_maybe_owned;
  searchsorted_pre_check(sorted_sequence, self, result, out_int32, right, side_opt, sorter);
  resize_output(result, self.sizes());

  // we have two inputs to set right, pre_check checks that they aren't set to opposites
  bool is_right = side_opt ? *side_opt == "right" : right;

  if (self.numel() == 0) {
    return result;
  }

  // for non-contiguous result tensors, we write the output to a contiguous copy so we can later copy back, maintaining the original result tensor
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

Tensor& searchsorted_out_cpu(
    const Tensor& sorted_sequence,
    const Scalar& self,
    bool out_int32,
    bool right,
    const std::optional<std::string_view> side_opt,
    const std::optional<Tensor>& sorter_opt,
    Tensor& result) {
  const Tensor& scalar_tensor = searchsorted_scalar_tensor(self, sorted_sequence.device());
  return searchsorted_out_cpu(sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter_opt, result);
}

Tensor searchsorted_cpu(
      const Tensor& sorted_sequence,
      const Tensor& self,
      bool out_int32,
      bool right,
      const std::optional<std::string_view> side_opt,
      const std::optional<Tensor>& sorter_opt) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::native::searchsorted_out_cpu(sorted_sequence, self, out_int32, right, side_opt, sorter_opt, result);
  return result;
}

Tensor searchsorted_cpu(
    const Tensor& sorted_sequence,
    const Scalar& self,
    bool out_int32,
    bool right,
    const std::optional<std::string_view> side_opt,
    const std::optional<Tensor>& sorter_opt) {
  const Tensor& scalar_tensor = searchsorted_scalar_tensor(self, sorted_sequence.device());
  return searchsorted_cpu(sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter_opt);
}

Tensor& bucketize_out_cpu(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right, Tensor& result) {
  TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  at::native::searchsorted_out_cpu(boundaries, self, out_int32, right, std::nullopt, std::nullopt, result);
  return result;
}

Tensor bucketize_cpu(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::native::bucketize_out_cpu(self, boundaries, out_int32, right, result);
  return result;
}

Tensor bucketize_cpu(const Scalar& self, const Tensor& boundaries, bool out_int32, bool right) {
  return bucketize_cpu(searchsorted_scalar_tensor(self, boundaries.device()), boundaries, out_int32, right);
}

} // namespace at::native
