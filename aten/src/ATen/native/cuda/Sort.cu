#include <limits>

#include <ATen/ATen.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/LegacyTHFunctionsCUDA.h>
#include <ATen/core/Array.h>
#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/detail/KernelUtils.h>

namespace {

template<typename scalar_t>
__global__ void sort_gather_kernel(const scalar_t *in, scalar_t *out, const int64_t *index, int nsegments, int nsort) {
  CUDA_KERNEL_LOOP(i, nsegments * nsort) {
    int segment = i / nsort;
    int j = i % nsort;
    int offset = segment * nsort;
    const scalar_t *in_ = in + offset;
    scalar_t *out_ = out + offset;
    const int64_t *index_ = index + offset;
    out_[j] = in_[index_[j]];
  }
}

}

namespace at { namespace native {

bool should_use_th_sort(const Tensor &self, int64_t dim) {
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  int64_t nsort = self.sizes()[dim];
  int64_t threshold;
  if (self.scalar_type() == kLong || self.scalar_type() == kDouble) {
    threshold = 1024;
  } else {
    threshold = 2048;
  }
  return nsort <= threshold;
}

std::vector<int64_t> infer_dense_strides_dim_last(const Tensor & self, int64_t dim);

// We perform a vectorized segmented sort in cub.
// Say we are sorting a (2, 3) tensor. We have in flattened form:
// values       0.4 1.2 5.3 6.2 1.3 2.3
// indices        0   1   2   0   1   2
// segment_id     0   0   0   1   1   1

// First we sort by values, globally:
// values       6.2 5.3 2.3 1.2 1.3 0.4
// indices        0   2   2   1   1   0
// segment_id     1   0   1   0   1   0

// Then we stable sort by segment id:
// values       5.3 1.2 0.4 6.2 2.3 1.3
// indices        2   1   0   0   2   1
// segment_id     0   0   0   1   1   1

// This method can only work if the slice we are sorting (`dim`) is
// innermost, and both values and indices are contiguous. We do this
// by re-arranging the input into this form as needed, which will
// unfortunately allocate memory if the request is not in this form.
// Vectorized sort is slower than iterated sort if the number of
// slices is small (since we're sorting twice, instead of invoking a
// smaller sort `numSlices` times), but the cub sort
// implementation here is a catch-all, so we're not looking for
// efficiency, but instead correctness.
std::tuple<Tensor &,Tensor &> sort_out_stable_cuda(const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
  if (should_use_th_sort(self, dim)) {
    return legacy::cuda::_th_sort_out_stable(self, stable, dim, descending, values, indices);
  }
  // this algorithm is always stable
  TORCH_INTERNAL_ASSERT(stable.has_value(), "sort_out(): c10::optional<bool> for stable has to have value.");
  bool is_non_overlapping_and_dense = self.is_non_overlapping_and_dense();
  int64_t numel = self.numel();
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  int64_t nsort = self.sizes()[dim];

  TORCH_CHECK(nsort <= std::numeric_limits<int>::max(),
    "The dimension being sorted can not have more than INT_MAX elsments.");

  if (ndim == 0) {
    if (!values.defined()) {
      values = self.clone();
    } else {
      values.resize_as_(self);
      values.copy_(self);
    }
    if (!indices.defined()) {
      indices = at::zeros({}, self.options().dtype(kLong));
    } else {
      indices.resize_as_(self);
      indices.zero_();
    }
    return {values, indices};
  }

  Tensor self_;
  if (is_non_overlapping_and_dense && self.stride(dim) == 1) {
    self_ = self;
  } else {
    auto new_strides_unsort = infer_dense_strides_dim_last(self, dim);
    self_ = at::empty_strided(self.sizes(), new_strides_unsort, self.options());
    self_.copy_(self);
  }

  Tensor values_tmp, indices_tmp;
  void *values_ptr_;
  int64_t *indices_ptr;
  if (!values.defined()) {
    if (is_non_overlapping_and_dense) {
      values = at::empty_strided(self.sizes(), self.strides(), self.options());
    } else {
      auto strides = at::infer_dense_strides(self.sizes(), self.strides());
      values = at::empty_strided(self.sizes(), strides, self.options());
    }
  } else {
    TORCH_CHECK(self_.scalar_type() == values.scalar_type(),
      "Unexpected dtype for values, expect ", self_.scalar_type(), ", got ", values.scalar_type());
    values.resize_as_(self);
  }
  if (values.strides() != self_.strides()) {
    values_tmp = at::empty_strided(self_.sizes(), self_.strides(), self_.options());
    values_ptr_ = values_tmp.data_ptr();
  } else {
    values_ptr_ = values.data_ptr();
  }

  if (!indices.defined()) {
    if (is_non_overlapping_and_dense) {
      indices = at::empty_strided(self.sizes(), self.strides(), self.options().dtype(kLong));
    } else {
      auto strides = at::infer_dense_strides(self.sizes(), self.strides());
      indices = at::empty_strided(self.sizes(), strides, self.options().dtype(kLong));
    }
  } else {
    TORCH_CHECK(kLong == indices.scalar_type(),
      "Unexpected dtype for values, expect torch.long, got ", indices.scalar_type());
    indices.resize_as_(self);
  }
  if (indices.strides() != self_.strides()) {
    indices_tmp = at::empty_strided(self_.sizes(), self_.strides(), self_.options().dtype(kLong));
    indices_ptr = indices_tmp.data_ptr<int64_t>();
  } else {
    indices_ptr = indices.data_ptr<int64_t>();
  }

  if (numel == 0) {
    return {values, indices};
  }

  int64_t numel_or_intmax = std::min(numel, static_cast<int64_t>(std::numeric_limits<int>::max()));
  int64_t nbatch = (numel_or_intmax / nsort) * nsort;

  AT_DISPATCH_ALL_TYPES_AND(kHalf, self_.scalar_type(), "sort", [&]{
    const scalar_t *self_ptr = self_.data_ptr<scalar_t>();
    auto values_ptr = reinterpret_cast<scalar_t *>(values_ptr_);
    int64_t remaining = numel;
    while (remaining > 0) {
      int64_t n = std::min(remaining, nbatch);
      int64_t nsegments = n / nsort;
      int64_t segment_bits = std::max<int64_t>(1L, static_cast<int64_t>(std::ceil(std::log2(nsegments))));

      auto indices_and_segment = at::empty({nsegments, nsort, 2}, indices.options());
      indices_and_segment.select(-1, 0).copy_(  // reverse indices
        at::arange(nsort, indices.options()).view({1, nsort}).expand({nsegments, nsort}));
      indices_and_segment.select(-1, 1).copy_(  // segment id
        at::arange(nsegments, indices.options()).view({nsegments, 1}).expand({nsegments, nsort}));

      using long2 = at::detail::Array<int64_t, 2>;

      long2 *i_s_ptr = reinterpret_cast<long2 *>(indices_and_segment.data_ptr<int64_t>());
      auto indices_and_segment2 = at::empty_like(indices_and_segment);
      long2 *i_s_ptr2 = reinterpret_cast<long2 *>(indices_and_segment2.data_ptr<int64_t>());

      at::cuda::cub::sort_pairs<scalar_t, long2>(
        self_ptr, nullptr, i_s_ptr, i_s_ptr2,
        n, descending);

      auto sorted_indices = indices_and_segment2.select(-1, 0).contiguous();
      auto segment_id = indices_and_segment2.select(-1, 1).contiguous();

      at::cuda::cub::sort_pairs<int64_t, int64_t>(
        segment_id.data_ptr<int64_t>(), nullptr,
        sorted_indices.data_ptr<int64_t>(), indices_ptr,
        n, false, 0, segment_bits);

      sort_gather_kernel<<<(n + 511) / 512, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
        self_ptr, values_ptr, indices_ptr, nsegments, nsort);

      remaining -= n;
      self_ptr += n;
      values_ptr += n;
      indices_ptr += n;
    }
  });

  if (values_tmp.defined()) {
    values.copy_(values_tmp);
  }
  if (indices_tmp.defined()) {
    indices.copy_(indices_tmp);
  }
  return {values, indices};
}

std::tuple<Tensor &,Tensor &> sort_out_cuda(const Tensor & self, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
  if (should_use_th_sort(self, dim)) {
    return legacy::cuda::_th_sort_out(self, dim, descending, values, indices);
  }
  return sort_out_stable_cuda(self, /*stable=*/false, dim, descending, values, indices);
}

std::tuple<Tensor,Tensor> sort_stable_cuda(const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending) {
  if (should_use_th_sort(self, dim)) {
    return legacy::cuda::_th_sort_stable(self, stable, dim, descending);
  }
  Tensor values, indices;
  return sort_out_stable_cuda(self, stable, dim, descending, values, indices);
}

std::tuple<Tensor,Tensor> sort_cuda(const Tensor & self, int64_t dim, bool descending) {  int64_t threshold;
  if (should_use_th_sort(self, dim)) {
    return legacy::cuda::_th_sort(self, dim, descending);
  }
  return sort_stable_cuda(self, /*stable=*/false, dim, descending);
}

}}  // namespace at::native
