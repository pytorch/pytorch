#include <limits>

#include <ATen/ATen.h>
#include <ATen/WrapDimUtils.h>

#include <thrust/sort.h>
#include <ATen/cuda/cub.cuh>

namespace at { namespace native {

std::tuple<Tensor &,Tensor &> sort_out_stable_cuda(Tensor & values, Tensor & indices, const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending) {
  // this algorithm is always stable
  TORCH_INTERNAL_ASSERT(stable.has_value(), "sort_out(): c10::optional<bool> for stable has to have value.");
  bool is_non_overlapping_and_dense = self.is_non_overlapping_and_dense();
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  int64_t numel = self.numel();
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
    // sort the strides in descending order according to its value,
    // keeping dim the last.
    std::vector<int64_t> strides = self.strides().vec();
    strides[dim] = -1;
    std::vector<int64_t> original_dim(ndim);
    for (int64_t i = 0; i < ndim; i++) {
      original_dim[i] = i;
    }
    thrust::stable_sort_by_key(
      thrust::host, strides.data(), strides.data() + ndim, original_dim.data(),
      thrust::greater<int64_t>()
    );
    // generate contiguous strides on permuted dims
    std::vector<int64_t> new_strides(ndim);
    std::vector<int64_t> new_strides_unsort(ndim);
    int64_t cumprod = 1;
    for (int64_t i = 0; i < ndim; i++) {
      new_strides[ndim - 1 - i] = cumprod;
      cumprod *= self.sizes()[original_dim[ndim - 1 - i]];
    }
    // unsort new strides
    for (int64_t i = 0; i < ndim; i++) {
      new_strides_unsort[original_dim[i]] = new_strides[i];
    }
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
  int64_t nrepeat = nbatch / nsort;

  auto segment_id = at::repeat_interleave(
    at::tensor(nsort, indices.options()).expand(nrepeat));
  int64_t *segment_id_ptr = segment_id.data_ptr<int64_t>();
  auto orig_indices = at::arange(nsort, indices.options()).repeat({nrepeat});
  int64_t *orig_indices_ptr = orig_indices.data_ptr<int64_t>();

  auto tmp = at::empty_like(self_);
  auto segment_id_tmp = at::empty_like(segment_id);
  int64_t *segment_id_tmp_ptr = segment_id_tmp.data_ptr<int64_t>();
  auto orig_indices_tmp = at::empty_like(orig_indices);
  int64_t *orig_indices_tmp_ptr = orig_indices_tmp.data_ptr<int64_t>();

  AT_DISPATCH_ALL_TYPES_AND(kHalf, self_.scalar_type(), "sort", [&]{
    const scalar_t *self_ptr = self_.data_ptr<scalar_t>();
    scalar_t *tmp_ptr = tmp.data_ptr<scalar_t>();
    auto values_ptr = reinterpret_cast<scalar_t *>(values_ptr_);
    int64_t remaining = numel;
    while (remaining > 0) {
      int64_t n = std::min(remaining, nbatch);
      at::cuda::cub::sort_pairs(
        self_ptr, tmp_ptr,
        orig_indices_ptr, orig_indices_tmp_ptr,
        n, descending);
      at::cuda::cub::sort_pairs(
        self_ptr, tmp_ptr,
        segment_id_ptr, segment_id_tmp_ptr,
        n, descending);
      at::cuda::cub::sort_pairs(
        segment_id_tmp_ptr, segment_id_ptr,
        tmp_ptr, values_ptr,
        n);
      at::cuda::cub::sort_pairs(
        segment_id_tmp_ptr, segment_id_ptr,
        orig_indices_tmp_ptr, indices_ptr,
        n);
      remaining -= n;
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

std::tuple<Tensor &,Tensor &> sort_out_cuda(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) {
  return sort_out_stable_cuda(values, indices, self, /*stable=*/false, dim, descending);
}

std::tuple<Tensor,Tensor> sort_stable_cuda(const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending) {
  Tensor values, indices;
  return sort_out_stable_cuda(values, indices, self, stable, dim, descending);
}

std::tuple<Tensor,Tensor> sort_cuda(const Tensor & self, int64_t dim, bool descending) {
  return sort_stable_cuda(self, /*stable=*/false, dim, descending);
}

}}  // namespace at::native
