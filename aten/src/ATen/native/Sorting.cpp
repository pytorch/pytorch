#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/SortingUtils.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/cpu/QuickPartition.h>
#include <vector>

namespace at {
namespace native {

namespace {


using vec256::int_same_size_t;

template <typename Fn>
void dim_apply(TensorList tensors, int64_t dim, Fn f) {
  AT_ASSERT(tensors.size() > 0);
  auto t = tensors[0];
  auto sizes = t.sizes();
  int64_t ndim = t.dim();
  int64_t itersize = 1;
  for (int64_t i = 0; i < ndim; i++) {
    if (i != dim) {
      itersize *= t.size(i);
    }
  }
  parallel_for(0, itersize, 1, [&](int64_t i_begin, int64_t i_end) {
    std::vector<Tensor> narrowed_tensors;
    narrowed_tensors.reserve(tensors.size());
    for (int64_t it = i_begin; it < i_end; it++) {
      narrowed_tensors.clear();
      for (auto ti : tensors) {
        int64_t i = it;
        Tensor nt = ti;
        for (size_t d = 0; d < ndim; d++) {
          if (d != dim) {
            // this could be avoided for slower-changing dimensions if done
            // better
            nt = nt.select((d > dim ? 1 : 0), i % sizes[d]);
            i = i / sizes[d];
          }
        }
        narrowed_tensors.emplace_back(nt);
      }
      f(it, narrowed_tensors);
    }
  });
}



template <typename scalar_t, typename ScalarComp, typename ScalarFn, typename PartitionFn>
void quick_select_template(
    scalar_t *arr,
    int64_t sz,
    int64_t k,
    ScalarComp gt_or_nan,
    ScalarFn swap_fn,
    PartitionFn partition
    ) {
  int64_t P, L, R;
  scalar_t piv;
  L = 0;
  R = sz - 1;

  do {
    if (R <= L) // One element only
      return;

    if (R == L + 1) { // Two elements only
      if (gt_or_nan(arr[L], arr[R])) {
        swap_fn(L, R);
      }
      return;
    }

    // Use median of three for pivot choice
    P = (L + R) >> 1;
    swap_fn(P, L + 1);
    if (gt_or_nan(arr[L + 1], arr[R])) {
      swap_fn(L + 1, R);
    }
    if (gt_or_nan(arr[L], arr[R])) {
      swap_fn(L, R);
    }
    if (gt_or_nan(arr[L + 1], arr[L])) {
      swap_fn(L + 1, L);
    }

    piv = arr[L];
    int64_t j = partition(L + 1, R, piv);

    // Re-set active partition
    if (j <= k)
      L = j;
    if (j >= k)
      R = j - 1;
  } while (1);
}

} // namespace

// via SFINAE, this version will be used
// if only a scalar partition function is available
template <typename scalar_t, typename index_t>
index_t partition(
    scalar_t *values,
    index_t *indices,
    index_t L,
    index_t R,
    scalar_t piv,
    bool largest,
    long) {
  auto cmp = largest ? [](scalar_t x, scalar_t y) {
      return gt_or_nan<scalar_t>(x, y);
    } : [](scalar_t x, scalar_t y) {
      return gt_or_nan<scalar_t>(y, x);
    };
  auto swap = [&](index_t i, index_t j) {
    std::swap(values[i], values[j]);
    std::swap(indices[i], indices[j]);
  };
  return scalar_partition(values, cmp, swap, L, R, piv);
}

template <typename scalar_t, typename index_t>
auto partition(
    scalar_t *values,
    index_t *indices,
    index_t L,
    index_t R,
    scalar_t piv,
    bool largest,
    int) -> decltype(vec_qs_partition_inplace((scalar_t *)(nullptr), (scalar_t *)(nullptr), (index_t *)(nullptr), scalar_t(), bool()), index_t()){
  scalar_t *begin = values + L;
  scalar_t *end = values + R + 1;
  return L + vec_qs_partition_inplace(
    begin, end, indices + L, piv, largest
  );
}



template <typename scalar_t, typename index_t, typename ValFn, typename IdxFn>
void qsel_with_indices(const Tensor& self, Tensor& values, Tensor& indices, ValFn finalize_vals, IdxFn finalize_idxs, index_t k, bool largest, int64_t dim) {
  dim_apply(
    {self, values, indices},
    dim,
    [&](int64_t i, TensorList tl) {
      auto self = tl[0].accessor<scalar_t, 1>();
      auto sz = self.size(0);
      std::unique_ptr<index_t[]> tmp_indices_(new index_t[sz]);
      std::unique_ptr<scalar_t[]> tmp_values_(new scalar_t[sz]);
      index_t *tmp_indices = tmp_indices_.get();
      scalar_t *tmp_values = tmp_values_.get();
      for (index_t j = 0; j < sz; ++j) {
        tmp_indices[j] = j;
        tmp_values[j] = self[j];
      }
      quick_select_template(tmp_values,
          sz,
          k,
          [largest] (scalar_t x, scalar_t y) -> bool {
            return largest ? gt_or_nan<scalar_t>(x,y)
              : gt_or_nan<scalar_t>(y,x);
          },
          [&] (index_t i, index_t j) {
            std::swap(tmp_values[i], tmp_values[j]);
            std::swap(tmp_indices[i], tmp_indices[j]);
          },
          [&] (index_t L, index_t R, scalar_t piv) {
            return partition<scalar_t, index_t>(tmp_values,
                tmp_indices, L, R, piv, largest, 0);
          }
        );
      finalize_vals(tmp_values, tl[1]);
      finalize_idxs(tmp_indices, tl[2]);
    }
  );
}


std::tuple<Tensor&, Tensor&> kthvalue_out_cpu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  // FIXME: This seems bogus, I only do this because it was the old behaviour.
  //        The reductions are fine, as long as the axis being reduced along
  //        isn't of 0 elements (and the output has elements).
  AT_CHECK(
      self.numel() > 0,
      "cannot perform reduction function kthvalue",
      " on tensor with no elements because the operation does not have an identity");
  AT_CHECK(
      k > 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");

  _reduction_with_indices_allocate_or_resize_output(
      values, indices, self, dim_, keepdim);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "kthvalue_cpu", [&] {
    bool can_use_small_index = self.size(dim) <= std::numeric_limits<int_same_size_t<scalar_t>>::max();
    if (can_use_small_index) {
      using index_t = int_same_size_t<scalar_t>;
      qsel_with_indices<scalar_t, index_t>(self, values, indices,
          [k](const scalar_t *vals, const Tensor& val_slice) {
            *val_slice.data<scalar_t>() = vals[k - 1];
          },
          [k](const index_t *idxs, const Tensor& idx_slice) {
            *idx_slice.data<int64_t>() = idxs[k - 1];
          },
          k,
          true,
          dim
        );
    } else {
      qsel_with_indices<scalar_t, int64_t>(self, values, indices,
          [k](const scalar_t *vals, const Tensor& val_slice) {
            *val_slice.data<scalar_t>() = vals[k - 1];
          },
          [k](const int64_t *idxs, const Tensor& idx_slice) {
            *idx_slice.data<int64_t>() = idxs[k - 1];
          },
          k,
          true,
          dim
        );
    }
  });
  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
  return std::forward_as_tuple(values, indices);
}


std::tuple<Tensor, Tensor> kthvalue(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  at::kthvalue_out(values, indices, self, k, dim, keepdim);
  return std::make_tuple(values, indices);
}

} // namespace native
} // namespace at
