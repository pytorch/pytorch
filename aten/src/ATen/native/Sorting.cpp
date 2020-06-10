#include <ATen/native/Sorting.h>

#include <ATen/ATen.h>
#include <ATen/NumericUtils.h>
#include <ATen/Parallel.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/SortingUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NamedTensorUtils.h>

namespace at {
namespace native {

namespace {

// maybe these days, one should define a random access iterator and use
// std::sort...
/* Note from TH:

   I cut and pasted (slightly adapted) the quicksort code from
   Sedgewick's 1978 "Implementing Quicksort Programs" article
   http://www.csie.ntu.edu.tw/~b93076/p847-sedgewick.pdf

   It is the state of the art existing implementation. The macros
   are here to make as close a match as possible to the pseudocode of
   Program 2 p.851

   Note that other partition schemes exist, and are typically presented
   in textbook, but those are less efficient. See e.g.
   http://cs.stackexchange.com/questions/11458/quicksort-partitioning-hoare-vs-lomuto

   Julien, November 12th 2013
*/

template <typename scalar_t, typename Comp, typename Fn>
void quick_select_template(
    TensorAccessor<scalar_t, 1> arr,
    int64_t k,
    Comp gt_or_nan,
    Fn swap_fn) {
  int64_t P, L, R, i, j;
  scalar_t piv;
  L = 0;
  R = arr.size(0) - 1;

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

    i = L + 1;
    j = R;
    piv = arr[L];
    do {
      do
        i++;
      while (gt_or_nan(piv, arr[i]));
      do
        j--;
      while (gt_or_nan(arr[j], piv));
      if (j < i)
        break;
      swap_fn(i, j);
    } while (1);
    swap_fn(L, j);

    // Re-set active partition
    if (j <= k)
      L = i;
    if (j >= k)
      R = j - 1;
  } while (1);
}

} // namespace

static std::tuple<Tensor&, Tensor&> kthvalue_out_impl_cpu(
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
  TORCH_CHECK(
      self.numel() > 0,
      "cannot perform reduction function kthvalue",
      " on tensor with no elements because the operation does not have an identity");
  TORCH_CHECK(
      k > 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");

  _reduction_with_indices_allocate_or_resize_output(
      values, indices, self, dim_, keepdim);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }
  auto tmp_values = self.clone(at::MemoryFormat::Contiguous);
  auto tmp_indices = at::empty(self.sizes(), self.options().dtype(kLong));
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "kthvalue_cpu", [&] {
    dim_apply(
        {tmp_values, tmp_indices, values, indices},
        dim,
        [&](int64_t i, TensorList tl) {
          auto tmp_values = tl[0].accessor<scalar_t, 1>();
          auto tmp_indices = tl[1].accessor<int64_t, 1>();
          scalar_t* mode_value = tl[2].data_ptr<scalar_t>();
          int64_t* mode_index = tl[3].data_ptr<int64_t>();
          for (int64_t j = 0; j < tmp_indices.size(0); j++) {
            tmp_indices[j] = j;
          }
          // we want NaN to be sorted as top for numpy compatibility
          quick_select_template(
              tmp_values,
              k - 1,
              [](scalar_t x, scalar_t y) -> bool {
                return (
                    (_isnan<scalar_t>(x) && !_isnan<scalar_t>(y)) || (x > y));
              },
              [&](int64_t i, int64_t j) {
                std::swap(tmp_values[i], tmp_values[j]);
                std::swap(tmp_indices[i], tmp_indices[j]);
              });
          *mode_value = tmp_values[k - 1];
          *mode_index = tmp_indices[k - 1];
        });
  });
  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor&, Tensor&> kthvalue_out_cpu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim) {
  auto result = [&]() {
    NoNamesGuard guard;
    return kthvalue_out_impl_cpu(values, indices, self, k, dim, keepdim);
  }();
  namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);
  return result;
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

std::tuple<Tensor&, Tensor&> topk_out_cpu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");

  _allocate_or_resize_output_with_indices(values, indices, self, dim_, k);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }

  topk_stub(kCPU, values, indices, self, k, dim, largest, sorted);

  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> topk(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  const int limit = 100000;
  // const int limit = 20000;
  const int _dividers[] = {100, 50, 25, 20, 15, 12, 10, 13, 11, 7, 5, 3, 2};
  // const int _dividers[] = {10, 12, 15, 20, 25, 50, 100};
  // const int _dividers[] = {10, 12, 15, 25};
  // const int _dividers[] = {15, 20, 25, 50, 100};
  // const int _dividers[] = {2, 3, 5, 7, 11, 13, 15, 17};
  int selected_divider = -1;

  if (self.size(dim) >= limit) {
    for (int _divider : _dividers) {
      if (self.size(dim) % _divider == 0 && self.size(dim) > k * _divider) {
        selected_divider = _divider;
        break;
      }
    }
  }

  if (selected_divider < 0) {
    // The original path
    Tensor values = at::empty({0}, self.options());
    Tensor indices = at::empty({0}, self.options().dtype(kLong));
    at::topk_out(values, indices, self, k, dim, largest, sorted);
    return std::make_tuple(values, indices);
  }
  else {
    // divide and conquer, see https://github.com/pytorch/pytorch/issues/38475
    const int64_t size_dim_old = self.size(dim);
    const int64_t size_dim_new = self.size(dim) / selected_divider;
    TORCH_INTERNAL_ASSERT(self.size(dim) % selected_divider == 0, 
      "\nShould have self.size(dim) %% selected_divider == 0, but self.size(dim) = ", 
      self.size(dim), ", selected_divisor = ", selected_divider);
    TORCH_INTERNAL_ASSERT(size_dim_new >= k, 
      "\nShould have size_dim_new >= k, but size_dim_new = ", 
      size_dim_new, ", k = ", k);

    std::vector<int64_t> sizes_new = self.sizes().vec();
    int insert_dim = dim >= 0 ? dim : self.ndimension() + dim;
    sizes_new.insert(sizes_new.begin() + insert_dim, selected_divider);
    sizes_new[insert_dim+1] = size_dim_new;

    std::vector<int64_t> indices_shape(self.ndimension()+1, 1);
    indices_shape[insert_dim] = selected_divider;
    indices_shape[insert_dim+1] = size_dim_new;

    std::vector<int64_t> sizes_new_infer = self.sizes().vec();
    sizes_new_infer[insert_dim] = -1;

    auto partial = at::topk(self.view(sizes_new), k, insert_dim+1, largest, sorted);
    auto indices_view = at::arange(size_dim_old, self.options().dtype(at::kLong))
                          .view(indices_shape)
                          .expand(sizes_new)
                          .gather(insert_dim+1, std::get<1>(partial))
                          .view(sizes_new_infer);

    auto values_2d = std::get<0>(partial).view(sizes_new_infer);
    auto topk_2 = at::topk(values_2d, k, dim, largest, sorted);
    return std::make_tuple(std::get<0>(topk_2), indices_view.gather(dim, std::get<1>(topk_2)));
  }
}

std::tuple<Tensor&, Tensor&> median_out(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  // note: kthvalue counts from 1..n
  int64_t k = self.dim() > 0 ? (self.size(dim) + 1) / 2 : 1;
  at::kthvalue_out(values, indices, self, k, dim, keepdim);
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> median(
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  at::median_out(values, indices, self, dim, keepdim);
  return std::make_tuple(values, indices);
}

std::tuple<Tensor&, Tensor&> median_out(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    Dimname dim,
    bool keepdim) {
  return at::median_out(values, indices, self, dimname_to_position(self, dim), keepdim);
}

std::tuple<Tensor, Tensor> median(
    const Tensor& self,
    Dimname dim,
    bool keepdim) {
  return at::median(self, dimname_to_position(self, dim), keepdim);
}

std::tuple<Tensor&, Tensor&> kthvalue_out(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    Dimname dim,
    bool keepdim) {
  return at::kthvalue_out(values, indices, self, k, dimname_to_position(self, dim), keepdim);
}

std::tuple<Tensor, Tensor> kthvalue(
    const Tensor& self,
    int64_t k,
    Dimname dim,
    bool keepdim) {
  return at::kthvalue(self, k, dimname_to_position(self, dim), keepdim);
}

// this does not reduce to median with dim because we don't want to copy twice
Tensor median_cpu(const Tensor& self) {
  NoNamesGuard guard;
  TORCH_CHECK(self.numel() > 0, "median cannot be called with empty tensor");
  if (self.dim() == 0 && self.numel() == 1) {
    return self.clone(at::MemoryFormat::Contiguous);
  }
  auto tmp_values = self.clone(at::MemoryFormat::Contiguous).view(-1);
  auto result = at::empty({1}, self.options());
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "median", [&] {
    // note, quick_select is 0 based while kthvalue is not
    int64_t k = (tmp_values.size(0) - 1) / 2;
    auto val_accessor = tmp_values.accessor<scalar_t, 1>();
    quick_select_template(
        val_accessor,
        k,
        [](scalar_t x, scalar_t y) -> bool {
          return ((_isnan<scalar_t>(x) && !_isnan<scalar_t>(y)) || (x > y));
        },
        [&](int64_t i, int64_t j) {
          std::swap(val_accessor[i], val_accessor[j]);
        });
    result.fill_(tmp_values[k]);
  });
  return result.view({});
}

DEFINE_DISPATCH(topk_stub);

} // namespace native
} // namespace at
