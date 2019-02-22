#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/SortingUtils.h>

namespace at {
namespace native {

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

namespace {
constexpr int64_t MAX_LEVELS = 300;
constexpr int64_t M_SMALL = 10; // Limit for small subfiles

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

template <typename scalar_t, typename ComparisonOp>
void quick_sort_impl(
    TensorAccessor<scalar_t, 1> arr,
    TensorAccessor<int64_t, 1> idx,
    ComparisonOp gt_or_nan) {
  auto ARR_SWAP = [&](int64_t i, int64_t j) { std::swap(arr[i], arr[j]); };
  auto BOTH_SWAP = [&](int64_t i, int64_t j) {
    std::swap(arr[i], arr[j]);
    std::swap(idx[i], idx[j]);
  };

  int64_t beg[MAX_LEVELS], end[MAX_LEVELS], i, j, L, R, P, swap, pid,
      stack = 0, sz_right, sz_left;
  scalar_t rswap, piv;
  unsigned char done = 0;

  // beg[0]=0; end[0]=arr.size(0);
  stack = 0;
  L = 0;
  R = arr.size(0) - 1;
  done = arr.size(0) - 1 <= M_SMALL;

  while (!done) {
    // Use median of three for pivot choice
    P = (L + R) >> 1;
    BOTH_SWAP(P, L + 1);
    if (gt_or_nan(arr[L + 1], arr[R])) {
      BOTH_SWAP(L + 1, R);
    }
    if (gt_or_nan(arr[L], arr[R])) {
      BOTH_SWAP(L, R);
    }
    if (gt_or_nan(arr[L + 1], arr[L])) {
      BOTH_SWAP(L + 1, L);
    }

    i = L + 1;
    j = R;
    piv = arr[L];
    pid = idx[L];

    do {
      do {
        i = i + 1;
      } while (gt_or_nan(piv, arr[i]));
      do {
        j = j - 1;
      } while (gt_or_nan(arr[j], piv));
      if (j < i)
        break;
      BOTH_SWAP(i, j);
    } while (1);
    BOTH_SWAP(L, j);
    // Left subfile is (L, j-1)
    // Right subfile is (i, R)
    sz_left = j - L;
    sz_right = R - i + 1;
    if (sz_left <= M_SMALL && sz_right <= M_SMALL) {
      // both subfiles are small
      // if stack empty
      if (stack == 0) {
        done = 1;
      } else {
        stack--;
        L = beg[stack];
        R = end[stack];
      }
    } else if (sz_left <= M_SMALL || sz_right <= M_SMALL) {
      // exactly one of the subfiles is small
      // (L,R) = large subfile
      if (sz_left > sz_right) {
        // Implicit: L = L;
        R = j - 1;
      } else {
        L = i;
        // Implicit: R = R;
      }
    } else {
      // none of the subfiles is small
      // push large subfile
      // (L,R) = small subfile
      if (sz_left > sz_right) {
        beg[stack] = L;
        end[stack] = j - 1;
        stack++;
        L = i;
        // Implicit: R = R
      } else {
        beg[stack] = i;
        end[stack] = R;
        stack++;
        // Implicit: L = L;
        R = j - 1;
      }
    }
  } // while not done
  // Now insertion sort on the concatenation of subfiles
  for (i = arr.size(0) - 2; i >= 0; i--) {
    if (gt_or_nan(arr[i], arr[i + 1])) {
      piv = arr[i];
      pid = idx[i];
      j = i + 1;
      do {
        arr[j - 1] = arr[j];
        idx[j - 1] = idx[j];
        j = j + 1;
      } while (j < arr.size(0) && gt_or_nan(piv, arr[j]));
      arr[j - 1] = piv;
      idx[j - 1] = pid;
    }
  }
}

template <typename scalar_t>
void quick_sort(
    TensorAccessor<scalar_t, 1> arr,
    TensorAccessor<int64_t, 1> idx,
    bool descending) {
  // ComparisonOp emulates NumPy behavior of putting NaNs
  // at the end of an ascending list.
  // We would use a lambda within quick_sort, if it were not for
  // https://stackoverflow.com/questions/27989031/msvc-error-when-using-capture-less-lambda-expressions-as-second-and-third-operan
  if (descending) {
    quick_sort_impl(arr, idx, [](scalar_t x, scalar_t y) -> bool {
      return ((y != y && x == x) || (x < y));
    });
  } else {
    quick_sort_impl(arr, idx, [](scalar_t x, scalar_t y) -> bool {
      return ((x != x && y == y) || (x > y));
    });
  }
}

} // anonymous namespace

std::tuple<Tensor&, Tensor&> sort_out_cpu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim_,
    bool descending) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());

  if (values.defined()) {
    AT_CHECK(
        self.type() == values.type(),
        "output values must be of same type as input");
    values.resize_as_(self);
  } else {
    values = at::empty_like(self);
  }
  at::_copy_same_type_(values, self);
  if (indices.defined()) {
    AT_CHECK(
        indices.dtype() == kLong, "output indices must be of scalar type Long");
    AT_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as input");
    indices.resize_(self.sizes());
  } else {
    indices = at::empty(self.sizes(), self.options().dtype(kLong));
  }
  if (self.dim() == 0 && self.numel() == 1) {
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }
  AT_DISPATCH_ALL_TYPES(self.type(), "sort", [&] {
    dim_apply({values, indices}, dim, [&](int64_t i, TensorList tl) {
      auto values = tl[0].accessor<scalar_t, 1>();
      auto indices = tl[1].accessor<int64_t, 1>();
      for (int64_t j = 0; j < indices.size(0); j++) {
        indices[j] = j;
      }
      quick_sort(values, indices, descending);
    });
  });
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> sort(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  at::sort_out(values, indices, self, dim, descending);
  return std::make_tuple(values, indices);
}

template <typename scalar_t, typename Fn>
void quick_select_template(
    TensorAccessor<scalar_t, 1> arr,
    int64_t k,
    Fn swap_fn) {
  int64_t P, L, R, i, j, swap;
  scalar_t rswap, piv;
  L = 0;
  R = arr.size(0) - 1;

  do {
    if (R <= L) // One element only
      return;

    if (R == L + 1) { // Two elements only
      if (arr[L] > arr[R]) {
        swap_fn(L, R);
      }
      return;
    }

    // Use median of three for pivot choice
    P = (L + R) >> 1;
    swap_fn(P, L + 1);
    if (arr[L + 1] > arr[R]) {
      swap_fn(L + 1, R);
    }
    if (arr[L] > arr[R]) {
      swap_fn(L, R);
    }
    if (arr[L + 1] > arr[L]) {
      swap_fn(L + 1, L);
    }

    i = L + 1;
    j = R;
    piv = arr[L];
    do {
      do
        i++;
      while (arr[i] < piv);
      do
        j--;
      while (arr[j] > piv);
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

std::tuple<Tensor&, Tensor&> mode_out_cpu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim_,
    bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  // FIXME: This seems bogus, I only do this because it was the old behaviour.
  //        The reductions are fine, as long as the axis being reduced along
  //        isn't of 0 elements (and the output has elements).
  AT_CHECK(
      self.numel() > 0,
      "cannot perform reduction function mode",
      " on tensor with no elements because the operation does not have an identity");
  _reduction_with_indices_allocate_or_resize_output(
      values, indices, self, dim_, keepdim);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }

  auto tmp_values = self.clone();
  auto tmp_indices = at::empty(self.sizes(), self.options().dtype(kLong));
  AT_DISPATCH_ALL_TYPES(self.type(), "mode", [&] {
    dim_apply(
        {tmp_values, tmp_indices, values, indices},
        dim,
        [&](int64_t i, TensorList tl) {
          auto tmp_values = tl[0].accessor<scalar_t, 1>();
          auto tmp_indices = tl[1].accessor<int64_t, 1>();
          scalar_t* mode_value = tl[2].data<scalar_t>();
          int64_t* mode_index = tl[3].data<int64_t>();
          for (int64_t j = 0; j < tmp_indices.size(0); j++) {
            tmp_indices[j] = j;
          }
          quick_sort(tmp_values, tmp_indices, /*descending=*/false);
          int64_t max_freq = 0;
          int64_t temp_freq = 0;
          for (i = 0; i < tmp_values.size(0); i++) {
            temp_freq++;
            if ((i == tmp_values.size(0) - 1) ||
                (tmp_values[i] != tmp_values[i + 1])) {
              if (temp_freq > max_freq) {
                *mode_value = tmp_values[i];
                *mode_index = tmp_indices[i];
                max_freq = temp_freq;
              }
              temp_freq = 0;
            }
          }
        });
  });
  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> mode(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  at::mode_out(values, indices, self, dim, keepdim);
  return std::make_tuple(values, indices);
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
  auto tmp_values = self.clone();
  auto tmp_indices = at::empty(self.sizes(), self.options().dtype(kLong));
  AT_DISPATCH_ALL_TYPES(self.type(), "kthvalue", [&] {
    dim_apply(
        {tmp_values, tmp_indices, values, indices},
        dim,
        [&](int64_t i, TensorList tl) {
          auto tmp_values = tl[0].accessor<scalar_t, 1>();
          auto tmp_indices = tl[1].accessor<int64_t, 1>();
          scalar_t* mode_value = tl[2].data<scalar_t>();
          int64_t* mode_index = tl[3].data<int64_t>();
          for (int64_t j = 0; j < tmp_indices.size(0); j++) {
            tmp_indices[j] = j;
          }
          quick_select_template(tmp_values, k - 1, [&](int64_t i, int64_t j) {
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

// this does not reduce to median with dim beause we don't want to copy twice
Tensor median_cpu(const Tensor& self) {
  AT_CHECK(self.numel() > 0, "median cannot be called with empty tensor");
  if (self.dim() == 0 && self.numel() == 1) {
    return self.clone();
  }
  auto tmp_values = self.clone().view(-1);
  auto result = at::empty({1}, self.options());
  AT_DISPATCH_ALL_TYPES(self.type(), "median", [&] {
    // note, quick_select is 0 based while kthvalue is not
    int64_t k = (tmp_values.size(0) - 1) / 2;
    auto val_accessor = tmp_values.accessor<scalar_t, 1>();
    quick_select_template(val_accessor, k, [&](int64_t i, int64_t j) {
      std::swap(val_accessor[i], val_accessor[j]);
    });
    result.fill_(tmp_values[k]);
  });
  return result.view({});
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
  AT_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected number k out of range");
  auto result_sizes = self.sizes().vec();
  if (result_sizes.size() > 0) {
    result_sizes[dim] = k;
  } else if (k == 0) {
    result_sizes.emplace_back(0);
  }
  if (values.defined()) {
    AT_CHECK(
        self.type() == values.type(),
        "output values must be of same type as input");
    AT_CHECK(
        values.device() == self.device(),
        "output values must be on same device as input");
    values.resize_(result_sizes);
  } else {
    values = at::empty(result_sizes, self.options());
  }
  if (indices.defined()) {
    AT_CHECK(
        indices.dtype() == kLong, "output indices must be of scalar type Long");
    AT_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as input");
    indices.resize_(result_sizes);
  } else {
    indices = at::empty(result_sizes, self.options().dtype(kLong));
  }
  if (values.numel() == 0) { // we're done already
    return std::forward_as_tuple(values, indices);
  }
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }
  auto tmp_values = self.clone();
  auto tmp_indices = at::empty(self.sizes(), self.options().dtype(kLong));

  // pivotal element for quick select, 0-based
  int64_t K = largest ? self.size(dim) - k : k - 1;

  AT_DISPATCH_ALL_TYPES(self.type(), "topk", [&] {
    dim_apply({tmp_values, tmp_indices}, dim, [&](int64_t i, TensorList tl) {
      auto tmp_values = tl[0].accessor<scalar_t, 1>();
      auto tmp_indices = tl[1].accessor<int64_t, 1>();
      for (int64_t j = 0; j < tmp_indices.size(0); j++) {
        tmp_indices[j] = j;
      }
      quick_select_template(tmp_values, K, [&](int64_t i, int64_t j) {
        std::swap(tmp_values[i], tmp_values[j]);
        std::swap(tmp_indices[i], tmp_indices[j]);
      });
      auto narrow_values = tl[0].narrow(0, largest ? K : 0, k);
      auto narrow_indices = tl[1].narrow(0, largest ? K : 0, k);
      if (sorted) {
        quick_sort(
            narrow_values.accessor<scalar_t, 1>(),
            narrow_indices.accessor<int64_t, 1>(),
            /*descending=*/largest);
      }
    });
  });
  at::_copy_same_type_(values, tmp_values.narrow(dim, largest ? K : 0, k));
  at::_copy_same_type_(indices, tmp_indices.narrow(dim, largest ? K : 0, k));
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> topk(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  at::topk_out(values, indices, self, k, dim, largest, sorted);
  return std::make_tuple(values, indices);
}

Tensor argsort(const Tensor & self, int64_t dim, bool descending) {
  return std::get<1>(at::sort(self, dim, descending));
}

} // namespace native
} // namespace at
