#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NumericUtils.h>
#include <ATen/Parallel.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/Sorting.h>
#include <ATen/native/SortingUtils.h>

#include <utility>

namespace at {
namespace native {

DEFINE_DISPATCH(topk_stub);

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

template <typename scalar_t>
void quickselect(
    scalar_t* start,
    scalar_t* first,
    scalar_t* last,
    typename std::set<int64_t>::iterator begin,
    typename std::set<int64_t>::iterator end) {
  if (last <= first || begin == end) {
    return;
  }

  scalar_t pivot = first[std::distance(first, last) >> 1];
  scalar_t* mid1 =
      std::partition(first, last, [=](scalar_t x) { return x < pivot; });
  scalar_t* mid2 =
      std::partition(mid1, last, [=](scalar_t x) { return x == pivot; });

  int64_t mid1_r = std::distance(start, mid1);
  int64_t mid2_r = std::distance(start, mid2);

  quickselect(start, first, mid1, begin, std::lower_bound(begin, end, mid1_r));
  quickselect(start, mid2, last, std::lower_bound(begin, end, mid2_r), end);
}

void quantile_cpu_impl(
    Tensor& out,
    const Tensor& self,
    const Tensor& q,
    const c10::optional<int64_t>& _dim,
    bool keepdim) {
  int64_t dim = _dim ? at::maybe_wrap_dim(*_dim, self.dim(), true) : 0;

  TORCH_CHECK(
      q.ge(0).logical_and_(q.le(1)).all().item<bool>(),
      "quantile() q values must be in the range [0, 1]");

  if (q.dim() == 0) {
    out.unsqueeze_(0);
  }

  Tensor in;
  if (!_dim) {
    in = self.clone(at::MemoryFormat::Contiguous).flatten().unsqueeze_(0);
  } else if (dim == self.dim() - 1) {
    in = self.clone(at::MemoryFormat::Contiguous).unsqueeze_(0).flatten(0, -2);
  } else {
    in = self.unsqueeze(-1).transpose_(dim, -1).flatten(0, -2).contiguous();
  }

  int64_t q_size = q.numel();

  AT_DISPATCH_FLOATING_TYPES(in.scalar_type(), "quantile_cpu_impl", [&] {
    scalar_t* const OP = out.data_ptr<scalar_t>();
    scalar_t* const IP = in.data_ptr<scalar_t>();
    const scalar_t* const QP = q.data_ptr<scalar_t>();

    std::vector<int64_t> rb(q_size);
    std::vector<int64_t> ra(q_size);
    std::vector<scalar_t> w(q_size);
    std::set<int64_t> kths;
    for (int64_t i = 0; i < q_size; ++i) {
      long double rank = QP[i] * (in.size(1) - 1);
      rb[i] = rank;
      ra[i] = rank;
      w[i] = rank - rb[i];
      kths.insert(rank);
      if (w[i] > 0) {
        ra[i] = rank + 1;
        kths.insert(rank + 1);
      }
    }

    at::parallel_for(0, in.size(0), 0, [&](int64_t begin, int64_t end) {
      scalar_t* ip = IP + begin * in.size(1);
      for (int64_t it = begin; it < end; ++it, ip += in.size(1)) {
        quickselect(ip, ip, ip + in.size(1), kths.begin(), kths.end());
        scalar_t* op = OP + it;
        for (int64_t i = 0; i < q_size; ++i, op += out.stride(0)) {
          *op = ip[rb[i]] + w[i] * (ip[ra[i]] - ip[rb[i]]);
        }
      }
    });
  });

  if (q.dim() == 0) {
    out.squeeze_(0);
  }
}

void quantile_impl(
    Tensor& out,
    const Tensor& self,
    const Tensor& q,
    c10::optional<int64_t> _dim,
    bool keepdim) {
  int64_t dim = _dim ? at::maybe_wrap_dim(*_dim, self.dim(), true) : 0;

  // Compute output shape
  std::vector<int64_t> out_shape;
  if (_dim) {
    out_shape = self.sizes().vec();
    if (keepdim) {
      out_shape[dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + dim);
    }
  } else if (keepdim) {
    out_shape = std::vector<int64_t>(self.dim(), 1);
  }
  if (q.dim() > 0) {
    out_shape.insert(out_shape.begin(), q.numel());
  }

  // Check or resize output
  if (out.numel() > 0) {
    TORCH_CHECK(
        out.sizes().vec() == out_shape,
        "quantile() expected output shape to be ",
        out_shape,
        " but got ",
        out.sizes().vec());
    TORCH_CHECK(
        self.scalar_type() == out.scalar_type(),
        "quantile() out tensor must be same dtype as the input tensor");
    TORCH_CHECK(
        self.device() == out.device(),
        "quantile() out tensor must be on the same device as the input tensor");
  } else {
    resize_output(out, out_shape);
  }

  TORCH_CHECK(self.numel() > 0, "quantile() input tensor must be non-empty");
  TORCH_CHECK(q.dim() <= 1, "quantile() q must be a scalar or 1D tensor");
  TORCH_CHECK(
      self.scalar_type() == kFloat || self.scalar_type() == kDouble,
      "quantile() input tensor must be either float or double dtype");
  TORCH_CHECK(
      self.scalar_type() == q.scalar_type(),
      "quantile() q must be same dtype as the input tensor");
  TORCH_CHECK(
      self.device() == q.device(),
      "quantile() q must be on the same device as the input tensor");

  if (self.device().is_cpu()) {
    return quantile_cpu_impl(out, self, q, _dim, keepdim);
  }

  // If q is scalar, treat as 1D during computations
  if (q.dim() == 0) {
    out_shape.insert(out_shape.begin(), q.numel());
  }

  // Move dimension to reduce as last dimension and flatten if no dim was
  // specified by the user. Sort to efficiently query kth values.
  Tensor sorted;
  if (!_dim) {
    sorted = std::get<0>(self.flatten().sort());
  } else if (dim == self.dim() - 1) {
    sorted = std::get<0>(self.sort());
  } else {
    sorted = std::get<0>(self.unsqueeze(-1).transpose_(dim, -1).sort());
  }

  // View input in correct reduced sizes
  std::vector<int64_t> in_shape(out_shape.size());
  std::copy(out_shape.begin() + 1, out_shape.end(), in_shape.begin());
  in_shape[in_shape.size() - 1] = sorted.size(-1);
  sorted = sorted.view(in_shape);

  // Ensure converting from int64_t to double won't overflow
  TORCH_CHECK(
      sorted.size(-1) <= std::pow(2, 24),
      "quantile() input tensor is too large");

  Tensor ranks = q * (sorted.size(-1) - 1);
  Tensor ranks_below = ranks.toType(kLong);
  Tensor weights = ranks - ranks_below;
  Tensor ranks_above = ranks.ceil_().toType(kLong);
  Tensor values_below = sorted.index_select(-1, ranks_below);
  Tensor values_above = sorted.index_select(-1, ranks_above);

  // Make out match expected shape for lerp
  Tensor result = q.dim() == 0
      ? out.unsqueeze(-1)
      : out.unsqueeze(-1).transpose_(0, -1).squeeze_(0);

  at::lerp_out(result, values_below, values_above, weights);
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

Tensor& quantile_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& q,
    c10::optional<int64_t> _dim,
    bool keepdim) {
  quantile_impl(out, self, q, std::move(_dim), keepdim);
  return out;
}

Tensor& quantile_out(
    Tensor& out,
    const Tensor& self,
    double q,
    c10::optional<int64_t> _dim,
    bool keepdim) {
  TORCH_CHECK(
      q >= 0 && q <= 1, "quantile() q must be in the range [0, 1] but got ", q);
  return at::quantile_out(
      out,
      self,
      at::scalar_tensor(q, self.options()),
      std::move(_dim),
      keepdim);
}

Tensor quantile(
    const Tensor& self,
    const Tensor& q,
    c10::optional<int64_t> _dim,
    bool keepdim) {
  Tensor out = at::empty({0}, self.options());
  quantile_impl(out, self, q, std::move(_dim), keepdim);
  return out;
}

Tensor quantile(
    const Tensor& self,
    double q,
    c10::optional<int64_t> _dim,
    bool keepdim) {
  TORCH_CHECK(
      q >= 0 && q <= 1, "quantile() q must be in the range [0, 1] but got ", q);
  return at::quantile(
      self, at::scalar_tensor(q, self.options()), std::move(_dim), keepdim);
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
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  at::topk_out(values, indices, self, k, dim, largest, sorted);
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

std::tuple<Tensor&, Tensor&> median_out(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    Dimname dim,
    bool keepdim) {
  return at::median_out(
      values, indices, self, dimname_to_position(self, dim), keepdim);
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
  return at::kthvalue_out(
      values, indices, self, k, dimname_to_position(self, dim), keepdim);
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

} // namespace native
} // namespace at
