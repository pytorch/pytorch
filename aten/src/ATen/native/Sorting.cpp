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

// Compute the output shape for the quantile operator
static std::vector<int64_t> quantile_out_shape(
    const Tensor& self,
    const Tensor& q,
    c10::optional<int64_t> _dim,
    bool keepdim
) {
  auto dim = _dim ? at::maybe_wrap_dim(*_dim, self.dim(), true) : 0;
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
  return out_shape;
}

static Tensor quantile_impl(
    const Tensor& self,
    const Tensor& q,
    c10::optional<int64_t> _dim,
    bool keepdim) {
  TORCH_CHECK(self.numel() > 0, "Input tensor must be non-empty");
  TORCH_CHECK(q.dim() <= 1, "q must be a scalar or 1D tensor");
  
  TORCH_CHECK(self.scalar_type() == kFloat || self.scalar_type() == kDouble, 
      "Input tensor must be either float or double dtype");
  TORCH_CHECK(self.scalar_type() == q.scalar_type(), 
      "q must be same dtype as the input tensor");
  
  TORCH_CHECK(self.device() == q.device(), 
      "q must be on the same device as the input tensor");

  if (self.device() == kCPU) {
    TORCH_CHECK(q.ge(0).logical_and_(q.le(1)).all().item<bool>(),
        "q values must be in the range [0, 1]");
  }

  // If user didn't specify a dimension then we flatten the input tensor
  auto in = _dim ? self : self.flatten();
  auto dim = _dim ? at::maybe_wrap_dim(*_dim, self.dim(), true) : 0;

  // We sort the input tensor to efficiently query multiple kth values.
  // In the future, it might be worthwhile to implement multiple quickselect.
  auto sorted = std::get<0>(in.sort(dim));

  // Convert quantile into indices in the given dimension.
  // Check for overflow when casting to a floating point type.
  TORCH_CHECK(sorted.size(dim) <= std::pow(2, 24), "Input tensor is too large");
  auto indices = q * (sorted.size(dim) - 1);
  auto indices_below = indices.floor().toType(kLong);
  auto indices_above = indices.ceil().toType(kLong);

  // Extract values from tensor
  auto values_below = sorted.index_select(dim, indices_below);
  auto values_above = sorted.index_select(dim, indices_above);

  // Interpolate values to get quantiles
  std::vector<int64_t> broadcast_shape(sorted.dim(), 1);
  broadcast_shape[dim] = -1;
  auto weights = (indices - indices_below).reshape(broadcast_shape);
  auto quantiles = values_below.lerp_(values_above, weights);

  // when the q tensor is not a scalar, numpy will prepend a new dimension
  // to contain the quantiles. This code ensures we follow the same order
  if (q.dim() > 0) {
    quantiles.unsqueeze_(0);
    ++dim;
    std::vector<int64_t> numpy_dim_order;
    for (int64_t i = 0; i < quantiles.dim(); ++i) {
      numpy_dim_order.push_back(i);
    }
    std::iter_swap(numpy_dim_order.begin(), numpy_dim_order.begin() + dim);
    quantiles = quantiles.permute(numpy_dim_order);
  }

  return quantiles;
}

Tensor& quantile_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& q,
    c10::optional<int64_t> _dim,
    bool keepdim) {
  auto out_shape = quantile_out_shape(self, q, _dim, keepdim);

  TORCH_CHECK(out.sizes().vec() == out_shape,
      "expected out shape to be ", out_shape, " but got ", out.sizes().vec());
  TORCH_CHECK(self.scalar_type() == out.scalar_type(), 
      "out tensor must be same dtype as the input tensor");
  TORCH_CHECK(self.device() == out.device(), 
      "out tensor must be on the same device as the input tensor");

  if (q.numel() == 0) {
    return out;
  }

  out.copy_(quantile_impl(self, q, _dim, keepdim).reshape(out_shape));

  return out;
}

Tensor& quantile_out(
    Tensor& out,
    const Tensor& self,
    double q,
    c10::optional<int64_t> _dim,
    bool keepdim) {
  TORCH_CHECK(q >= 0 && q <= 1, "q must be in the range [0, 1]");
  return at::quantile_out(out, self, at::scalar_tensor(q, self.options()), _dim, keepdim);
}

Tensor quantile(
    const Tensor& self,
    const Tensor& q,
    c10::optional<int64_t> _dim,
    bool keepdim) {
  auto out_shape = quantile_out_shape(self, q, _dim, keepdim);

  if (q.numel() == 0) {
    return at::empty(out_shape, self.options());
  }

  return quantile_impl(self, q, _dim, keepdim).reshape(out_shape);
}

Tensor quantile(
    const Tensor& self,
    double q,
    c10::optional<int64_t> _dim,
    bool keepdim) {
  TORCH_CHECK(q >= 0 && q <= 1, "q must be in the range [0, 1]");
  return at::quantile(self, at::scalar_tensor(q, self.options()), _dim, keepdim);
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
