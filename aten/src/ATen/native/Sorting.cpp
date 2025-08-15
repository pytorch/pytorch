#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NumericUtils.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/Sorting.h>
#include <ATen/native/SortingUtils.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/argsort_native.h>
#include <ATen/ops/broadcast_tensors.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/kthvalue.h>
#include <ATen/ops/kthvalue_native.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/median.h>
#include <ATen/ops/median_native.h>
#include <ATen/ops/msort_native.h>
#include <ATen/ops/nanmedian.h>
#include <ATen/ops/nanmedian_native.h>
#include <ATen/ops/nanquantile_native.h>
#include <ATen/ops/quantile_native.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/sort_native.h>
#include <ATen/ops/topk_native.h>
#endif

#include <utility>

namespace at::meta {

using namespace ::at::native;

TORCH_META_FUNC(topk)
(const Tensor& self, int64_t k, int64_t dim_, bool largest, bool sorted) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");
  int64_t sliceSize = self.dim() == 0 ? 1 : self.size(dim);
  TORCH_CHECK(k >= 0 && k <= sliceSize, "k not in range for dimension");
  TORCH_CHECK(!self.is_complex(), " topk does not support complex dtypes on CPU");
  TORCH_CHECK(!(self.scalar_type() == kBool), "topk does not support bool dtypes on CPU");

  // Build the output size, which is the dim being selected set to
  // size k
  DimVector topKSize(self.sizes().vec());
  if (!topKSize.empty()) {
    topKSize[dim] = k;
  }
  set_output_raw_strided(0, topKSize, {}, self.options());
  set_output_raw_strided(1, topKSize, {}, self.options().dtype(at::kLong));
}

TORCH_META_FUNC2(sort, stable)
(const Tensor& self, std::optional<bool> stable, int64_t dim, bool descending) {
  maybe_wrap_dim(dim, self.dim());

  TORCH_CHECK(!self.is_complex(), " Sort does not support complex dtypes on CPU");

  // See issue: https://github.com/pytorch/pytorch/issues/65863
  // Strides should be dense, so as not to allocate too much memory.
  // We either use 'self' strides, or infer dense strides from them.
  std::vector<int64_t> strides = (self.is_non_overlapping_and_dense())
      ? self.strides().vec()
      : at::infer_dense_strides(self.sizes(), self.strides());

  set_output_raw_strided(0, self.sizes(), strides, self.options(), {});
  set_output_raw_strided(1, self.sizes(), strides, self.options().dtype(kLong), {});
}

} // namespace at::meta

namespace at::native {

DEFINE_DISPATCH(sort_stub);
DEFINE_DISPATCH(topk_stub);

void _fill_indices(const TensorBase &indices, int64_t dim) {
  auto ndim = indices.dim();
  assert(0 <= dim && dim < ndim);
  auto dim_size = indices.size(dim);
  auto idx_dim = at::arange(0, dim_size, indices.options().dtype(at::kLong));
  auto idx_dim_sizes = std::vector<int64_t>(ndim, 1);
  auto idx_dim_strides = std::vector<int64_t>(ndim, 0);
  idx_dim_sizes[dim] = dim_size;
  idx_dim_strides[dim] = 1;
  auto idx_dim_restrided = idx_dim.as_strided(idx_dim_sizes, idx_dim_strides);
  OptionalTensorRef(indices)->copy_(idx_dim_restrided);
}

namespace {

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
  int64_t L = 0;
  int64_t R = arr.size(0) - 1;

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
    auto P = L + (R - L) / 2;
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

    auto i = L + 1;
    auto j = R;
    auto piv = arr[L];
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
    } while (true);
    swap_fn(L, j);

    // Re-set active partition
    if (j <= k)
      L = i;
    if (j >= k)
      R = j - 1;
  } while (true);
}

namespace {

QUANTILE_INTERPOLATION_MODE get_quantile_interpolation_mode(
    const std::string_view interpolation) {
  if (interpolation == "linear") {
    return QUANTILE_INTERPOLATION_MODE::LINEAR;
  } else if (interpolation == "lower") {
    return QUANTILE_INTERPOLATION_MODE::LOWER;
  } else if (interpolation == "higher") {
    return QUANTILE_INTERPOLATION_MODE::HIGHER;
  } else if (interpolation == "midpoint") {
    return QUANTILE_INTERPOLATION_MODE::MIDPOINT;
  } else if (interpolation == "nearest") {
    return QUANTILE_INTERPOLATION_MODE::NEAREST;
  } else {
    TORCH_CHECK(
        false,
        "quantile() interpolation must be one of linear, lower, higher, midpoint or nearest, but got ",
        interpolation);
  }
}

void quantile_checks(const Tensor& self, const Tensor& q) {
  TORCH_CHECK(self.numel() > 0, "quantile() input tensor must be non-empty");
  TORCH_CHECK(q.dim() <= 1, "quantile() q must be a scalar or 1D tensor");
  TORCH_CHECK(
      self.scalar_type() == kFloat || self.scalar_type() == kDouble,
      "quantile() input tensor must be either float or double dtype");
  TORCH_CHECK(
      self.scalar_type() == q.scalar_type(),
      "quantile() q tensor must be same dtype as the input tensor");
  TORCH_CHECK(
      self.device() == q.device(),
      "quantile() q tensor must be on the same device as the input tensor");
}

std::vector<int64_t> quantile_output_shape(
    const std::optional<int64_t> original_dim,
    const Tensor& self,
    const Tensor& q,
    const bool keepdim,
    int64_t wrapped_dim) {
  // Compute output shape: q_size + reduced_size
  std::vector<int64_t> out_shape;
  if (original_dim && self.dim() > 0) {
    out_shape = self.sizes().vec();
    if (keepdim) {
      out_shape[wrapped_dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + wrapped_dim);
    }
  } else if (keepdim) {
    out_shape = std::vector<int64_t>(self.dim(), 1);
  }
  if (q.dim() > 0) {
    out_shape.insert(out_shape.begin(), q.numel());
  }

  return out_shape;
}

Tensor quantile_compute(
    const Tensor& self,
    const Tensor& q,
    const std::optional<int64_t> orginal_dim,
    const bool keepdim,
    const QUANTILE_INTERPOLATION_MODE& interpolation,
    const bool ignore_nan,
    int64_t wrapped_dim,
    std::vector<int64_t> out_shape) {
  // Checks that all q values are between 0 and 1, inclusive
  // NOTE: this check is only performed when running on the CPU to avoid
  // synchronizing an accelerator with the CPU
  if (self.device().is_cpu()) {
    auto all_q_in_range = q.ge(0).logical_and_(q.le(1)).all();
    TORCH_CHECK(at::is_scalar_tensor_true(all_q_in_range),
                "quantile() q values must be in the range [0, 1]");
  }

  // Flatten input if no dim provided else move dim to reduce as last dimension.
  // Sort to efficiently query kth values.
  Tensor sorted;
  if (!orginal_dim) {
    sorted = std::get<0>(self.flatten().sort());
  } else if (wrapped_dim == self.dim() - 1) {
    sorted = std::get<0>(self.sort());
  } else {
    sorted = std::get<0>(self.unsqueeze(-1).transpose(wrapped_dim, -1).sort());
  }

  // Treat q as a 1D tensor for the following computations
  if (q.dim() == 0) {
    out_shape.insert(out_shape.begin(), q.numel());
  }

  // View input as reduced_size + size of dim to reduce
  std::vector<int64_t> in_shape(out_shape.size());
  std::copy(out_shape.begin() + 1, out_shape.end(), in_shape.begin());
  in_shape[in_shape.size() - 1] = sorted.size(-1);
  sorted = sorted.view(in_shape);

  // Ensure converting from int64_t to double won't overflow
  TORCH_CHECK(
      sorted.size(-1) <= std::pow(2, 24),
      "quantile() input tensor is too large");

  // Convert q in [0, 1] to ranks in [0, reduction_size)
  Tensor ranks;
  if (ignore_nan) {
    // For nanquantile, compute ranks based on number of non-nan values.
    // If all values are nan, set rank to 0 so the quantile computed is nan.
    ranks = q * (sorted.isnan().logical_not_().sum(-1, true) - 1);
    // For Composite Compliance,
    // if `ranks` is `CCT` but it's tangent is a regular Tensor,
    // then while computing jvp, we end calling `masked_fill_`
    // on a regular Tensor with CCT args, so we call
    // `masked_fill` instead.
    if (isTensorSubclassLike(ranks) && ranks._fw_grad(/*level=*/0).defined()) {
      ranks = ranks.masked_fill(ranks < 0, 0);
    } else {
      ranks.masked_fill_(ranks < 0, 0);
    }
  } else {
    // For quantile, compute ranks based on reduction size. If there is nan
    // set rank to last index so the quantile computed will be nan.
    int64_t last_index = sorted.size(-1) - 1;
    std::vector<Tensor> tl =
        at::broadcast_tensors({q * last_index, sorted.isnan().any(-1, true)});
    ranks = at::masked_fill(tl[0], tl[1], last_index);
  }

  // adjust ranks based on the interpolation mode
  if (interpolation == QUANTILE_INTERPOLATION_MODE::LOWER) {
    ranks.floor_();
  } else if (interpolation == QUANTILE_INTERPOLATION_MODE::HIGHER) {
    ranks.ceil_();
  } else if (interpolation == QUANTILE_INTERPOLATION_MODE::NEAREST) {
    ranks.round_();
  }

  Tensor ranks_below = ranks.toType(kLong);
  Tensor values_below = sorted.gather(-1, ranks_below);

  // Actual interpolation is only needed for the liner and midpoint modes
  if (interpolation == QUANTILE_INTERPOLATION_MODE::LINEAR ||
      interpolation == QUANTILE_INTERPOLATION_MODE::MIDPOINT) {
    // calculate weights for linear and midpoint
    Tensor weights = interpolation == QUANTILE_INTERPOLATION_MODE::MIDPOINT
        ? at::full_like(ranks, 0.5)
        : ranks - ranks_below;

    // Interpolate to compute quantiles and store in values_below
    Tensor ranks_above = ranks.ceil_().toType(kLong);
    Tensor values_above = sorted.gather(-1, ranks_above);
    // For Composite Compliance,
    // if either `values_below`, `values_above` or `weights` are a CCT
    // or tangents of `value_above` and `weights` are a CCT,
    // but if the tangent of `value_below` is a regular Tensor,
    // then while computing jvp, we will end-up copying a `CCT`,
    // into regular Tensor. So we use out-of-place variant of `lerp`
    auto is_primal_cct =
        areAnyTensorSubclassLike({values_below, values_above, weights});
    auto is_tangent_cct = areAnyTensorSubclassLike(
        {values_above._fw_grad(/*level=*/0), weights._fw_grad(/*level=*/0)});
    if ((is_primal_cct || is_tangent_cct) &&
        values_below._fw_grad(/*level=*/0).defined() &&
        !isTensorSubclassLike(values_below._fw_grad(/*level=*/0))) {
      values_below = values_below.lerp(values_above, weights);
    } else {
      values_below.lerp_(values_above, weights);
    }
  }

  if (q.dim() == 0) {
    // If q is scalar, remove last dim to match out shape
    values_below.squeeze_(-1);
  } else {
    // Move quantiles to first dim to match out shape
    values_below.unsqueeze_(0).transpose_(0, -1).squeeze_(-1);
  }

  return values_below;
}

} // namespace

void quantile_out_impl(
    Tensor& out,
    const Tensor& self,
    const Tensor& q,
    const std::optional<int64_t> original_dim,
    const bool keepdim,
    const QUANTILE_INTERPOLATION_MODE& interpolation,
    const bool ignore_nan) {
  quantile_checks(self, q);
  TORCH_CHECK(
      self.scalar_type() == out.scalar_type(),
      "quantile() out tensor must be same dtype as the input tensor");
  TORCH_CHECK(
      self.device() == out.device(),
      "quantile() out tensor must be on the same device as the input tensor");

  int64_t wrapped_dim = at::maybe_wrap_dim(original_dim.value_or(0), self.dim());

  auto out_shape = quantile_output_shape(original_dim, self, q, keepdim, wrapped_dim);
  resize_output(out, out_shape);

  auto quantile = quantile_compute(
      self, q, original_dim, keepdim, interpolation, ignore_nan, wrapped_dim, std::move(out_shape));
  out.copy_(quantile);
}

Tensor quantile_impl(
    const Tensor& self,
    const Tensor& q,
    const std::optional<int64_t> original_dim,
    const bool keepdim,
    const QUANTILE_INTERPOLATION_MODE& interpolation,
    const bool ignore_nan) {
  quantile_checks(self, q);

  int64_t wrapped_dim = at::maybe_wrap_dim(original_dim.value_or(0), self.dim());

  auto out_shape = quantile_output_shape(original_dim, self, q, keepdim, wrapped_dim);

  return quantile_compute(
      self, q, original_dim, keepdim, interpolation, ignore_nan, wrapped_dim, std::move(out_shape));
}

std::tuple<Tensor&, Tensor&> kthvalue_out_impl_cpu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  int64_t slicesize = self.dim() == 0 ? 1 : self.size(dim);
  zero_numel_check_dims(self, dim, "kthvalue()");

  TORCH_CHECK(k >= 1 && k <= slicesize,
              "kthvalue(): selected number k out of range for dimension ", dim);

  at::assert_no_overlap(self, values);

  _reduction_with_indices_allocate_or_resize_output(
      values, indices, self, dim_, keepdim);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }
  auto tmp_values = self.clone(at::MemoryFormat::Contiguous);
  auto tmp_indices = at::empty(self.sizes(), self.options().dtype(kLong));

  auto tmp_values_stride = tmp_values.strides()[dim];
  auto tmp_indices_stride = tmp_indices.strides()[dim];
  auto sizes = self.sizes();

  TORCH_CHECK(indices.scalar_type() == kLong);

  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(sizes, /*squash_dims=*/dim)
    .add_output(tmp_values)
    .add_output(tmp_indices)
    .add_output(values)
    .add_output(indices)
    .build();

  AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "kthvalue_cpu", [&] {
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
      for (const auto i : c10::irange(n)) {
        TensorAccessor<scalar_t, 1> tmp_values(
            reinterpret_cast<scalar_t*>(data[0] + i * strides[0]),
            &sizes[dim], &tmp_values_stride);
        TensorAccessor<int64_t, 1> tmp_indices(
            reinterpret_cast<int64_t*>(data[1] + i * strides[1]),
            &sizes[dim], &tmp_indices_stride);
        auto mode_value = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
        auto mode_index = reinterpret_cast<int64_t*>(data[3] + i * strides[3]);

        for (const auto j : c10::irange(tmp_indices.size(0))) {
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
      }
    };

    int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, sizes[dim]);
    iter.for_each(loop, /*grain_size=*/grain_size);
  });

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
  return std::forward_as_tuple(values, indices);
}

// Computes both the median and its index along dimension dim of the input
std::tuple<Tensor&, Tensor&> median_with_indices_impl(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    bool ignore_nan) {
  dim = at::maybe_wrap_dim(dim, self.dim());

  int64_t size = self.dim() > 0 ? self.size(dim) : 1;
  zero_numel_check_dims(self, dim, "median()");

  checkDeviceType("median", {values, indices}, self.device().type());
  checkScalarType("median", {indices, "indices", 1}, kLong);
  checkSameType("median", {values, "values", 0}, {self, "self", 2});

  std::vector<int64_t> out_shape = self.sizes().vec();
  if (self.dim() > 0) {
    if (keepdim) {
      out_shape[dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + dim);
    }
  }

  resize_output(values, out_shape);
  resize_output(indices, out_shape);

  // Ensure #dim is the same for all tensors required for dim_apply
  Tensor in = self.dim() > 0 ? self : self.unsqueeze(0);
  Tensor vals = keepdim && self.dim() > 0 ? values : values.unsqueeze(dim);
  Tensor inds = keepdim && self.dim() > 0 ? indices : indices.unsqueeze(dim);

  // Make dim to reduce contiguous (stride=1)
  if (in.stride(dim) > 1) {
    in = in.unsqueeze(-1).transpose_(dim, -1).squeeze_(dim).contiguous();
    vals = vals.unsqueeze(-1).transpose_(dim, -1).squeeze_(dim);
    inds = inds.unsqueeze(-1).transpose_(dim, -1).squeeze_(dim);
    dim = in.dim() - 1;
  }

  auto sizes = in.sizes();
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(sizes, /*squash_dims=*/dim)
    .add_output(vals)
    .add_output(inds)
    .add_const_input(in)
    .build();

  AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, in.scalar_type(), "median_out", [&] {
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
      for (const auto i : c10::irange(n)) {
        auto valp = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto indp = reinterpret_cast<int64_t*>(data[1] + i * strides[1]);
        auto ip = reinterpret_cast<const scalar_t*>(data[2] + i * strides[2]);

        // For torch.median, search for NaN and return it if found
        if (!ignore_nan) {
          const scalar_t* nanp = std::find_if(ip, ip + size, _isnan<scalar_t>);
          if (nanp != ip + size) {
            *valp = *nanp;
            *indp = nanp - ip;
            continue;
          }
        }

        // Vector of indices for indirectly partitioning input around median
        std::vector<int64_t> idx(size);
        auto first = idx.begin();
        auto last = idx.end();
        std::iota(first, last, 0);

        // We partition the input around the median indirectly using the indices
        // vector so that nth points to the index of the median in the unmodified
        // input tensor.
        auto nth = first;
        if (!ignore_nan) {
          // If we got here, there are no nan values
          nth += (size - 1) / 2;
          std::nth_element(first, nth, last, [&ip](int64_t i, int64_t j) {
            return ip[i] < ip[j] || (ip[i] == ip[j] && i < j);
          });
        } else {
          // For torch.nanmedian, compute median of non-nan values only
          int64_t num_nan = std::count_if(ip, ip + size, _isnan<scalar_t>);
          nth += (size - num_nan - 1) / 2;
          std::nth_element(first, nth, last, [&ip](int64_t i, int64_t j) {
            return ip[i] < ip[j] || (ip[i] == ip[j] && i < j) ||
                (_isnan(ip[j]) && !_isnan(ip[i]));
          });
        }

        *valp = ip[*nth];
        *indp = *nth;
      }
    };
    int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, sizes[dim]);
    iter.for_each(loop, /*grain_size=*/grain_size);
  });

  return std::forward_as_tuple(values, indices);
}

// Computes the median of all values in the input
Tensor median_impl(const Tensor& self, bool ignore_nan) {
  NoNamesGuard guard;
  const int64_t size = self.numel();

  // Return nan for empty tensors
  if (size <= 0) {
    return at::full({}, std::numeric_limits<float>::quiet_NaN()).to(self.options());
  }

  // Clone the input tensor so we can partition it around the median value
  Tensor in = self.clone();
  Tensor out = at::empty({}, self.options());

  AT_DISPATCH_ALL_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, in.scalar_type(), "median_cpu", [&] {
    scalar_t* op = out.data_ptr<scalar_t>();
    scalar_t* first = in.data_ptr<scalar_t>();
    scalar_t* last = first + size;

    // For torch.median, if there are nan values return nan
    if (!ignore_nan && std::any_of(first, last, _isnan<scalar_t>)) {
      *op = std::numeric_limits<scalar_t>::quiet_NaN();
      return;
    }

    scalar_t* median = first;
    if (!ignore_nan) {
      // If we got here, there are no nan values
      median += (size - 1) / 2;
      std::nth_element(first, median, last);
    } else {
      // For torch.nanmedian, compute median of non-nan values only
      int64_t num_nan = std::count_if(first, last, _isnan<scalar_t>);
      median += (size - num_nan - 1) / 2;
      std::nth_element(first, median, last, [](scalar_t a, scalar_t b) {
        return a < b || (_isnan(b) && !_isnan(a));
      });
    }

    *op = *median;
  });

  return out;
}

} // namespace

Tensor& quantile_out(
    const Tensor& self,
    const Tensor& q,
    std::optional<int64_t> dim,
    bool keepdim,
    const std::string_view interpolation,
    Tensor& out) {
  quantile_out_impl(
      out,
      self,
      q,
      dim,
      keepdim,
      get_quantile_interpolation_mode(interpolation),
      /*ignore_nan=*/false);
  return out;
}

Tensor& quantile_out(
    const Tensor& self,
    double q,
    std::optional<int64_t> dim,
    bool keepdim,
    const std::string_view interpolation,
    Tensor& out) {
  TORCH_CHECK(
      q >= 0 && q <= 1, "quantile() q must be in the range [0, 1] but got ", q);
  return at::native::quantile_out(
      self,
      at::scalar_tensor(q, self.options()),
      dim,
      keepdim,
      interpolation,
      out);
}

Tensor quantile(
    const Tensor& self,
    const Tensor& q,
    std::optional<int64_t> dim,
    bool keepdim,
    const std::string_view interpolation) {
  return quantile_impl(
      self,
      q,
      dim,
      keepdim,
      get_quantile_interpolation_mode(interpolation),
      /*ignore_nan=*/false);
}

Tensor quantile(
    const Tensor& self,
    double q,
    std::optional<int64_t> dim,
    bool keepdim,
    const std::string_view interpolation) {
  TORCH_CHECK(
      q >= 0 && q <= 1, "quantile() q must be in the range [0, 1] but got ", q);
  return at::native::quantile(
      self, at::scalar_tensor(q, self.options()), dim, keepdim, interpolation);
}

Tensor& nanquantile_out(
    const Tensor& self,
    const Tensor& q,
    std::optional<int64_t> dim,
    bool keepdim,
    const std::string_view interpolation,
    Tensor& out) {
  quantile_out_impl(
      out,
      self,
      q,
      dim,
      keepdim,
      get_quantile_interpolation_mode(interpolation),
      /*ignore_nan=*/true);
  return out;
}

Tensor& nanquantile_out(
    const Tensor& self,
    double q,
    std::optional<int64_t> dim,
    bool keepdim,
    const std::string_view interpolation,
    Tensor& out) {
  TORCH_CHECK(
      q >= 0 && q <= 1, "quantile() q must be in the range [0, 1] but got ", q);
  return at::native::nanquantile_out(
      self,
      at::scalar_tensor(q, self.options()),
      dim,
      keepdim,
      interpolation,
      out);
}

Tensor nanquantile(
    const Tensor& self,
    const Tensor& q,
    std::optional<int64_t> dim,
    bool keepdim,
    const std::string_view interpolation) {
  return quantile_impl(
      self,
      q,
      dim,
      keepdim,
      get_quantile_interpolation_mode(interpolation),
      /*ignore_nan=*/true);
}

Tensor nanquantile(
    const Tensor& self,
    double q,
    std::optional<int64_t> dim,
    bool keepdim,
    const std::string_view interpolation) {
  TORCH_CHECK(
      q >= 0 && q <= 1, "quantile() q must be in the range [0, 1] but got ", q);
  return at::native::nanquantile(
      self, at::scalar_tensor(q, self.options()), dim, keepdim, interpolation);
}

std::tuple<Tensor&, Tensor&> kthvalue_out_cpu(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  auto result = [&]() {
    NoNamesGuard guard;
    return kthvalue_out_impl_cpu(values, indices, self, k, dim, keepdim);
  }();
  namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);
  return result;
}

std::tuple<Tensor&, Tensor&> kthvalue_out(
    const Tensor& self,
    int64_t k,
    Dimname dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return at::kthvalue_out(
      values, indices, self, k, dimname_to_position(self, dim), keepdim);
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

std::tuple<Tensor, Tensor> kthvalue(
    const Tensor& self,
    int64_t k,
    Dimname dim,
    bool keepdim) {
  return at::kthvalue(self, k, dimname_to_position(self, dim), keepdim);
}

TORCH_IMPL_FUNC(topk_out_cpu)
   (const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted,
    const Tensor& values,
    const Tensor& indices) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");

  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
  } else {
    topk_stub(kCPU, values, indices, self, k, dim, largest, sorted);
  }
}

std::tuple<Tensor&, Tensor&> median_out_cpu(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  auto result = [&]() {
    NoNamesGuard guard;
    return median_with_indices_impl(
        values, indices, self, dim, keepdim, /*ignore_nan=*/false);
  }();
  namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);
  return result;
}

std::tuple<Tensor&, Tensor&> median_out(
    const Tensor& self,
    Dimname dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return at::median_out(
      values, indices, self, dimname_to_position(self, dim), keepdim);
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

std::tuple<Tensor, Tensor> median(
    const Tensor& self,
    Dimname dim,
    bool keepdim) {
  return at::median(self, dimname_to_position(self, dim), keepdim);
}

Tensor median_cpu(const Tensor& self) {
  return median_impl(self, /*ignore_nan=*/false);
}

std::tuple<Tensor&, Tensor&> nanmedian_out_cpu(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  auto result = [&]() {
    NoNamesGuard guard;
    return median_with_indices_impl(
        values, indices, self, dim, keepdim, /*ignore_nan=*/true);
  }();
  namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);
  return result;
}

std::tuple<Tensor&, Tensor&> nanmedian_out(
    const Tensor& self,
    Dimname dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return at::nanmedian_out(
      values, indices, self, dimname_to_position(self, dim), keepdim);
}

std::tuple<Tensor, Tensor> nanmedian(
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  at::nanmedian_out(values, indices, self, dim, keepdim);
  return std::make_tuple(values, indices);
}

std::tuple<Tensor, Tensor> nanmedian(
    const Tensor& self,
    Dimname dim,
    bool keepdim) {
  return at::nanmedian(self, dimname_to_position(self, dim), keepdim);
}

Tensor nanmedian_cpu(const Tensor& self) {
  return median_impl(self, /*ignore_nan=*/true);
}

TORCH_IMPL_FUNC(sort_stable_out)
(const Tensor& self,
 std::optional<bool> stable,
 int64_t dim,
 bool descending,
 const Tensor& values,
 const Tensor& indices) {
  values.copy_(self);
  // check if self is scalar
  if (self.dim() == 0 && self.numel() == 1) {
    indices.zero_();
  } else {
    dim = maybe_wrap_dim(dim, self.dim());
    sort_stub(self.device().type(), self, values, indices, dim, descending, stable.value_or(false));
  }
}

std::tuple<Tensor&, Tensor&> sort_out(
    const Tensor& self,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  return at::sort_out(values, indices, self, false, dim, descending);
}

std::tuple<Tensor, Tensor> sort(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  return at::sort(self, false, dim, descending);
}

Tensor& msort_out(const Tensor& self, Tensor& values) {
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  at::sort_out(values, indices, self, 0, false);
  return values;
}

Tensor msort(const Tensor& self) {
  return std::get<0>(at::sort(self, 0, false));
}

Tensor argsort(const Tensor & self, int64_t dim, bool descending) {
  return std::get<1>(at::sort(self, dim, descending));
}

Tensor argsort(const Tensor & self, bool stable, int64_t dim, bool descending) {
  return std::get<1>(at::sort(self, stable, dim, descending));
}

Tensor& argsort_out(const Tensor & self, bool stable, int64_t dim, bool descending, Tensor& out) {
  auto values = at::empty({0}, self.options());
  at::sort_outf(self, stable, dim, descending, values, out);
  return out;
}


} // namespace at::native
