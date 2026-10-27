#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ElementwiseRefMeta.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <ATen/EmptyTensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <c10/core/Contiguity.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/core/impl/PyInterpreterHooks.h>
#include <c10/util/StringUtil.h>
#include <c10/util/irange.h>
#include <c10/util/safe_numerics.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#endif

namespace at::native {

namespace {

// Tensor metadata as the Python refs see it. Wrapped numbers are Python
// numbers there and never take part in shape or stride logic.
struct MetaDesc {
  c10::SymDimVector sizes;
  c10::SymDimVector strides;
  ScalarType dtype = ScalarType::Undefined;
  Device device = kMeta;
  bool is_cpu_scalar_tensor = false;
  bool is_number = false;
  bool has_symbolic_sizes_strides = false;

  int64_t dim() const {
    return static_cast<int64_t>(sizes.size());
  }
};

bool is_nested_int(const c10::SymInt& s) {
  return s.is_heap_allocated() && s.toSymNodeImplUnowned()->is_nested_int();
}

bool any_heap(c10::SymIntArrayRef xs) {
  return std::ranges::any_of(xs, &c10::SymInt::is_heap_allocated);
}

bool desc_is_symbolic(const MetaDesc& d) {
  return any_heap(d.sizes) || any_heap(d.strides);
}

c10::SymInt sym_numel(c10::SymIntArrayRef sizes) {
  return std::accumulate(sizes.begin(), sizes.end(), c10::SymInt(1), std::multiplies<>());
}

// Python fake runs the ref under FakeTensorMode when the inputs are symbolic
// (fake devices visible, so cpu scalars are detected) and as the Meta kernel
// otherwise (every tensor reports device meta).
MetaDesc meta_desc(const Tensor& t, bool fake_devices) {
  MetaDesc d;
  d.sizes = c10::SymDimVector(t.sym_sizes().begin(), t.sym_sizes().end());
  d.strides = c10::SymDimVector(t.sym_strides().begin(), t.sym_strides().end());
  d.dtype = t.scalar_type();
  d.is_number = t.unsafeGetTensorImpl()->is_wrapped_number();
  const auto fake_device = t.unsafeGetTensorImpl()->fake_device();
  d.device = fake_device.has_value() ? (fake_devices ? *fake_device : Device(kMeta)) : t.device();
  d.is_cpu_scalar_tensor = !d.is_number && t.dim() == 0 && d.device.is_cpu();
  d.has_symbolic_sizes_strides = t.unsafeGetTensorImpl()->has_symbolic_sizes_strides();
  return d;
}

// Identical nodes fold to true, like sympy's Eq(x, x).
c10::SymBool sym_eq_folded(const c10::SymInt& a, const c10::SymInt& b) {
  if (a.is_heap_allocated() && b.is_heap_allocated() && a.toSymNodeImplUnowned() == b.toSymNodeImplUnowned()) {
    return c10::SymBool(true);
  }
  return a.sym_eq(b);
}

bool is_same_shape(c10::SymIntArrayRef a, c10::SymIntArrayRef b) {
  if (a.size() != b.size()) {
    return false;
  }
  c10::SymBool result(true);
  for (const auto i : c10::irange(a.size())) {
    result = result.sym_and(sym_eq_folded(a[i], b[i]));
  }
  return result.guard_bool(__FILE__, __LINE__);
}

void check_same_device(ArrayRef<MetaDesc> args) {
  const MetaDesc* first = nullptr;
  for (const auto& arg : args) {
    if (arg.is_number || arg.is_cpu_scalar_tensor) {
      continue;
    }
    if (first == nullptr) {
      first = &arg;
    }
    TORCH_CHECK(
        arg.device == first->device,
        "Tensor on device ", arg.device, " is not on the expected device ", first->device, "!");
  }
}

void check_same_shape(ArrayRef<MetaDesc> args) {
  const MetaDesc* first = nullptr;
  for (const auto& arg : args) {
    if (arg.is_number || arg.is_cpu_scalar_tensor) {
      continue;
    }
    if (first == nullptr) {
      first = &arg;
    }
    TORCH_CHECK(
        is_same_shape(first->sizes, arg.sizes),
        "Shape ", c10::SymIntArrayRef(arg.sizes), " is not the expected shape ", c10::SymIntArrayRef(first->sizes), "!");
  }
}

std::vector<const MetaDesc*> filter_tensors(ArrayRef<MetaDesc> args) {
  std::vector<const MetaDesc*> tensors;
  tensors.reserve(args.size());
  for (const auto& arg : args) {
    if (!arg.is_number && !arg.is_cpu_scalar_tensor) {
      tensors.push_back(&arg);
    }
  }
  return tensors;
}

bool is_contiguous_or_false(const MetaDesc& a) {
  if (TORCH_GUARD_OR_FALSE(sym_numel(a.sizes).sym_lt(2))) {
    return true;
  }
  return c10::_is_contiguous_or_false(a.sizes, a.strides);
}

// K.__lt__ in _prims_common, on strides only.
bool stride_lt(const c10::SymInt& s, const c10::SymInt& o) {
  return TORCH_GUARD_OR_FALSE(s.sym_lt(o)) ||
      ((TORCH_GUARD_OR_FALSE(s.sym_eq(0)) || TORCH_GUARD_OR_FALSE((o % s).sym_eq(0))) && TORCH_GUARD_OR_TRUE(s.sym_ne(o)));
}

// CPython <= 3.12's list.sort for n < 64 (count_run + binarysort), so the
// non-transitive K.__lt__ sees the same calls in the same order as sorted().
template <typename Lt>
void python_sort(std::vector<int64_t>& a, Lt lt) {
  const size_t n = a.size();
  if (n < 2) {
    return;
  }
  if (n >= 64) {
    std::stable_sort(a.begin(), a.end(), lt);
    return;
  }
  size_t run = 2;
  if (lt(a[1], a[0])) {
    while (run < n && lt(a[run], a[run - 1])) {
      ++run;
    }
    std::reverse(a.begin(), a.begin() + run);
  } else {
    while (run < n && !lt(a[run], a[run - 1])) {
      ++run;
    }
  }
  for (size_t start = run; start < n; ++start) {
    const int64_t pivot = a[start];
    size_t l = 0;
    size_t r = start;
    while (l < r) {
      const size_t p = std::midpoint(l, r);
      if (lt(pivot, a[p])) {
        r = p;
      } else {
        l = p + 1;
      }
    }
    std::move_backward(a.begin() + l, a.begin() + start, a.begin() + start + 1);
    a[l] = pivot;
  }
}

// _prims_common._is_non_overlapping_and_dense_or_false, whose unbacked semantics
// differ from TensorImpl::is_non_overlapping_and_dense_or_false. For unbacked
// sizes, TensorImpl evaluates IsNonOverlappingAndDenseIndicator(sizes, strides)
// == 1 (via SymbolicShapeMeta), which sympy only folds when every stride is an
// integer. Here the strides are sorted size-obliviously first (stride_lt proves
// e.g. u0 < 2 * u0), so this can return true where TensorImpl returns false,
// e.g. sizes (s0, u0) with strides (1, s0). Backed sizes give the same result,
// though the guards may be phrased differently.
// TODO: unify the unbacked semantics of the two.
bool is_non_overlapping_and_dense_or_false(const MetaDesc& a) {
  if (TORCH_GUARD_OR_FALSE(sym_numel(a.sizes).sym_lt(2))) {
    return true;
  }
  if (a.dim() == 1) {
    return TORCH_GUARD_OR_FALSE(a.strides[0].sym_eq(1));
  }
  std::vector<int64_t> order(a.dim());
  std::iota(order.begin(), order.end(), 0);
  python_sort(order, [&](int64_t i, int64_t j) { return stride_lt(a.strides[i], a.strides[j]); });
  c10::SymDimVector sorted_sizes;
  sorted_sizes.reserve(a.dim());
  c10::SymDimVector sorted_strides;
  sorted_strides.reserve(a.dim());
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    sorted_sizes.push_back(a.sizes[*it]);
    sorted_strides.push_back(a.strides[*it]);
  }
  return c10::_is_contiguous_or_false(sorted_sizes, sorted_strides);
}

// a >= b assuming a >= 0, b >= 0.
bool ge(const c10::SymInt& a, const c10::SymInt& b) {
  if (TORCH_GUARD_OR_FALSE(b.sym_eq(0))) {
    return true;
  } else if (TORCH_GUARD_OR_FALSE(a.sym_eq(0))) {
    return false;
  }
  return TORCH_GUARD_OR_FALSE(a.sym_ge(b)) || TORCH_GUARD_OR_FALSE((a % b).sym_eq(0));
}

// Takes the tensors left after the shape check and the cpu scalar filtering.
DimVector compute_elementwise_output_logical_to_physical_perm(const std::vector<const MetaDesc*>& tensors) {
  if (tensors.empty()) {
    return {};
  }
  const int64_t ndim = tensors[0]->dim();
  if (ndim == 0) {
    return {};
  }
  if (ndim == 1) {
    return DimVector{0};
  }

  bool is_contiguous = true;
  bool is_channels_last = true;
  for (const auto* t : tensors) {
    is_contiguous = is_contiguous && is_contiguous_or_false(*t);
    is_channels_last = is_channels_last && c10::_is_channels_last_contiguous_2d_or_false(t->sizes, t->strides);
  }

  DimVector perm(ndim);
  if (is_contiguous && !is_channels_last) {
    std::iota(perm.begin(), perm.end(), 0);
    return perm;
  }
  if (is_channels_last && !is_contiguous) {
    perm[0] = 0;
    std::iota(perm.begin() + 1, perm.end() - 1, 2);
    perm[ndim - 1] = 1;
    return perm;
  }

  const auto& shape = tensors[0]->sizes;
  auto should_swap = [&](int64_t idx_a, int64_t idx_b) -> int {
    for (const auto* t : tensors) {
      const auto& stride_a = t->strides[idx_a];
      const auto& stride_b = t->strides[idx_b];
      if (TORCH_GUARD_OR_FALSE(stride_a.sym_eq(0)) || TORCH_GUARD_OR_FALSE(stride_b.sym_eq(0))) {
        continue;
      }
      if (TORCH_GUARD_OR_FALSE(stride_a.sym_eq(stride_b))) {
        if (ge(shape[idx_b], shape[idx_a])) {
          continue;
        }
        return 1;
      }
      if (ge(stride_b, stride_a)) {
        return -1;
      }
      if (ge(stride_a, stride_b)) {
        return 1;
      }
    }
    return 0;
  };

  for (const auto i : c10::irange(ndim)) {
    perm[i] = ndim - 1 - i;
  }
  for (const auto i : c10::irange(1, ndim)) {
    int64_t dim1 = i;
    for (int64_t dim0 = i - 1; dim0 >= 0; --dim0) {
      const int comparison = should_swap(perm[dim0], perm[dim1]);
      if (comparison > 0) {
        std::swap(perm[dim0], perm[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }
  std::reverse(perm.begin(), perm.end());
  return perm;
}

// torch.empty_permuted: a contiguous allocation of the physical sizes, which
// runs the size checks, restrided to the logical order.
MetaDesc empty_permuted(c10::SymIntArrayRef shape, IntArrayRef l2p_perm, ScalarType dtype) {
  const int64_t dim = static_cast<int64_t>(shape.size());
  c10::SymDimVector phys_size(dim);
  for (const auto i : c10::irange(dim)) {
    phys_size[i] = shape[l2p_perm[i]];
  }
  const auto phys = at::native::empty_meta_symint(phys_size, dtype, std::nullopt, kMeta, std::nullopt, std::nullopt);
  const auto phys_strides = phys.sym_strides();
  MetaDesc out;
  out.sizes = c10::SymDimVector(shape.begin(), shape.end());
  out.strides = c10::SymDimVector(dim);
  for (const auto i : c10::irange(dim)) {
    out.strides[l2p_perm[i]] = phys_strides[i];
  }
  out.dtype = dtype;
  out.has_symbolic_sizes_strides = desc_is_symbolic(out);
  return out;
}

// Plain ints are their own hint; nested ints have none.
std::optional<int64_t> guarding_hint(const c10::SymInt& s) {
  return s.is_heap_allocated() ? s.toSymNodeImplUnowned()->guarding_hint() : s.as_int_unchecked();
}

bool backed_size_oblivious() {
  return (*c10::impl::getGlobalPyInterpreter())->backed_size_oblivious();
}

// torch._check's default message.
constexpr const char* kCheckFailedMsg =
    "Expected cond to be True, but got False. (Could this error message be improved? If so, please report an "
    "enhancement request to PyTorch.)";

c10::SymDimVector _broadcast_shapes(ArrayRef<c10::SymIntArrayRef> shapes) {
  size_t maxlen = 0;
  for (const auto& shape : shapes) {
    maxlen = std::max(maxlen, shape.size());
  }
  const int64_t common_len = static_cast<int64_t>(maxlen);
  c10::SymDimVector common_shape(maxlen, c10::SymInt(1));
  for (const auto arg_idx : c10::irange(shapes.size())) {
    const auto& shape = shapes[arg_idx];
    const int64_t len = static_cast<int64_t>(shape.size());
    for (int64_t idx = -1; idx >= -len; --idx) {
      const auto& s = shape[len + idx];
      auto& common = common_shape[common_len + idx];
      if (is_nested_int(s)) {
        if (is_nested_int(common) && TORCH_GUARD_OR_FALSE(s.sym_eq(common))) {
          continue;
        }
      } else {
        // Under backed_size_oblivious, specialize a size to 1 if broadcasting
        // is the only way to handle the example inputs. The checks are trivially
        // true between plain ints, so only read the flag when there's a SymInt.
        const bool backed_so = (s.is_heap_allocated() || common.is_heap_allocated()) && backed_size_oblivious();
        const auto s_hint = backed_so ? guarding_hint(s) : std::nullopt;
        const auto common_hint = backed_so ? guarding_hint(common) : std::nullopt;
        if (s_hint && common_hint) {
          if (*s_hint == 1 && *common_hint != 1) {
            TORCH_SYM_CHECK(s.sym_eq(1), kCheckFailedMsg);
          }
          if (*common_hint == 1 && *s_hint != 1) {
            TORCH_SYM_CHECK(common.sym_eq(1), kCheckFailedMsg);
          }
        }
        if (TORCH_GUARD_OR_FALSE(s.sym_eq(common))) {
          continue;
        }
      }

      if (TORCH_GUARD_OR_FALSE(common.sym_eq(1))) {
        TORCH_CHECK_VALUE(
            !s.sym_lt(0).guard_bool(__FILE__, __LINE__), "Attempting to broadcast a dimension with negative length!");
        common = s;
      }

      if (!is_nested_int(s) && TORCH_GUARD_OR_FALSE(s.sym_eq(1))) {
        continue;
      }
      TORCH_SYM_CHECK(
          sym_eq_folded(common, s),
          "Attempting to broadcast a dimension of length ", s, " at ", idx, "! Mismatching argument at index ", arg_idx,
          " had torch.Size([", c10::Join(", ", shape), "]); but expected shape should be broadcastable to ",
          c10::SymIntArrayRef(common_shape));
    }
  }
  return common_shape;
}

bool should_expand(c10::SymIntArrayRef a, c10::SymIntArrayRef b) {
  if (a.size() != b.size()) {
    return true;
  }
  for (const auto i : c10::irange(a.size())) {
    const auto& x = a[i];
    const auto& y = b[i];
    if (TORCH_GUARD_OR_FALSE(x.sym_ne(y))) {
      return true;
    }
    if (!TORCH_GUARD_OR_FALSE(x.sym_eq(1).sym_and(y.sym_eq(1))) && TORCH_GUARD_OR_FALSE(x.sym_eq(1).sym_or(y.sym_eq(1)))) {
      return true;
    }
    TORCH_SYM_CHECK(x.sym_eq(y), "sizes assumed to be the same due to unbacked broadcasting semantics");
  }
  return false;
}

MetaDesc _broadcast_in_dim_meta(const MetaDesc& a, c10::SymIntArrayRef shape, IntArrayRef broadcast_dimensions) {
  const int64_t ndim = a.dim();
  const int64_t out_ndim = static_cast<int64_t>(shape.size());
  for (const auto idx : c10::irange(ndim)) {
    const auto new_idx = broadcast_dimensions[idx];
    TORCH_SYM_CHECK(
        a.sizes[idx].sym_eq(1).sym_or(shape[new_idx].sym_eq(a.sizes[idx])),
        a.sizes[idx], " must be broadcastable to ", shape[new_idx]);
  }

  c10::SymDimVector new_strides;
  new_strides.reserve(out_ndim);
  int64_t original_idx = 0;
  for (const auto idx : c10::irange(out_ndim)) {
    if (std::find(broadcast_dimensions.begin(), broadcast_dimensions.end(), idx) != broadcast_dimensions.end()) {
      const auto& size = a.sizes[original_idx];
      if (TORCH_GUARD_OR_FALSE(size.sym_eq(1))) {
        new_strides.push_back(TORCH_GUARD_OR_FALSE(size.sym_eq(shape[idx])) ? a.strides[original_idx] : c10::SymInt(0));
      } else {
        TORCH_SYM_CHECK(size.sym_eq(shape[idx]), "non-broadcasting semantics require ", size, " == ", shape[idx]);
        new_strides.push_back(a.strides[original_idx]);
      }
      original_idx++;
    } else if (TORCH_GUARD_OR_TRUE(shape[idx].sym_ne(1))) {
      new_strides.push_back(c10::SymInt(0));
    } else if (original_idx == ndim) {
      new_strides.push_back(c10::SymInt(1));
    } else {
      new_strides.push_back(a.strides[original_idx] * a.sizes[original_idx]);
    }
  }

  MetaDesc out;
  out.sizes = c10::SymDimVector(shape.begin(), shape.end());
  out.strides = std::move(new_strides);
  out.dtype = a.dtype;
  out.device = a.device;
  out.is_cpu_scalar_tensor = a.is_cpu_scalar_tensor && shape.empty();
  out.has_symbolic_sizes_strides = desc_is_symbolic(out);
  return out;
}

MetaDesc expand(const MetaDesc& a, c10::SymIntArrayRef shape) {
  const int64_t ndim = a.dim();
  TORCH_CHECK(static_cast<int64_t>(shape.size()) >= ndim, "expand: the requested shape has too few dimensions!");
  const int64_t offset = static_cast<int64_t>(shape.size()) - ndim;
  c10::SymDimVector shape_(shape.begin(), shape.end());
  for (const auto idx : c10::irange(ndim)) {
    const auto& x = a.sizes[idx];
    const int64_t offset_idx = idx + offset;
    const auto& requested_length = shape[offset_idx];
    if (TORCH_GUARD_OR_FALSE(requested_length.sym_eq(-1))) {
      shape_[offset_idx] = x;
    } else {
      if ((x.is_heap_allocated() || requested_length.is_heap_allocated()) && backed_size_oblivious()) {
        const auto x_hint = guarding_hint(x);
        const auto requested_hint = guarding_hint(requested_length);
        if (x_hint == 1 && requested_hint && *requested_hint != 1) {
          TORCH_SYM_CHECK(x.sym_eq(1), kCheckFailedMsg);
        }
      }
      TORCH_SYM_CHECK(
          x.sym_eq(1).sym_or(requested_length.sym_eq(x)),
          "expand: attempting to expand a dimension of length ", x, " -> ", requested_length, "!");
      TORCH_SYM_CHECK(requested_length.sym_ge(0), "expand: expected a non-negative length, got ", requested_length);
      shape_[offset_idx] = requested_length;
    }
  }
  for (const auto& l : shape_) {
    TORCH_SYM_CHECK(l.sym_ge(0), "Expected a non-negative length, got ", l);
  }
  DimVector broadcast_dimensions(ndim);
  std::iota(broadcast_dimensions.begin(), broadcast_dimensions.end(), offset);
  return _broadcast_in_dim_meta(a, shape_, broadcast_dimensions);
}

// Tensor.expand's CompositeExplicitAutograd kernel, which Python fake runs
// when neither the tensor nor the size is symbolic.
MetaDesc aten_expand_desc(const MetaDesc& a, c10::SymIntArrayRef size) {
  TORCH_CHECK(
      size.size() >= static_cast<size_t>(a.dim()),
      "expand(size=", size, "): the number of sizes provided (", size.size(),
      ") must be greater or equal to the number of dimensions in the tensor (", a.dim(), ")");
  auto geometry = at::inferExpandGeometry_dimvector(
      c10::asIntArrayRefUnchecked(a.sizes), c10::asIntArrayRefUnchecked(a.strides), c10::asIntArrayRefUnchecked(size));
  // Raised by TensorImpl when the expanded view is created.
  uint64_t numel = 1;
  TORCH_CHECK(
      !c10::safe_multiplies_u64(geometry.sizes, &numel) &&
          numel <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "numel: integer multiplication overflow");
  MetaDesc out;
  out.sizes = c10::SymDimVector(geometry.sizes.begin(), geometry.sizes.end());
  out.strides = c10::SymDimVector(geometry.strides.begin(), geometry.strides.end());
  out.dtype = a.dtype;
  out.device = a.device;
  out.is_cpu_scalar_tensor = a.is_cpu_scalar_tensor && out.sizes.empty();
  return out;
}

std::vector<MetaDesc> _maybe_broadcast(ArrayRef<MetaDesc> args) {
  std::vector<MetaDesc> out(args.begin(), args.end());
  std::vector<c10::SymIntArrayRef> shapes;
  shapes.reserve(args.size());
  for (const auto& arg : args) {
    if (!arg.is_number) {
      shapes.emplace_back(arg.sizes);
    }
  }
  if (shapes.empty()) {
    return out;
  }
  const auto common_shape = _broadcast_shapes(shapes);
  // x.expand(common_shape) runs refs.expand only when x or the shape is
  // symbolic; otherwise it runs ATen's expand.
  const bool common_symbolic = any_heap(common_shape);
  for (auto& x : out) {
    if (x.is_number || x.is_cpu_scalar_tensor) {
      continue;
    }
    if (should_expand(x.sizes, common_shape)) {
      const bool symbolic = x.has_symbolic_sizes_strides || common_symbolic;
      x = symbolic ? expand(x, common_shape) : aten_expand_desc(x, common_shape);
    }
  }
  return out;
}

ScalarType get_computation_dtype(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::BFloat16:
    case ScalarType::Half:
      return ScalarType::Float;
    case ScalarType::ComplexHalf:
    case ScalarType::BComplex32:
      return ScalarType::ComplexFloat;
    default:
      return dtype;
  }
}

// utils.dtype_to_type, as a rank over bool < int < float < complex
int python_type_rank(ScalarType t) {
  if (t == kBool) {
    return 0;
  }
  if (isIntegralType(t, /*includeBool=*/false)) {
    return 1;
  }
  if (isFloatingType(t)) {
    return 2;
  }
  TORCH_CHECK_VALUE(isComplexType(t), "Invalid dtype!");
  return 3;
}

// utils.get_higher_dtype. Unlike promoteTypes it never rejects a pair: a dtype
// outside the ordering below (e.g. uint16 or float8) wins over any dtype in it.
ScalarType get_higher_dtype(std::optional<ScalarType> a, ScalarType b) {
  if (!a.has_value() || *a == b) {
    return b;
  }
  const auto order = [](ScalarType t) -> std::optional<int> {
    switch (t) {
      case kBool:
        return 0;
      case kByte:
      case kChar:
        return 1;
      case kShort:
        return 2;
      case kInt:
        return 3;
      case kLong:
        return 4;
      case kHalf:
      case kBFloat16:
        return 5;
      case kFloat:
        return 6;
      case kDouble:
        return 7;
      case kComplexHalf:
      case ScalarType::BComplex32:
        return 8;
      case kComplexFloat:
        return 9;
      case kComplexDouble:
        return 10;
      default:
        return std::nullopt;
    }
  };
  const auto oa = order(*a);
  const auto ob = order(b);
  TORCH_CHECK(oa.has_value() || ob.has_value(), "Unexpected termination!");
  if (!oa.has_value()) {
    return *a;
  }
  if (!ob.has_value()) {
    return b;
  }
  if (*oa == *ob) {
    static constexpr std::array<ScalarType, 10> next = {
        kByte, kShort, kInt, kLong, kHalf, kFloat, kDouble, kComplexHalf, kComplexFloat, kComplexDouble};
    return next[*oa];
  }
  return *oa < *ob ? b : *a;
}

// A non-dense tensor gets compute_elementwise_output_strides(a), which for one
// tensor of rank >= 2 is torch.empty_like(a) (refs.empty_like: empty_permuted
// with the l2p perm).
MetaDesc _convert_element_type_meta(const MetaDesc& a, ScalarType dtype) {
  MetaDesc out = a;
  out.dtype = dtype;
  if (!is_non_overlapping_and_dense_or_false(a)) {
    out.strides = a.dim() == 1
        ? c10::SymDimVector{c10::SymInt(1)}
        : empty_permuted(a.sizes, compute_elementwise_output_logical_to_physical_perm({&a}), dtype).strides;
  }
  out.has_symbolic_sizes_strides = desc_is_symbolic(out);
  return out;
}

// Tensors go through Tensor.to (the _to_copy decomposition), numbers through
// utils.dtype_to_type_ctor. number is the value of a number when known; bool(),
// sym_int() and complex() guard on a symbolic one, sym_float() does not.
MetaDesc _maybe_convert_to_dtype(
    const MetaDesc& a,
    ScalarType dtype,
    const std::optional<Scalar>& number = std::nullopt) {
  if (a.is_number) {
    const bool symbolic = number.has_value() && number->isSymbolic();
    MetaDesc out = a;
    if (dtype == kBool) {
      if (symbolic) {
        if (number->isSymInt()) {
          number->toSymInt().sym_ne(0).guard_bool(__FILE__, __LINE__);
        } else if (number->isSymFloat()) {
          number->toSymFloat().toSymNodeImplUnowned()->bool_();
        } else {
          number->toSymBool().guard_bool(__FILE__, __LINE__);
        }
      }
      out.dtype = kBool;
    } else if (isIntegralType(dtype, /*includeBool=*/false)) {
      if (symbolic && number->isSymBool()) {
        number->toSymBool().guard_bool(__FILE__, __LINE__);
      }
      out.dtype = kLong;
    } else if (isComplexType(dtype)) {
      if (symbolic) {
        TORCH_CHECK_TYPE(!number->isSymBool(), "complex() first argument must be a string or a number, not 'SymBool'");
        if (number->isSymInt()) {
          number->toSymInt().guard_int(__FILE__, __LINE__);
        } else {
          number->toSymFloat().guard_float(__FILE__, __LINE__);
        }
      }
      out.dtype = toComplexType(c10::get_default_dtype_as_scalartype());
    } else {
      out.dtype = c10::get_default_dtype_as_scalartype();
    }
    return out;
  }
  return a.dtype == dtype ? a : _convert_element_type_meta(a, dtype);
}

MetaDesc _prim_elementwise_meta(ArrayRef<MetaDesc> args, ScalarType dtype) {
  check_same_device(args);
  check_same_shape(args);
  const auto perm = compute_elementwise_output_logical_to_physical_perm(filter_tensors(args));

  // utils.extract_shape, whose mismatch case check_same_shape has ruled out.
  const MetaDesc* shape = nullptr;
  const MetaDesc* scalar_shape = nullptr;
  for (const auto& arg : args) {
    if (arg.is_number) {
      continue;
    }
    if (arg.is_cpu_scalar_tensor) {
      scalar_shape = &arg;
    } else if (shape == nullptr) {
      shape = &arg;
    }
  }

  if (shape == nullptr && scalar_shape == nullptr) {
    MetaDesc out;
    out.dtype = dtype;
    out.is_number = true;
    return out;
  }
  const auto& like = shape != nullptr ? *shape : *scalar_shape;
  auto out = empty_permuted(like.sizes, perm, dtype);
  out.device = like.device;
  out.is_cpu_scalar_tensor = shape == nullptr;
  return out;
}

// handle_noncontiguous_outputs checks the device of the first non-number
// input. Python fake keeps its device on the subclass, so read the backend key
// both fakes carry (Note [Fake Tensor Dispatch Keys]).
bool is_noncontiguous_supported(const Tensor& self, const Tensor& other) {
  const auto& first = self.unsafeGetTensorImpl()->is_wrapped_number() ? other : self;
  return !first.key_set().has_backend(BackendComponent::HPUBit);
}

// The SymInt, SymFloat or SymBool a symbolic wrapped number stands for, which
// the refs see as the Python number. Its placeholder value has the matching
// dtype: kLong, kDouble or kBool.
std::optional<Scalar> symbolic_number(const Tensor& t) {
  const auto* impl = t.unsafeGetTensorImpl();
  auto* node = impl->symbolic_wrapped_number();
  if (node == nullptr) {
    return std::nullopt;
  }
  auto sym_node = c10::SymNode::reclaim_copy(node);
  switch (impl->dtype().toScalarType()) {
    case kLong:
      return c10::SymInt(std::move(sym_node));
    case kDouble:
      return c10::SymFloat(std::move(sym_node));
    default:
      return c10::SymBool(std::move(sym_node));
  }
}

// Returns an empty tensor with the sizes, strides and dtype of an elementwise
// binary op's output. The operands are converted to the computation dtype for
// kind and broadcast to a common shape. If alpha is set, other is scaled by it;
// alpha must not be a wider number type than the computation dtype unless that
// is bool. The output strides follow the operands' memory layout, and the
// output dtype is the result dtype for kind.
// other_number is other's value when other wraps a Scalar argument.
Tensor binary_ref_meta_impl(
    const Tensor& self,
    const Tensor& other,
    const std::optional<Scalar>& other_number,
    ELEMENTWISE_TYPE_PROMOTION_KIND kind,
    bool fake_devices,
    const std::optional<Scalar>& alpha) {
  // A kernel running under the Meta key (e.g. a composite) can mix its own meta
  // tensors with fakes; every tensor then reports meta, as in Python fake.
  const auto real_meta = [](const Tensor& t) {
    return t.is_meta() && !t.unsafeGetTensorImpl()->fake_device().has_value();
  };
  fake_devices = fake_devices && !real_meta(self) && !real_meta(other);
  const auto [compute_dtype, result_dtype] = elementwise_dtypes(self, other, kind);
  auto args = _maybe_broadcast(
      {_maybe_convert_to_dtype(meta_desc(self, fake_devices), compute_dtype, symbolic_number(self)),
       _maybe_convert_to_dtype(
           meta_desc(other, fake_devices), compute_dtype, other_number ? other_number : symbolic_number(other))});
  if (alpha.has_value()) {
    // utils.is_weakly_lesser_type
    static constexpr std::array<const char*, 4> python_type_names = {
        "<class 'bool'>", "<class 'int'>", "<class 'float'>", "<class 'complex'>"};
    const auto rank = python_type_rank(compute_dtype);
    const auto alpha_rank = python_type_rank(alpha->type());
    TORCH_CHECK_VALUE(
        rank == 0 || alpha_rank <= rank,
        "alpha argument of type ", python_type_names[alpha_rank], " cannot be safely cast to type ",
        python_type_names[rank], "!");
    auto& b = args[1];
    if (!b.is_number) {
      MetaDesc alpha_desc;
      alpha_desc.dtype = alpha->type();
      alpha_desc.is_number = true;
      b = _prim_elementwise_meta({b, alpha_desc}, b.dtype);
    }
  }
  auto out = _prim_elementwise_meta(args, kind == ELEMENTWISE_TYPE_PROMOTION_KIND::ALWAYS_BOOL ? kBool : compute_dtype);
  if (!is_noncontiguous_supported(self, other) && !is_contiguous_or_false(out)) {
    DimVector identity(out.dim());
    std::iota(identity.begin(), identity.end(), 0);
    out.strides = empty_permuted(out.sizes, identity, out.dtype).strides;
  }
  if (out.dtype == result_dtype) {
    // torch.empty_permuted: contiguous physical allocation, then restrided
    auto result = at::detail::empty_symint_meta(out.sizes, out.dtype, std::nullopt, kMeta, std::nullopt, std::nullopt);
    result.unsafeGetTensorImpl()->set_sizes_and_strides(out.sizes, out.strides);
    return Tensor(std::move(result));
  }
  const auto converted = _convert_element_type_meta(out, result_dtype);
  return Tensor(at::detail::empty_strided_symint_meta(converted.sizes, converted.strides, result_dtype));
}

Tensor elementwise_binary_ref_meta_impl(
    const char* name,
    const Tensor& self,
    const Tensor& other,
    const std::optional<Scalar>& other_number,
    ELEMENTWISE_TYPE_PROMOTION_KIND kind,
    bool fake_devices,
    bool supports_lhs_python_scalar) {
  const bool self_number = self.unsafeGetTensorImpl()->is_wrapped_number();
  TORCH_CHECK_VALUE(
      supports_lhs_python_scalar || !self_number, name,
      ": Received a lhs Python scalar to an elementwise binary operation that does not accept lhs scalars!");
  TORCH_CHECK_VALUE(
      !self_number || !other.unsafeGetTensorImpl()->is_wrapped_number(), name,
      ": Receive two Number inputs to an elementwise binary operation!");
  return binary_ref_meta_impl(self, other, other_number, kind, fake_devices, std::nullopt);
}

// A Scalar argument as the Python number the refs see. The value is only
// passed along for the conversion guards.
Tensor python_number(const Scalar& s) {
  auto t = at::detail::scalar_tensor_static(s.isSymbolic() ? Scalar(0) : s, s.type(), kCPU);
  t.unsafeGetTensorImpl()->set_wrapped_number(true);
  return t;
}

} // namespace

// utils.elementwise_dtypes. Wrapped numbers are Python numbers there: they only
// pick the type kind, and never the dtype within it.
std::pair<ScalarType, ScalarType> elementwise_dtypes(
    const Tensor& a,
    const Tensor& b,
    ELEMENTWISE_TYPE_PROMOTION_KIND kind) {
  const std::array<const Tensor*, 2> args = {&a, &b};
  int highest_type = 0;
  for (const auto* x : args) {
    highest_type = std::max(highest_type, python_type_rank(x->scalar_type()));
  }
  // _find_highest_dtype_filtered: prefers the dtype of tensors with one or
  // more dimensions.
  const auto find_highest_dtype = [&](int min_rank, bool float_as_complex = false) -> std::optional<ScalarType> {
    std::optional<ScalarType> zero_dim_dtype;
    std::optional<ScalarType> one_plus_dim_dtype;
    for (const auto* x : args) {
      auto dtype = x->scalar_type();
      const auto rank = python_type_rank(dtype);
      if (x->unsafeGetTensorImpl()->is_wrapped_number() || rank < min_rank) {
        continue;
      }
      if (float_as_complex && rank == 2) {
        dtype = toComplexType(dtype);
      }
      auto& slot = x->dim() == 0 ? zero_dim_dtype : one_plus_dim_dtype;
      slot = get_higher_dtype(slot, dtype);
    }
    return one_plus_dim_dtype.has_value() ? one_plus_dim_dtype : zero_dim_dtype;
  };
  const auto default_dtype = c10::get_default_dtype_as_scalartype();
  ScalarType result_dtype = kBool;
  if (highest_type == 1) {
    result_dtype = find_highest_dtype(1).value_or(kLong);
  } else if (highest_type == 2) {
    result_dtype = find_highest_dtype(2).value_or(default_dtype);
  } else if (highest_type == 3) {
    result_dtype = find_highest_dtype(2, /*float_as_complex=*/true).value_or(toComplexType(default_dtype));
  }
  if (kind == ELEMENTWISE_TYPE_PROMOTION_KIND::INT_TO_FLOAT && isIntegralType(result_dtype, /*includeBool=*/true)) {
    result_dtype = default_dtype;
  }
  const auto out_dtype = kind == ELEMENTWISE_TYPE_PROMOTION_KIND::ALWAYS_BOOL ? kBool : result_dtype;
  return {get_computation_dtype(result_dtype), out_dtype};
}

Tensor binary_ref_meta(
    const Tensor& self,
    const Tensor& other,
    ELEMENTWISE_TYPE_PROMOTION_KIND kind,
    bool fake_devices,
    const std::optional<Scalar>& alpha) {
  return binary_ref_meta_impl(self, other, std::nullopt, kind, fake_devices, alpha);
}

Tensor elementwise_binary_ref_meta(
    const char* name,
    const Tensor& self,
    const Tensor& other,
    ELEMENTWISE_TYPE_PROMOTION_KIND kind,
    bool fake_devices,
    bool supports_lhs_python_scalar) {
  return elementwise_binary_ref_meta_impl(
      name, self, other, std::nullopt, kind, fake_devices, supports_lhs_python_scalar);
}

Tensor elementwise_binary_ref_meta(
    const char* name,
    const Tensor& self,
    const Scalar& other,
    ELEMENTWISE_TYPE_PROMOTION_KIND kind,
    bool fake_devices,
    bool supports_lhs_python_scalar) {
  return elementwise_binary_ref_meta_impl(
      name, self, python_number(other), other, kind, fake_devices, supports_lhs_python_scalar);
}

bool is_symbolic_operand(const Tensor& t) {
  const auto* impl = t.unsafeGetTensorImpl();
  if (impl->has_symbolic_sizes_strides()) {
    return true;
  }
  // Python fake counts a SymInt, which is wrapped as kLong, but not a SymFloat
  // or SymBool.
  return impl->is_symbolic_wrapped_number() && impl->dtype() == kLong;
}

void check_inplace_broadcast(c10::SymIntArrayRef self_shape, c10::SymIntArrayRef other_shape) {
  const auto shape = _broadcast_shapes({self_shape, other_shape});
  // tuple(shape) == self_shape, which torch.Size reflects to self_shape[i] ==
  // shape[i]: elements up to the shorter length, stopping at the first
  // mismatch, then the lengths.
  bool same = true;
  for (size_t i = 0; same && i < std::min(shape.size(), self_shape.size()); ++i) {
    same = sym_eq_folded(self_shape[i], shape[i]).guard_bool(__FILE__, __LINE__);
  }
  same = same && shape.size() == self_shape.size();
  TORCH_CHECK(
      same, "output with shape torch.Size([", c10::Join(", ", self_shape), "]) doesn't match the broadcast shape (",
      c10::Join(", ", shape), shape.size() == 1 ? ",)" : ")");
}

// Python fake tries this first when the inputs are symbolic. Returns an
// undefined tensor where it falls back to the ref (the ref then raises for
// mismatched devices). The output device is left to the caller.
Tensor fast_binary_impl(const Tensor& self, const Tensor& other, ELEMENTWISE_TYPE_PROMOTION_KIND kind) {
  const std::array<const Tensor*, 2> operands = {&self, &other};
  c10::SymDimVector final_shape(self.sym_sizes().begin(), self.sym_sizes().end());
  for (const auto* op : operands) {
    final_shape = at::infer_size_symdimvector(final_shape, op->sym_sizes());
  }

  bool obvious = false;
  for (const auto* op : operands) {
    if (op->unsafeGetTensorImpl()->is_wrapped_number() || op->dim() != static_cast<int64_t>(final_shape.size())) {
      continue;
    }
    const auto sizes = op->sym_sizes();
    c10::SymBool eq(true);
    for (const auto i : c10::irange(final_shape.size())) {
      eq = eq.sym_and(sizes[i].sym_eq(final_shape[i]));
    }
    if (TORCH_GUARD_OR_FALSE(eq)) {
      obvious = true;
      break;
    }
  }
  if (!obvious) {
    return {};
  }

  const bool self_number = self.unsafeGetTensorImpl()->is_wrapped_number();
  const bool other_number = other.unsafeGetTensorImpl()->is_wrapped_number();
  auto dtype = self.scalar_type();
  if (kind != ELEMENTWISE_TYPE_PROMOTION_KIND::DEFAULT || self_number || other_number || other.scalar_type() != dtype) {
    dtype = elementwise_dtypes(self, other, kind).second;
  }

  c10::SmallVector<MetaDesc, 2> descs;
  for (const auto* op : operands) {
    if (!op->unsafeGetTensorImpl()->is_wrapped_number()) {
      descs.push_back(meta_desc(*op, /*fake_devices=*/true));
    }
  }
  // With two operands at most one can be the allowed CPU scalar.
  Device common_device = kCPU;
  for (const auto& desc : descs) {
    if (common_device.is_cpu() && !desc.device.is_cpu()) {
      common_device = desc.device;
    }
  }
  for (const auto& desc : descs) {
    const bool cpu_scalar_on_non_cpu = !common_device.is_cpu() && desc.dim() == 0 && desc.device == Device(kCPU);
    if (!cpu_scalar_on_non_cpu && desc.device != common_device) {
      return {};
    }
  }

  bool contiguous = true;
  bool channels_last = true;
  // fake_impls.is_noncontiguous_supported; HPU outputs stay contiguous.
  if (common_device.type() != kHPU) {
    for (const auto& desc : descs) {
      contiguous = contiguous && is_contiguous_or_false(desc);
      channels_last = channels_last && c10::_is_channels_last_contiguous_2d_or_false(desc.sizes, desc.strides);
    }
  }
  if (!contiguous && !channels_last) {
    return {};
  }
  return at::native::empty_meta_symint(
      final_shape, dtype, std::nullopt, kMeta, std::nullopt,
      contiguous ? MemoryFormat::Contiguous : MemoryFormat::ChannelsLast);
}

} // namespace at::native
