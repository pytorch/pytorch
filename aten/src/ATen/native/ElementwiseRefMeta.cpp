#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ElementwiseRefMeta.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <utility>
#include <vector>

#include <ATen/EmptyTensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/TypeProperties.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/util/StringUtil.h>
#include <c10/util/irange.h>

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
  return std::any_of(xs.begin(), xs.end(), [](const c10::SymInt& s) { return s.is_heap_allocated(); });
}

bool desc_is_symbolic(const MetaDesc& d) {
  return any_heap(d.sizes) || any_heap(d.strides);
}

c10::SymInt sym_numel(c10::SymIntArrayRef sizes) {
  c10::SymInt n = 1;
  for (const auto& s : sizes) {
    n = n * s;
  }
  return n;
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
  for (const auto& arg : args) {
    if (!arg.is_number && !arg.is_cpu_scalar_tensor) {
      tensors.push_back(&arg);
    }
  }
  return tensors;
}

// With false_if_dde=True.
bool check_contiguous_sizes_strides(c10::SymIntArrayRef sizes, c10::SymIntArrayRef strides) {
  c10::SymInt expected_stride = 1;
  c10::SymInt expected_stride_max = 1;
  for (int64_t i = static_cast<int64_t>(std::min(sizes.size(), strides.size())) - 1; i >= 0; --i) {
    const auto& x = sizes[i];
    const auto& y = strides[i];
    if (TORCH_GUARD_OR_FALSE(x.sym_eq(1))) {
      continue;
    }
    if (TORCH_GUARD_OR_TRUE(y.sym_ne(expected_stride)) && TORCH_GUARD_OR_TRUE(y.sym_ne(expected_stride_max))) {
      return false;
    }
    expected_stride_max = expected_stride_max * (is_nested_int(x) ? x : x.max(1));
    expected_stride = expected_stride * x;
  }
  return true;
}

bool is_contiguous_or_false(const MetaDesc& a) {
  if (TORCH_GUARD_OR_FALSE(sym_numel(a.sizes).sym_lt(2))) {
    return true;
  }
  return check_contiguous_sizes_strides(a.sizes, a.strides);
}

bool is_channels_last_contiguous_or_false_2d(const MetaDesc& a) {
  if (a.dim() != 4) {
    return false;
  }
  c10::SymInt expected_stride = 1;
  for (const int64_t idx : {1, 3, 2, 0}) {
    const auto& length = a.sizes[idx];
    if (TORCH_GUARD_OR_FALSE(length.sym_eq(1))) {
      continue;
    }
    if (TORCH_GUARD_OR_TRUE(a.strides[idx].sym_ne(expected_stride))) {
      return false;
    }
    expected_stride = expected_stride * length;
  }
  return true;
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
      const size_t p = l + ((r - l) >> 1);
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
  c10::SymDimVector sorted_strides;
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    sorted_sizes.push_back(a.sizes[*it]);
    sorted_strides.push_back(a.strides[*it]);
  }
  return check_contiguous_sizes_strides(sorted_sizes, sorted_strides);
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
    is_channels_last = is_channels_last && is_channels_last_contiguous_or_false_2d(*t);
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

MetaDesc empty_permuted(c10::SymIntArrayRef shape, IntArrayRef l2p_perm, ScalarType dtype) {
  const int64_t dim = static_cast<int64_t>(shape.size());
  c10::SymDimVector phys_size(dim);
  for (const auto i : c10::irange(dim)) {
    phys_size[i] = shape[l2p_perm[i]];
  }
  // Contiguous strides as computed by empty_tensor_restride_symint.
  c10::SymDimVector phys_strides(dim);
  if (dim > 0) {
    phys_strides[dim - 1] = c10::SymInt(1);
    for (int64_t i = dim - 2; i >= 0; --i) {
      phys_strides[i] = phys_strides[i + 1] * phys_size[i + 1].max(1);
    }
  }
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
      } else if (TORCH_GUARD_OR_FALSE(s.sym_eq(common))) {
        continue;
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
          " had ", shape, "; but expected shape should be broadcastable to ", c10::SymIntArrayRef(common_shape));
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
      return ScalarType::ComplexFloat;
    default:
      return dtype;
  }
}

std::pair<ScalarType, ScalarType> elementwise_dtypes(
    const Tensor& a,
    const Tensor& b,
    ELEMENTWISE_TYPE_PROMOTION_KIND kind) {
  auto result_dtype = result_type(update_result_type_state(b, update_result_type_state(a, ResultTypeState{})));
  if (kind == ELEMENTWISE_TYPE_PROMOTION_KIND::INT_TO_FLOAT && isIntegralType(result_dtype, /*includeBool=*/true)) {
    result_dtype = c10::get_default_dtype_as_scalartype();
  }
  const auto out_dtype = kind == ELEMENTWISE_TYPE_PROMOTION_KIND::ALWAYS_BOOL ? kBool : result_dtype;
  return {get_computation_dtype(result_dtype), out_dtype};
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
// utils.dtype_to_type_ctor.
MetaDesc _maybe_convert_to_dtype(const MetaDesc& a, ScalarType dtype) {
  if (a.is_number) {
    MetaDesc out = a;
    if (dtype == kBool) {
      out.dtype = kBool;
    } else if (isIntegralType(dtype, /*includeBool=*/false)) {
      out.dtype = kLong;
    } else if (isComplexType(dtype)) {
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

  // utils.extract_shape
  const MetaDesc* shape = nullptr;
  const MetaDesc* scalar_shape = nullptr;
  bool shape_mismatch = false;
  for (const auto& arg : args) {
    if (arg.is_number) {
      continue;
    }
    if (arg.is_cpu_scalar_tensor) {
      scalar_shape = &arg;
      continue;
    }
    if (shape == nullptr) {
      shape = &arg;
    }
    if (!is_same_shape(shape->sizes, arg.sizes)) {
      shape_mismatch = true;
      break;
    }
  }

  if (shape == nullptr && scalar_shape == nullptr) {
    MetaDesc out;
    out.dtype = dtype;
    out.is_number = true;
    return out;
  }
  TORCH_CHECK(!shape_mismatch, "shape must not be None when device is not None");
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

// alpha != 1 in Python
bool python_ne_one(const Scalar& s) {
  if (s.isSymInt()) {
    return s.toSymInt().sym_ne(1).guard_bool(__FILE__, __LINE__);
  }
  if (s.isSymFloat()) {
    return s.toSymFloat().sym_ne(1.0).guard_bool(__FILE__, __LINE__);
  }
  if (s.isSymBool()) {
    // Python evaluates SymBool != 1 to True without guarding.
    return true;
  }
  return s.isComplex() ? s.toComplexDouble() != c10::complex<double>(1, 0) : s.toDouble() != 1;
}

// Unlike at::infer_size_symdimvector, compares sizeA == sizeB in Python's
// operand order.
c10::SymDimVector infer_size(c10::SymIntArrayRef a, c10::SymIntArrayRef b) {
  const auto dims_a = static_cast<int64_t>(a.size());
  const auto dims_b = static_cast<int64_t>(b.size());
  const auto ndim = std::max(dims_a, dims_b);
  c10::SymDimVector expanded_sizes(ndim);
  for (int64_t i = ndim - 1; i >= 0; --i) {
    const int64_t offset = ndim - 1 - i;
    const int64_t dim_a = dims_a - 1 - offset;
    const int64_t dim_b = dims_b - 1 - offset;
    const c10::SymInt size_a = dim_a >= 0 ? a[dim_a] : c10::SymInt(1);
    const c10::SymInt size_b = dim_b >= 0 ? b[dim_b] : c10::SymInt(1);
    if (!TORCH_GUARD_OR_FALSE(size_a.sym_eq(1)) && !TORCH_GUARD_OR_FALSE(size_b.sym_eq(1))) {
      TORCH_SYM_CHECK(
          size_a.sym_eq(size_b),
          "The size of tensor a (", size_a, ") must match the size of tensor b (", size_b,
          ") at non-singleton dimension ", i);
    }
    expanded_sizes[i] = TORCH_GUARD_OR_FALSE(size_a.sym_eq(1)) ? size_b : size_a;
  }
  return expanded_sizes;
}

} // namespace

// _make_elementwise_binary_reference / refs.add (alpha is None when unset) /
// refs.sub: elementwise_type_promotion_wrapper -> _maybe_broadcast ->
// [prims.mul(b, alpha)] -> prim -> conversion to the result dtype.
Tensor binary_ref_meta(
    const Tensor& self,
    const Tensor& other,
    ELEMENTWISE_TYPE_PROMOTION_KIND kind,
    bool fake_devices,
    const std::optional<Scalar>& alpha,
    bool is_sub) {
  // A kernel running under the Meta key (e.g. a composite) can mix its own meta
  // tensors with fakes; every tensor then reports meta, as in Python fake.
  const auto real_meta = [](const Tensor& t) {
    return t.is_meta() && !t.unsafeGetTensorImpl()->fake_device().has_value();
  };
  fake_devices = fake_devices && !real_meta(self) && !real_meta(other);
  const auto [compute_dtype, result_dtype] = elementwise_dtypes(self, other, kind);
  auto args = _maybe_broadcast(
      {_maybe_convert_to_dtype(meta_desc(self, fake_devices), compute_dtype),
       _maybe_convert_to_dtype(meta_desc(other, fake_devices), compute_dtype)});
  if (is_sub) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        args[0].is_number || args[1].is_number || (args[0].dtype != kBool && args[1].dtype != kBool),
        "Subtraction, the `-` operator, with two bool tensors is not supported. "
        "Use the `^` or `logical_xor()` operator instead.");
  }
  // refs.sub applies alpha when alpha != 1, after broadcasting (the check may
  // guard), and has no bool exemption in the type check below.
  if (alpha.has_value() && (!is_sub || python_ne_one(*alpha))) {
    // utils.is_weakly_lesser_type over bool < int < float < complex
    auto python_type_rank = [](ScalarType t) {
      return t == kBool ? 0 : isIntegralType(t, /*includeBool=*/false) ? 1 : isFloatingType(t) ? 2 : 3;
    };
    static constexpr std::array<const char*, 4> python_type_names = {
        "<class 'bool'>", "<class 'int'>", "<class 'float'>", "<class 'complex'>"};
    const auto rank = python_type_rank(compute_dtype);
    const auto alpha_rank = python_type_rank(alpha->type());
    TORCH_CHECK_VALUE(
        (rank == 0 && !is_sub) || alpha_rank <= rank,
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

Tensor elementwise_binary_ref_meta(
    const char* name,
    const Tensor& self,
    const Tensor& other,
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
  return binary_ref_meta(self, other, kind, fake_devices);
}

Tensor python_number(const Scalar& s) {
  auto t = at::detail::scalar_tensor_static(s.isSymbolic() ? Scalar(0) : s, s.type(), kCPU);
  t.unsafeGetTensorImpl()->set_wrapped_number(true);
  return t;
}

void check_inplace_broadcast(c10::SymIntArrayRef self_shape, c10::SymIntArrayRef other_shape) {
  const auto shape = _broadcast_shapes({self_shape, other_shape});
  // tuple(shape) == self_shape, which stops at the first mismatch
  bool same = shape.size() == self_shape.size();
  for (size_t i = 0; same && i < shape.size(); ++i) {
    same = sym_eq_folded(shape[i], self_shape[i]).guard_bool(__FILE__, __LINE__);
  }
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
    final_shape = infer_size(final_shape, op->sym_sizes());
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
      channels_last = channels_last && is_channels_last_contiguous_or_false_2d(desc);
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
