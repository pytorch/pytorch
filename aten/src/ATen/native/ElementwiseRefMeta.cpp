#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ElementwiseRefMeta.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <utility>
#include <vector>

#include <ATen/EmptyTensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/TypeProperties.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/SymNodeImpl.h>
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
  // utils.is_cpu_scalar_tensor
  bool cpu_scalar = false;
  bool is_number = false;
  // has_symbolic_sizes_strides; picks refs.expand over ATen's expand
  bool symbolic = false;

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
  d.cpu_scalar = !d.is_number && t.dim() == 0 && d.device.is_cpu();
  d.symbolic = t.unsafeGetTensorImpl()->has_symbolic_sizes_strides();
  return d;
}

// Python evaluates `a <op> b` with an int a and a SymInt b as b's reflected
// op, so the recorded guard is Eq(s0, 8) rather than Eq(8, s0).
bool reflects(const c10::SymInt& a, const c10::SymInt& b) {
  return !a.is_symbolic() && b.is_symbolic();
}

c10::SymBool py_eq(const c10::SymInt& a, const c10::SymInt& b) {
  return reflects(a, b) ? b.sym_eq(a) : a.sym_eq(b);
}

c10::SymBool py_ne(const c10::SymInt& a, const c10::SymInt& b) {
  return reflects(a, b) ? b.sym_ne(a) : a.sym_ne(b);
}

c10::SymBool py_lt(const c10::SymInt& a, const c10::SymInt& b) {
  return reflects(a, b) ? b.sym_gt(a) : a.sym_lt(b);
}

c10::SymBool py_ge(const c10::SymInt& a, const c10::SymInt& b) {
  return reflects(a, b) ? b.sym_le(a) : a.sym_ge(b);
}

// Identical nodes fold to true, like sympy's Eq(x, x).
c10::SymBool sym_eq_folded(const c10::SymInt& a, const c10::SymInt& b) {
  if (a.is_heap_allocated() && b.is_heap_allocated() && a.toSymNodeImplUnowned() == b.toSymNodeImplUnowned()) {
    return c10::SymBool(true);
  }
  return py_eq(a, b);
}

// bool(utils.is_same_shape(a, b))
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

// utils.check_same_device(*args, allow_cpu_scalar_tensors=True)
void check_same_device(ArrayRef<MetaDesc> args) {
  const MetaDesc* first = nullptr;
  for (const auto& arg : args) {
    if (arg.is_number || arg.cpu_scalar) {
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

// utils.check_same_shape(*args, allow_cpu_scalar_tensors=True)
void check_same_shape(ArrayRef<MetaDesc> args) {
  const MetaDesc* first = nullptr;
  for (const auto& arg : args) {
    if (arg.is_number || arg.cpu_scalar) {
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
    if (!arg.is_number && !arg.cpu_scalar) {
      tensors.push_back(&arg);
    }
  }
  return tensors;
}

// check_contiguous_sizes_strides(sizes, strides, false_if_dde=True)
bool check_contiguous_sizes_strides_or_false(c10::SymIntArrayRef sizes, c10::SymIntArrayRef strides) {
  c10::SymInt expected_stride = 1;
  c10::SymInt expected_stride_max = 1;
  for (int64_t i = static_cast<int64_t>(std::min(sizes.size(), strides.size())) - 1; i >= 0; --i) {
    const auto& x = sizes[i];
    const auto& y = strides[i];
    if (TORCH_GUARD_OR_FALSE(x.sym_eq(1))) {
      continue;
    }
    if (TORCH_GUARD_OR_TRUE(py_ne(y, expected_stride)) && TORCH_GUARD_OR_TRUE(py_ne(y, expected_stride_max))) {
      return false;
    }
    expected_stride_max = expected_stride_max * (is_nested_int(x) ? x : x.max(1));
    expected_stride = expected_stride * x;
  }
  return true;
}

// utils.is_contiguous_or_false
bool is_contiguous_or_false(const MetaDesc& a) {
  if (TORCH_GUARD_OR_FALSE(sym_numel(a.sizes).sym_lt(2))) {
    return true;
  }
  return check_contiguous_sizes_strides_or_false(a.sizes, a.strides);
}

// utils.is_channels_last_contiguous_or_false_2d
bool is_channels_last_contiguous_or_false(const MetaDesc& a) {
  if (a.dim() != 4) {
    return false;
  }
  c10::SymInt expected_stride = 1;
  for (const int64_t idx : {1, 3, 2, 0}) {
    const auto& length = a.sizes[idx];
    if (TORCH_GUARD_OR_FALSE(length.sym_eq(1))) {
      continue;
    }
    if (TORCH_GUARD_OR_TRUE(py_ne(a.strides[idx], expected_stride))) {
      return false;
    }
    expected_stride = expected_stride * length;
  }
  return true;
}

// K.__lt__ in _prims_common, on strides only.
bool stride_lt(const c10::SymInt& s, const c10::SymInt& o) {
  return TORCH_GUARD_OR_FALSE(py_lt(s, o)) ||
      ((TORCH_GUARD_OR_FALSE(s.sym_eq(0)) || TORCH_GUARD_OR_FALSE((o % s).sym_eq(0))) && TORCH_GUARD_OR_TRUE(py_ne(s, o)));
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

// utils.is_non_overlapping_and_dense_or_false
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
  return check_contiguous_sizes_strides_or_false(sorted_sizes, sorted_strides);
}

// ge() inside should_swap: a >= b assuming a >= 0, b >= 0.
bool stride_ge(const c10::SymInt& a, const c10::SymInt& b) {
  if (TORCH_GUARD_OR_FALSE(b.sym_eq(0))) {
    return true;
  } else if (TORCH_GUARD_OR_FALSE(a.sym_eq(0))) {
    return false;
  }
  return TORCH_GUARD_OR_FALSE(py_ge(a, b)) || TORCH_GUARD_OR_FALSE((a % b).sym_eq(0));
}

// utils.compute_elementwise_output_logical_to_physical_perm after the shape
// check and the cpu scalar filtering.
DimVector l2p_perm(const std::vector<const MetaDesc*>& tensors) {
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
    is_channels_last = is_channels_last && is_channels_last_contiguous_or_false(*t);
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
      if (TORCH_GUARD_OR_FALSE(py_eq(stride_a, stride_b))) {
        if (stride_ge(shape[idx_b], shape[idx_a])) {
          continue;
        }
        return 1;
      }
      if (stride_ge(stride_b, stride_a)) {
        return -1;
      }
      if (stride_ge(stride_a, stride_b)) {
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

// torch.empty_permuted(shape, l2p_perm), i.e. empty_permuted_symint
MetaDesc empty_permuted_desc(c10::SymIntArrayRef shape, IntArrayRef l2p_perm, ScalarType dtype) {
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
  out.symbolic = desc_is_symbolic(out);
  return out;
}

// refs._broadcast_shapes
c10::SymDimVector broadcast_shapes(ArrayRef<c10::SymIntArrayRef> shapes) {
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
        if (is_nested_int(common) && TORCH_GUARD_OR_FALSE(py_eq(s, common))) {
          continue;
        }
      } else if (TORCH_GUARD_OR_FALSE(py_eq(s, common))) {
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

// should_expand inside refs._maybe_broadcast
bool should_expand(c10::SymIntArrayRef a, c10::SymIntArrayRef b) {
  if (a.size() != b.size()) {
    return true;
  }
  for (const auto i : c10::irange(a.size())) {
    const auto& x = a[i];
    const auto& y = b[i];
    if (TORCH_GUARD_OR_FALSE(py_ne(x, y))) {
      return true;
    }
    if (!TORCH_GUARD_OR_FALSE(x.sym_eq(1).sym_and(y.sym_eq(1))) && TORCH_GUARD_OR_FALSE(x.sym_eq(1).sym_or(y.sym_eq(1)))) {
      return true;
    }
    TORCH_SYM_CHECK(py_eq(x, y), "sizes assumed to be the same due to unbacked broadcasting semantics");
  }
  return false;
}

// prims.broadcast_in_dim meta
MetaDesc broadcast_in_dim_desc(const MetaDesc& a, c10::SymIntArrayRef shape, IntArrayRef broadcast_dimensions) {
  const int64_t ndim = a.dim();
  const int64_t out_ndim = static_cast<int64_t>(shape.size());
  for (const auto idx : c10::irange(ndim)) {
    const auto new_idx = broadcast_dimensions[idx];
    TORCH_SYM_CHECK(
        a.sizes[idx].sym_eq(1).sym_or(py_eq(shape[new_idx], a.sizes[idx])),
        a.sizes[idx], " must be broadcastable to ", shape[new_idx]);
  }

  c10::SymDimVector new_strides;
  new_strides.reserve(out_ndim);
  int64_t original_idx = 0;
  for (const auto idx : c10::irange(out_ndim)) {
    if (std::find(broadcast_dimensions.begin(), broadcast_dimensions.end(), idx) != broadcast_dimensions.end()) {
      const auto& size = a.sizes[original_idx];
      if (TORCH_GUARD_OR_FALSE(size.sym_eq(1))) {
        new_strides.push_back(TORCH_GUARD_OR_FALSE(py_eq(size, shape[idx])) ? a.strides[original_idx] : c10::SymInt(0));
      } else {
        TORCH_SYM_CHECK(py_eq(size, shape[idx]), "non-broadcasting semantics require ", size, " == ", shape[idx]);
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
  out.cpu_scalar = a.cpu_scalar && shape.empty();
  out.symbolic = desc_is_symbolic(out);
  return out;
}

// refs.expand(a, shape) lowering to prims.broadcast_in_dim
MetaDesc expand_desc(const MetaDesc& a, c10::SymIntArrayRef shape) {
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
          x.sym_eq(1).sym_or(py_eq(requested_length, x)),
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
  return broadcast_in_dim_desc(a, shape_, broadcast_dimensions);
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
  out.cpu_scalar = a.cpu_scalar && out.sizes.empty();
  return out;
}

// refs._maybe_broadcast(*args, preserve_cpu_scalar_tensors=True)
std::vector<MetaDesc> maybe_broadcast(ArrayRef<MetaDesc> args) {
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
  const auto common_shape = broadcast_shapes(shapes);
  // x.expand(common_shape) runs refs.expand only when x or the shape is
  // symbolic; otherwise it runs ATen's expand.
  const bool common_symbolic = any_heap(common_shape);
  for (auto& x : out) {
    if (x.is_number || x.cpu_scalar) {
      continue;
    }
    if (should_expand(x.sizes, common_shape)) {
      x = (x.symbolic || common_symbolic) ? expand_desc(x, common_shape) : aten_expand_desc(x, common_shape);
    }
  }
  return out;
}

// utils.get_computation_dtype
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

// utils.elementwise_dtypes -> (computation dtype, result dtype); wrapped
// numbers promote as Python numbers.
std::pair<ScalarType, ScalarType> elementwise_dtypes(const Tensor& a, const Tensor& b, TypePromotionKind kind) {
  auto result_dtype = result_type(update_result_type_state(b, update_result_type_state(a, ResultTypeState{})));
  if (kind == TypePromotionKind::INT_TO_FLOAT && isIntegralType(result_dtype, /*includeBool=*/true)) {
    result_dtype = c10::get_default_dtype_as_scalartype();
  }
  return {get_computation_dtype(result_dtype), result_dtype};
}

// prims.convert_element_type meta. A non-dense tensor gets
// compute_elementwise_output_strides(a), which for one tensor of rank >= 2 is
// torch.empty_like(a) (refs.empty_like: empty_permuted with the l2p perm).
MetaDesc convert_element_type_desc(const MetaDesc& a, ScalarType dtype) {
  MetaDesc out = a;
  out.dtype = dtype;
  if (!is_non_overlapping_and_dense_or_false(a)) {
    out.strides = a.dim() == 1 ? c10::SymDimVector{c10::SymInt(1)} : empty_permuted_desc(a.sizes, l2p_perm({&a}), dtype).strides;
  }
  out.symbolic = desc_is_symbolic(out);
  return out;
}

// _maybe_convert_to_dtype: tensors go through Tensor.to (the _to_copy
// decomposition), numbers through utils.dtype_to_type_ctor.
MetaDesc maybe_convert_desc(const MetaDesc& a, ScalarType dtype) {
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
  return a.dtype == dtype ? a : convert_element_type_desc(a, dtype);
}

// _prim_elementwise_meta over already broadcast args
MetaDesc prim_elementwise_desc(ArrayRef<MetaDesc> args, ScalarType dtype) {
  check_same_device(args);
  check_same_shape(args);
  const auto perm = l2p_perm(filter_tensors(args));

  // utils.extract_shape
  const MetaDesc* shape = nullptr;
  const MetaDesc* scalar_shape = nullptr;
  bool shape_mismatch = false;
  for (const auto& arg : args) {
    if (arg.is_number) {
      continue;
    }
    if (arg.cpu_scalar) {
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
  auto out = empty_permuted_desc(like.sizes, perm, dtype);
  out.device = like.device;
  out.cpu_scalar = shape == nullptr;
  return out;
}

// refs.is_noncontiguous_supported for the device handle_noncontiguous_outputs
// picks: that of the first fake input. Wrapped numbers are Python numbers there.
// Python fake keeps its device on the subclass, so read the backend key both
// fakes carry (Note [Fake Tensor Dispatch Keys]).
bool is_noncontiguous_supported(const Tensor& self, const Tensor& other) {
  const auto& first = self.unsafeGetTensorImpl()->is_wrapped_number() ? other : self;
  return !first.key_set().has_backend(BackendComponent::HPUBit);
}

// fake_impls.infer_size. Unlike at::infer_size_symdimvector, it compares
// sizeA == sizeB in Python's operand order.
c10::SymDimVector fake_infer_size(c10::SymIntArrayRef a, c10::SymIntArrayRef b) {
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
          py_eq(size_a, size_b),
          "The size of tensor a (", size_a, ") must match the size of tensor b (", size_b,
          ") at non-singleton dimension ", i);
    }
    expanded_sizes[i] = TORCH_GUARD_OR_FALSE(size_a.sym_eq(1)) ? size_b : size_a;
  }
  return expanded_sizes;
}

} // namespace

// _make_elementwise_binary_reference / refs.add (alpha is None when unset):
// elementwise_type_promotion_wrapper -> _maybe_broadcast -> [prims.mul(b,
// alpha)] -> prim -> conversion to the result dtype.
Tensor binary_ref_meta(
    const Tensor& self,
    const Tensor& other,
    TypePromotionKind kind,
    bool fake_devices,
    const std::optional<Scalar>& alpha) {
  const auto [compute_dtype, result_dtype] = elementwise_dtypes(self, other, kind);
  auto args = maybe_broadcast(
      {maybe_convert_desc(meta_desc(self, fake_devices), compute_dtype),
       maybe_convert_desc(meta_desc(other, fake_devices), compute_dtype)});
  if (alpha.has_value()) {
    // utils.is_weakly_lesser_type over bool < int < float < complex
    auto python_type_rank = [](ScalarType t) {
      return t == kBool ? 0 : isIntegralType(t, /*includeBool=*/false) ? 1 : isFloatingType(t) ? 2 : 3;
    };
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
      b = prim_elementwise_desc({b, alpha_desc}, b.dtype);
    }
  }
  auto out = prim_elementwise_desc(args, compute_dtype);
  if (!is_noncontiguous_supported(self, other) && !is_contiguous_or_false(out)) {
    DimVector identity(out.dim());
    std::iota(identity.begin(), identity.end(), 0);
    out.strides = empty_permuted_desc(out.sizes, identity, out.dtype).strides;
  }
  if (out.dtype == result_dtype) {
    // torch.empty_permuted: contiguous physical allocation, then restrided
    auto result = at::detail::empty_symint_meta(out.sizes, out.dtype, std::nullopt, kMeta, std::nullopt, std::nullopt);
    result.unsafeGetTensorImpl()->set_sizes_and_strides(out.sizes, out.strides);
    return Tensor(std::move(result));
  }
  const auto converted = convert_element_type_desc(out, result_dtype);
  return Tensor(at::detail::empty_strided_symint_meta(converted.sizes, converted.strides, result_dtype));
}

// fake_impls.make_fast_binary_impl, which Python fake tries first when the
// inputs are symbolic. Returns an undefined tensor where it falls back to the
// ref (the ref then raises for mismatched devices). The output device is left
// to the caller.
Tensor fast_binary_meta(const Tensor& self, const Tensor& other, TypePromotionKind kind) {
  const std::array<const Tensor*, 2> operands = {&self, &other};
  c10::SymDimVector final_shape(self.sym_sizes().begin(), self.sym_sizes().end());
  for (const auto* op : operands) {
    final_shape = fake_infer_size(final_shape, op->sym_sizes());
  }

  bool obvious = false;
  for (const auto* op : operands) {
    if (op->unsafeGetTensorImpl()->is_wrapped_number() || op->dim() != static_cast<int64_t>(final_shape.size())) {
      continue;
    }
    const auto sizes = op->sym_sizes();
    c10::SymBool eq(true);
    for (const auto i : c10::irange(final_shape.size())) {
      eq = eq.sym_and(py_eq(sizes[i], final_shape[i]));
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
  if (kind != TypePromotionKind::DEFAULT || self_number || other_number || other.scalar_type() != dtype) {
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
      channels_last = channels_last && is_channels_last_contiguous_or_false(desc);
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
