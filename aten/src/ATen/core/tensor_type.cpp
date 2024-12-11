#include <ATen/core/Tensor.h>
#include <ATen/core/jit_type.h>
#include <c10/core/GradMode.h>

#include <utility>

namespace c10 {

namespace {

// The idea is to only mark possible overlap across dimensions. We want to
// return false for expanded tensors and permuted tensors, for which dimensional
// collapsing is safe.
bool possible_cross_dimension_overlap(c10::IntArrayRef sizes, c10::IntArrayRef strides) {
  int n_dim = static_cast<int>(sizes.size());
  std::vector<size_t> stride_indices(n_dim);
  std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);

  // sort indices going with ascending strides
  for (int i = 1; i < n_dim; i++) {
    auto c = i;
    for (int j = i - 1; j >= 0; j--) {
      if (strides[stride_indices[j]] > strides[stride_indices[c]]) {
        std::swap(stride_indices[j], stride_indices[c]);
        c = j;
      }
    }
  }

  for (const auto i : c10::irange(1, n_dim)) {
    if (i != 0) {
      // we are being conservative on checking for memory overlap
      if (sizes[stride_indices[i]] != 1 && strides[stride_indices[i]] < sizes[stride_indices[i-1]] * strides[stride_indices[i-1]]) {
        return true;
      }
    }
  }
  return false;
}

}

const TensorTypePtr& TensorType::get() {
  static auto value = TensorType::create(
      {}, {}, SymbolicShape(), VaryingShape<Stride>{}, {});
  return value;
}

ListTypePtr ListType::ofTensors() {
  static auto value = ListType::create(TensorType::get());
  return value;
}

template <typename T>
VaryingShape<T> VaryingShape<T>::merge(const VaryingShape<T>& other) const {
  if (!dims_ || !other.dims_ || dims_->size() != other.dims_->size()) {
    return VaryingShape<T>();
  }
  ListOfOptionalElements dims;
  for (size_t i = 0, n = dims_->size(); i < n; i++) {
    dims.push_back(merge_primitive((*dims_)[i], (*other.dims_)[i]));
  }
  return VaryingShape<T>(std::move(dims));
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const VaryingShape<T>& vs) {
  out << "(";
  if (!vs.size()) {
    out << "*)";
    return out;
  }

  for (size_t i = 0; i < vs.size(); i++) {
    if (i > 0) {
      out << ", ";
    }
    auto const& v = vs[i];
    if (v.has_value()) {
      out << v.value();
    } else {
      out << "*";
    }
  }
  out << ")";
  return out;
}

template std::ostream& operator<<(
    std::ostream& out,
    const VaryingShape<int64_t>& vs);
template std::ostream& operator<<(
    std::ostream& out,
    const VaryingShape<Stride>& vs);

std::ostream& operator<<(
    std::ostream& os,
    const SymbolicShape& ss) {
  // TODO: Unranked SymbolicShape printing is ambiguous with that of
  // dynamic-shaped vector.
  auto rank_opt = ss.rank();
  if(!rank_opt.has_value()) {
    os << "(*)";
    return os;
  }
  auto sizes_opt = ss.sizes();

  os << "(";
  for (size_t i = 0; i < rank_opt.value(); i++) {
    if (i > 0) {
      os << ", ";
    }
    if(sizes_opt.has_value() && sizes_opt.value()[i].is_static()) {
      os << sizes_opt.value()[i];
    } else {
      os << "*";
    }
  }
  os << ")";

  return os;
}

std::ostream& operator<<(std::ostream& os, const ShapeSymbol& s) {
  if (s.value_ >= 0) {
    os << s.value_;
  } else {
    os << "SS(" << s.value_ << ')';
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Stride& s) {
  os << "{";
  if (s.stride_index_.has_value()) {
    os << *s.stride_index_;
  } else {
    os << "*";
  }
  os << ":";
  if (s.stride_.has_value()) {
    os << *s.stride_;
  } else {
    os << "*";
  }
  os << '}';
  return os;
}

VaryingShape<Stride> TensorType::computeStrideProps(
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    bool tensor_contiguity) {
  int n_dim = static_cast<int>(sizes.size());
  std::vector<size_t> stride_indices(n_dim);
  // default has_overlap to false as we only compute overlap when:
  // 1. input sizes/strides fails format check;
  // 2. tensor_contiguity are not set.
  bool has_overlap = false;

  // Sorting strides in ascending order
  // Example:
  //  Prior to sorting
  //  Idx:     [0,   1,  2,  3]
  //  sizes:   [8,   1, 10, 16]
  //  Strides: [160, 1, 16,  1]
  //  After sorting
  //  Idx:     [1,  3,  2,   0]
  //  sizes:   [1, 16, 10,   8]
  //  Strides: [1,  1, 16, 160]
  //
  // The logic below follows what TensorIterator uses in its logic:
  //   1. Fast_set_up is the short-cut to identify a. channels_last and
  //      b. contiguous format, which is what we have in the below logic.
  //   2. In more generla cases, it does best effort to preserve permutatoin.
  if (is_channels_last_strides_2d(sizes, strides) || is_channels_last_strides_3d(sizes, strides)) {
    // case 1.a. short cut channels last
    std::iota(stride_indices.rbegin() + 1, stride_indices.rend() - 1, 2);
    stride_indices[0] = 1;
    stride_indices[n_dim - 1] = 0;
  } else if (is_contiguous_strides(sizes, strides)) {
    // case 1.b. short cut contiguous
    std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);
  } else {
    std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);
    // case 2.
    //
    // For broadcasted dimension where stride is 0, we have to stick to
    // TensorIterator behavior in eager, where they introduce an ambiguous
    // comparison result to preserve permutation by best effort.
    // For more details, see NOTE: [Computing output strides]
    auto should_swap = [&](size_t a, size_t b) {
      if (strides[a] == 0 || strides[b] == 0) {
        return 0;
      } else if (strides[a] < strides[b]) {
        return -1;
      } else if (strides[a] > strides[b]) {
        return 1;
      } else { // strides[a] == strides[b]
        if (sizes[a] > sizes[b]) {
          return 1;
        }
      }
      return 0;
    };
    for (int i = 1; i < n_dim; i++) {
      int dim1 = i;
      for (int dim0 = i - 1; dim0 >= 0; dim0--) {
        int comparison = should_swap(stride_indices[dim0], stride_indices[dim1]);
        if (comparison > 0) {
          std::swap(stride_indices[dim0], stride_indices[dim1]);
          dim1 = dim0;
        } else if (comparison < 0) {
          break;
        }
      }
    }
    // conveniently is_contiguous_strides/is_contiguous_strides only returns
    // true when there's no memory overlap, so we only re-compute has_overlap
    // in the last branch when both returns false
    if (!tensor_contiguity) {
      // trust tensor_contiguity and only computes overlap when it is not set
      has_overlap = possible_cross_dimension_overlap(sizes, strides);
    }
  }

  std::vector<Stride> stride_properties;
  stride_properties.reserve(stride_indices.size());
  for (size_t i = 0; i < stride_indices.size(); i++) {
    bool contiguous_ = tensor_contiguity;
    if (!contiguous_) {
      if (!has_overlap) {
        // innermost stride expected to be 1
        // TODO: turn contiguous_ into an enum CONTIGUOUS, NONCONTIGUOUS,
        // BROADCASTED
        if (i == 0) {
          contiguous_ = strides[stride_indices[i]] == 1;
        } else {
          contiguous_ = strides[stride_indices[i]] == 1 ||
              (strides[stride_indices[i]] != 0 &&
               strides[stride_indices[i]] ==
                   strides[stride_indices[i - 1]] * sizes[stride_indices[i - 1]]);
        }
      } else {
        // leaving this assign statement for readability;
        contiguous_ = false;
      }
    }
    stride_properties.emplace_back(stride_indices[i], contiguous_, strides[stride_indices[i]]);
  }

  return VaryingShape<Stride>{stride_properties};
}

TensorTypePtr TensorType::create(const at::Tensor& t) {
  VaryingShape<bool> contiguity;
  VaryingShape<size_t> stride_indices;
  VaryingShape<int64_t> strides;
  VaryingShape<int64_t> sizes;
  if (t.layout() == at::kStrided && !t.is_nested()) {
    sizes = VaryingShape<int64_t>{t.sizes().vec()};
    strides = VaryingShape<int64_t>{t.strides().vec()};
    return TensorType::create(
        t.scalar_type(), t.device(), sizes, strides, t.requires_grad(), false, t.is_contiguous());
  }

  return TensorType::create(
      t.scalar_type(),
      t.device(),
      SymbolicShape(),
      VaryingShape<Stride>{},
      t.requires_grad(),
      false);
}

TensorTypePtr TensorType::create(
    std::optional<at::ScalarType> scalar_type,
    std::optional<Device> device,
    const VaryingShape<int64_t>& sizes,
    const VaryingShape<int64_t>& strides,
    std::optional<bool> requires_grad,
    std::optional<bool> undefined, bool tensor_contiguity) {
  const auto stride_concrete_sizes = strides.concrete_sizes();
  if(stride_concrete_sizes.has_value()){
    const auto size_concrete_sizes = sizes.concrete_sizes();
    // handles case where strides are set
    TORCH_INTERNAL_ASSERT(size_concrete_sizes.has_value() && size_concrete_sizes->size() == stride_concrete_sizes->size());
    auto sprops =
       computeStrideProps(*size_concrete_sizes, *stride_concrete_sizes, tensor_contiguity);
    auto symbol_sizes = SymbolicShape(*size_concrete_sizes);
    return TensorType::create(
      scalar_type, device, symbol_sizes, sprops, requires_grad, undefined);
  } else {
    // strides are all null, but still have number of strides equal to number of ranks
    TORCH_INTERNAL_ASSERT(sizes.sizes().has_value() && sizes.size());
    auto symbol_sizes = SymbolicShape(*sizes.sizes());
    return TensorType::create(
      scalar_type, device, symbol_sizes, VaryingShape<Stride>(*sizes.size()), requires_grad, undefined);
  }
}

TensorTypePtr TensorType::create(
    std::optional<at::ScalarType> scalar_type,
    std::optional<Device> device,
    SymbolicShape sizes,
    VaryingShape<Stride> strides,
    std::optional<bool> requires_grad,
    std::optional<bool> undefined) {
  auto pt = TensorTypePtr(new TensorType(
      scalar_type, device, std::move(sizes), std::move(strides), requires_grad, undefined));
  return pt;
}

TensorTypePtr TensorType::create(
    std::optional<at::ScalarType> scalar_type,
    std::optional<Device> device,
    std::optional<size_t> dim,
    std::optional<bool> requires_grad) {
  return TensorType::create(
      scalar_type,
      device,
      SymbolicShape(dim),
      VaryingShape<Stride>(dim),
      requires_grad);
}

std::string TensorType::str() const {
  return "Tensor";
}

std::atomic<size_t> ShapeSymbol::num_symbols{1};

template struct VaryingShape<c10::ShapeSymbol>;
template struct VaryingShape<bool>;
template struct VaryingShape<size_t>;
template struct VaryingShape<int64_t>;
template struct VaryingShape<c10::Stride>;

VaryingShape<int64_t> TensorType::sizes() const {
  if (!sizes_.rank()) {
    return VaryingShape<int64_t>();
  }
  return VaryingShape<int64_t>(
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      fmap(*sizes_.sizes(), [](ShapeSymbol ss) {
        // we turn symbolic shapes into unknowns
        return ss.is_static()
            ? std::optional<int64_t>(ss.static_size())
            : std::nullopt;
      }));
}

TensorTypePtr TensorType::merge(const TensorType& other, bool merge_sizes) const {
  auto scalar_type = merge_primitive(scalarType(), other.scalarType());
  auto dev = merge_primitive(device(), other.device());
  auto sprops = stride_properties().merge(other.stride_properties());
  auto gr = merge_primitive(requiresGrad(), other.requiresGrad());
  auto undef = merge_primitive(undefined(), other.undefined());
  return TensorType::create(
      scalar_type,
      dev,
      merge_sizes ? symbolic_sizes().merge(other.symbolic_sizes())
                  : symbolic_sizes(),
      sprops,
      gr,
      undef);
}

template <typename T>
static bool is_null_or_equal(std::optional<T> a, c10::IntArrayRef b) {
  return !a.has_value() || a.value() == b;
}

bool TensorType::matchTensor(const at::Tensor& t) {
  bool undef = undefined().value_or(!t.defined());
  if (undef != !t.defined()) {
    // When the followings are true, we consider it's not a match:
    // - undefined().has_value() == true
    // - undefined().value() != !t.defined()
    return false;
  } else if (!t.defined()) {
    // When the followings are true, we consider it's a match:
    // - t is not defined
    // - undefined() == null or undefined().value() == true
    return true;
  }
  // Here we know t.defined() == true and compare all other properties.
  bool rg = at::GradMode::is_enabled() && t.requires_grad();
  bool matched_strides = (!stride_properties().size()) ||
      (!t.has_storage() && !stride_properties().isComplete()) ||
      stride_properties() ==
          computeStrideProps(t.sizes(), t.strides(), t.is_contiguous());
  return scalarType().value_or(t.scalar_type()) == t.scalar_type()
    && device().value_or(t.device()) == t.device()
    && requiresGrad().value_or(rg) == rg
    && matched_strides
    && is_null_or_equal(sizes().concrete_sizes(), t.sizes());
}

bool TensorType::equals(const c10::Type& rhs) const {
  if (rhs.kind() != kind()) {
    return false;
  }
  auto rt = rhs.expect<TensorType>();

  return scalar_type_ == rt->scalarType() && sizes() == rt->sizes() &&
      stride_properties() == rt->stride_properties() &&
      device() == rt->device() && requiresGrad() == rt->requiresGrad() &&
      undefined() == rt->undefined();
}

VaryingShape<int64_t> TensorType::strides() const {
  auto const strides_sizes = strides_.sizes();
  if (!strides_sizes.has_value()) {
    return VaryingShape<int64_t>();
  }
  std::vector<std::optional<int64_t>> ss(strides_sizes->size());
  for (auto const& stride:strides_sizes.value()) {
    if (!stride.has_value()) {
      continue;
    }
    const auto& s = *stride;
    if (s.stride_index_.has_value() && s.stride_.has_value()) {
      ss[*s.stride_index_] = *s.stride_;
    }
  }
  return VaryingShape<int64_t>(std::move(ss));
}

TensorType::TensorType(
    std::optional<at::ScalarType> scalar_type,
    std::optional<Device> device,
    SymbolicShape sizes,
    VaryingShape<Stride> strides,
    std::optional<bool> requires_grad,
    std::optional<bool> undefined)
    : SharedType(TypeKind::TensorType),
      scalar_type_(scalar_type),
      device_(device),
      sizes_(std::move(sizes)),
      strides_(std::move(strides)),
      requires_grad_(requires_grad),
      undefined_(undefined) {}

TensorTypePtr TensorType::createContiguous(
    at::ScalarType scalar_type,
    at::Device device,
    at::IntArrayRef sizes) {
  auto strides = contiguousStridesOf(sizes);
  TORCH_INTERNAL_ASSERT(strides.size() == sizes.size());
  return create(
      scalar_type,
      device,
      VaryingShape<int64_t>(sizes),
      VaryingShape<int64_t>(strides),
      std::nullopt);
}

const SymbolicShape& TensorType::symbolic_sizes() const {
  return sizes_;
}

bool TensorType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  if (auto rhs_p = rhs.cast<TensorType>()) {
    // if we have the same pointer, avoid computing the merge
    if (this == rhs_p.get()) {
      return true;
    }
    return *merge(*rhs_p) == *rhs_p;
  }
  return Type::isSubtypeOfExt(rhs, why_not);
}

}
