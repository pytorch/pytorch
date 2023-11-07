#include <c10/core/Contiguity.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/SymbolicShapeMeta.h>

namespace c10 {

SymbolicShapeMeta::SymbolicShapeMeta(const SymbolicShapeMeta& other) {
  std::scoped_lock lock{other.mutables_};
  sizes_ = other.sizes_;
  strides_ = other.strides_;
  storage_offset_ = other.storage_offset_;
  strides_valid_ = other.strides_valid_;
  numel_ = other.numel_;
  is_contiguous_ = other.is_contiguous_;
  is_channels_last_contiguous_ = other.is_channels_last_contiguous_;
  is_channels_last_3d_contiguous_ = other.is_channels_last_3d_contiguous_;
  is_channels_last_ = other.is_channels_last_;
  is_channels_last_3d_ = other.is_channels_last_3d_;
  is_non_overlapping_and_dense_ = other.is_non_overlapping_and_dense_;
  available_.store(other.available_.load());
}

// base, sizes, strides
static c10::optional<
    std::tuple<SymNode, std::vector<SymNode>, std::vector<SymNode>>>
normalize_sym_sizes_strides(SymIntArrayRef sizes, SymIntArrayRef strides) {
  // Look for a SymNode to dispatch on
  SymNode base;
  bool all_hinted = true;
  // NB: sizes/strides guaranteed to be positive, so only need
  // is_heap_allocated
  for (const auto& s : sizes) {
    if (all_hinted && !s.has_hint()) {
      all_hinted = false;
    }
    if (!base && s.is_heap_allocated()) {
      base = s.toSymNode();
    }
  }
  for (const auto& s : strides) {
    if (all_hinted && !s.has_hint()) {
      all_hinted = false;
    }
    if (!base && s.is_heap_allocated()) {
      base = s.toSymNode();
    }
  }
  if (!base || all_hinted) {
    // Couldn't find.  Tell the caller to do the normal computation
    // Alternately, if everything is hinted, we want the normal computation
    // too
    return c10::nullopt;
  }
  // Populate the SymNode array
  std::vector<SymNode> size_nodes;
  std::vector<SymNode> stride_nodes;
  size_nodes.reserve(sizes.size());
  stride_nodes.reserve(strides.size());
  for (const auto& s : sizes) {
    size_nodes.emplace_back(s.wrap_node(base));
  }
  for (const auto& s : strides) {
    stride_nodes.emplace_back(s.wrap_node(base));
  }
  return c10::make_optional(
      std::tuple<SymNode, std::vector<SymNode>, std::vector<SymNode>>(
          std::move(base), std::move(size_nodes), std::move(stride_nodes)));
}

// Special treatment because of numel
SymBool SymbolicShapeMeta::compute_contiguous() const {
  if (!strides_valid_) {
    return false;
  }
  c10::SymIntArrayRef sizes(sizes_);
  c10::SymIntArrayRef strides(strides_);
  return _compute_contiguous(sizes, strides, numel());
}

// The rest of them
#define DEFINE_EAGER_SYMBOOL_COMPUTE(name, nodeimpl, fallback) \
  SymBool SymbolicShapeMeta::name() const {                    \
    if (!strides_valid_) {                                     \
      return false;                                            \
    }                                                          \
    c10::SymIntArrayRef sizes(sizes_);                         \
    c10::SymIntArrayRef strides(strides_);                     \
    return fallback(sizes, strides);                           \
  }

#define DEFINE_SYMBOOL_COMPUTE(name, nodeimpl, fallback)        \
  SymBool SymbolicShapeMeta::name() const {                     \
    if (!strides_valid_) {                                      \
      return false;                                             \
    }                                                           \
    auto n = normalize_sym_sizes_strides(sizes_, strides_);     \
    if (n.has_value()) {                                        \
      SymNode base;                                             \
      std::vector<SymNode> size_nodes;                          \
      std::vector<SymNode> stride_nodes;                        \
      std::tie(base, size_nodes, stride_nodes) = *n;            \
      return SymBool(base->nodeimpl(size_nodes, stride_nodes)); \
    } else {                                                    \
      c10::SymIntArrayRef sizes(sizes_);                        \
      c10::SymIntArrayRef strides(strides_);                    \
      return fallback(sizes, strides);                          \
    }                                                           \
  }

// clang-format off
DEFINE_EAGER_SYMBOOL_COMPUTE(compute_channels_last_contiguous_2d, is_channels_last_contiguous_2d, _compute_channels_last_contiguous_2d)
DEFINE_EAGER_SYMBOOL_COMPUTE(compute_channels_last_contiguous_3d, is_channels_last_contiguous_3d, _compute_channels_last_contiguous_3d)
DEFINE_EAGER_SYMBOOL_COMPUTE(compute_strides_like_channels_last_2d, is_channels_last_strides_2d, is_channels_last_strides_2d)
DEFINE_EAGER_SYMBOOL_COMPUTE(compute_strides_like_channels_last_3d, is_channels_last_strides_3d, is_channels_last_strides_3d)
DEFINE_SYMBOOL_COMPUTE(compute_non_overlapping_and_dense, is_non_overlapping_and_dense, _compute_non_overlapping_and_dense)
// clang-format on

#undef DEFINE_SYMBOOL_COMPUTE

// Glue compute
// NB: this logic very intentionally short circuits if possible.  Without
// short circuiting, it causes
// python test/functorch/test_aotdispatch.py -k
// test_aot_autograd_symbolic_exhaustive_nn_functional_unfold_cpu_float32 to run
// very slowly.

static bool definitely_true(const SymBool& b) {
  return b.has_hint() && b.guard_bool(__FILE__, __LINE__);
}

SymBool SymbolicShapeMeta::compute_is_non_overlapping_and_dense_dim4_locked() const {
  init_is_contiguous_locked();
  if (definitely_true(is_contiguous_)) {
    return true;
  }
  init_is_channels_last_contiguous_locked();
  if (definitely_true(is_channels_last_contiguous_)) {
    return true;
  }
  return is_contiguous_ | is_channels_last_contiguous_ |
      compute_non_overlapping_and_dense();
}

SymBool SymbolicShapeMeta::compute_channels_last_contiguous_3d_dim5_locked() const {
  init_is_channels_last_contiguous_locked();
  if (definitely_true(is_channels_last_contiguous_)) {
    return false;
  }
  return ~is_channels_last_contiguous_ & compute_channels_last_contiguous_3d();
}

SymBool SymbolicShapeMeta::compute_channels_last_2d_dim5_locked() const {
  init_is_channels_last_3d_contiguous_locked();
  if (definitely_true(is_channels_last_3d_contiguous_)) {
    return false;
  }
  return ~is_channels_last_3d_contiguous_ &
      compute_strides_like_channels_last_2d();
}

SymBool SymbolicShapeMeta::compute_channels_last_3d_dim5_locked() const {
  init_is_channels_last_locked();
  if (definitely_true(is_channels_last_)) {
    return false;
  }
  return ~is_channels_last_ & compute_strides_like_channels_last_3d();
}

SymBool SymbolicShapeMeta::compute_is_non_overlapping_and_dense_dim5_locked() const {
  init_is_contiguous_locked();
  if (definitely_true(is_contiguous_)) {
    return true;
  }
  init_is_channels_last_contiguous_locked();
  if (definitely_true(is_channels_last_contiguous_)) {
    return true;
  }
  init_is_channels_last_3d_contiguous_locked();
  if (definitely_true(is_channels_last_3d_contiguous_)) {
    return true;
  }
  return is_contiguous_ | is_channels_last_contiguous_ |
      is_channels_last_3d_contiguous_ | compute_non_overlapping_and_dense();
}

SymBool SymbolicShapeMeta::compute_is_non_overlapping_and_dense_anydim_locked() const {
  init_is_contiguous_locked();
  if (definitely_true(is_contiguous_)) {
    return true;
  }
  return is_contiguous_ | compute_non_overlapping_and_dense();
}

void SymbolicShapeMeta::init_numel() const {
  std::scoped_lock lock{mutables_};
  init_numel_locked();
}

void SymbolicShapeMeta::init_is_contiguous() const {
  std::scoped_lock lock{mutables_};
  init_is_contiguous_locked();
}

void SymbolicShapeMeta::init_is_channels_last_contiguous() const {
  std::scoped_lock lock{mutables_};
  init_is_channels_last_contiguous_locked();
}

void SymbolicShapeMeta::init_is_channels_last_3d_contiguous() const {
  std::scoped_lock lock{mutables_};
  init_is_channels_last_3d_contiguous_locked();
}

void SymbolicShapeMeta::init_is_channels_last() const {
  std::scoped_lock lock{mutables_};
  init_is_channels_last_locked();
}

void SymbolicShapeMeta::init_is_channels_last_3d() const {
  std::scoped_lock lock{mutables_};
  init_is_channels_last_3d_locked();
}

void SymbolicShapeMeta::init_is_non_overlapping_and_dense() const {
  std::scoped_lock lock{mutables_};
  init_is_non_overlapping_and_dense_locked();
}

void SymbolicShapeMeta::init_numel_locked() const {
  if (has_numel()) { return; }
  numel_ = multiply_integers(sizes_);
  available_.fetch_or(numel_avail);
}

void SymbolicShapeMeta::init_is_contiguous_locked() const {
  if (has_is_contiguous()) { return; }
  is_contiguous_ = compute_contiguous();
  available_.fetch_or(is_contiguous_avail);
}

void SymbolicShapeMeta::init_is_channels_last_contiguous_locked() const {
  if (has_is_channels_last_contiguous()) { return; }
  is_channels_last_contiguous_ = [&] {
    switch (dim()) {
      case 5:
      case 4: {
        return compute_channels_last_contiguous_2d();
      }
      default:
        return SymBool{false};
    }
  }();
  available_.fetch_or(is_channels_last_contiguous_avail);
}

void SymbolicShapeMeta::init_is_channels_last_3d_contiguous_locked() const {
  if (has_is_channels_last_3d_contiguous()) { return; }
  is_channels_last_3d_contiguous_ = [&] {
    switch (dim()) {
      case 5:
        return compute_channels_last_contiguous_3d_dim5_locked();
      default:
        return SymBool{false};
    }
  }();
  available_.fetch_or(is_channels_last_3d_contiguous_avail);
}

void SymbolicShapeMeta::init_is_channels_last_locked() const {
  if (has_is_channels_last()) { return; }
  is_channels_last_ = [&] {
    switch (dim()) {
      case 5:
        return compute_channels_last_2d_dim5_locked();
      case 4:
        return compute_strides_like_channels_last_2d();
      default:
        return SymBool{false};
    }
  }();
  available_.fetch_or(is_channels_last_avail);
}

void SymbolicShapeMeta::init_is_channels_last_3d_locked() const {
  if (has_is_channels_last_3d()) { return; }
  is_channels_last_3d_ = [&] {
    switch (dim()) {
      case 5:
        return compute_channels_last_3d_dim5_locked();
      default:
        return SymBool{false};
    }
  }();
  available_.fetch_or(is_channels_last_3d_avail);
}

void SymbolicShapeMeta::init_is_non_overlapping_and_dense_locked() const {
  if (has_is_non_overlapping_and_dense()) { return; }
  is_non_overlapping_and_dense_ = [&] {
    switch (dim()) {
      case 5:
        return compute_is_non_overlapping_and_dense_dim5_locked();
      case 4:
        return compute_is_non_overlapping_and_dense_dim4_locked();
      default:
        return compute_is_non_overlapping_and_dense_anydim_locked();
    }
  }();
  available_.fetch_or(is_non_overlapping_and_dense_avail);
}

} // namespace c10
