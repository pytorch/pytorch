#include <ATen/MemoryOverlap.h>
#include <ATen/core/TensorBase.h>
#include <c10/util/irange.h>
#include <optional>

namespace at {

MemOverlap has_internal_overlap(const TensorBase& tensor) {
  return has_internal_overlap(tensor.unsafeGetTensorImpl());
}

MemOverlap has_internal_overlap(TensorImpl* t) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t->layout() == kStrided);

  auto sizes = t->sym_sizes();
  auto strides = t->sym_strides();

  // When we have unbacked symint strides, is_non_overlapping_and_dense
  // often results in guard on data dependent errors. For now
  // let us bail early if there are unbacked symint strides.
  for (const auto i : c10::irange(strides.size())) {
    if (!strides[i].has_hint()) {
      return MemOverlap::TooHard;
    }
  }

  if (t->is_non_overlapping_and_dense_or_false()) {
    return MemOverlap::No;
  }

  for (const auto i : c10::irange(strides.size())) {
    // NB: The size oblivious test is written very carefully here.  When
    // unbacked SymInts are involved, we should try to conservatively report
    // if memory overlap /could/ happen under some setting of unbacked
    // SymInts.  Thus, if I have u0 size, we should assume that this has > 1
    // elements (first expression), but if I have a u0 stride, I should NOT
    // assume that it is not zero (second expression)
    if (TORCH_GUARD_OR_FALSE(sizes[i].sym_gt(1)) && strides[i] == 0) {
      return MemOverlap::Yes;
    }
  }

  return MemOverlap::TooHard;
}

void assert_no_internal_overlap(const TensorBase& t) {
  assert_no_internal_overlap(t.unsafeGetTensorImpl());
}

void assert_no_internal_overlap(TensorImpl* t) {
  TORCH_CHECK(has_internal_overlap(t) != MemOverlap::Yes,
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation.");
}

MemOverlapStatus get_overlap_status(const TensorBase& a, const TensorBase& b) {
  return get_overlap_status(a.unsafeGetTensorImpl(), b.unsafeGetTensorImpl());
}

static std::optional<bool> maybe_guard_bool(const c10::SymBool& value) {
  if (!value.has_hint()) {
    return std::nullopt;
  }
  return TORCH_GUARD_OR_FALSE(value);
}

static MemOverlapStatus symbolic_same_start_overlap(
    const TensorImpl* a,
    const TensorImpl* b) {
  if (a->itemsize() != b->itemsize() || a->dim() != b->dim()) {
    return MemOverlapStatus::TooHard;
  }

  const auto same_start = maybe_guard_bool(
      a->sym_storage_offset().sym_eq(b->sym_storage_offset()));
  const auto sizes_equal =
      maybe_guard_bool(c10::sym_equals(a->sym_sizes(), b->sym_sizes()));
  if (!same_start || !*same_start || !sizes_equal || !*sizes_equal) {
    return MemOverlapStatus::TooHard;
  }

  bool different_mapping = false;
  bool expanded_src = false;
  const auto a_strides = a->sym_strides();
  const auto b_strides = b->sym_strides();
  for (const auto i : c10::irange(a->dim())) {
    const auto size_zero = maybe_guard_bool(a->sym_size(i).sym_eq(0));
    if (!size_zero) {
      return MemOverlapStatus::TooHard;
    }
    if (*size_zero) {
      return MemOverlapStatus::No;
    }
    const auto size_gt_one = maybe_guard_bool(a->sym_size(i).sym_gt(1));
    if (!size_gt_one) {
      return MemOverlapStatus::TooHard;
    }
    if (!*size_gt_one) {
      continue;
    }

    const auto a_stride_zero =
        maybe_guard_bool(a_strides[i].sym_eq(0));
    const auto b_stride_zero =
        maybe_guard_bool(b_strides[i].sym_eq(0));
    if (!a_stride_zero || !b_stride_zero) {
      return MemOverlapStatus::TooHard;
    }
    if (*a_stride_zero) {
      return MemOverlapStatus::TooHard;
    }
    if (*b_stride_zero) {
      expanded_src = true;
      continue;
    }

    const auto stride_diff =
        maybe_guard_bool(a_strides[i].sym_ne(b_strides[i]));
    if (!stride_diff) {
      return MemOverlapStatus::TooHard;
    }
    different_mapping |= *stride_diff;
  }

  if (expanded_src && !different_mapping) {
    return MemOverlapStatus::TooHard;
  }
  return different_mapping ? MemOverlapStatus::Partial
                           : MemOverlapStatus::Full;
}

MemOverlapStatus get_overlap_status(const TensorImpl* a, const TensorImpl* b) {
  if (a == b) return MemOverlapStatus::Full;
  const auto has_symbolic_sizes_strides =
      a->has_symbolic_sizes_strides() || b->has_symbolic_sizes_strides() ||
      a->sym_storage_offset().is_symbolic() ||
      b->sym_storage_offset().is_symbolic();
  if (has_symbolic_sizes_strides && a->layout() == kStrided &&
      b->layout() == kStrided) {
    const auto& a_storage = a->unsafe_storage();
    if (a_storage && a_storage.is_alias_of(b->unsafe_storage())) {
      return symbolic_same_start_overlap(a, b);
    }
  }
  if (has_symbolic_sizes_strides) {
    return MemOverlapStatus::TooHard;
  }
  if (a->numel() == 0 || b->numel() == 0) {
    return MemOverlapStatus::No;
  }
  if (a->layout() == kStrided && b->layout() == kStrided &&
      a->itemsize() == b->itemsize()) {
    const auto& a_storage = a->unsafe_storage();
    const auto same_start =
        a->storage_offset() == b->storage_offset() && a_storage &&
        a_storage.is_alias_of(b->unsafe_storage());
    if (same_start && a->sizes() == b->sizes()) {
      bool different_mapping = false;
      bool expanded_src = false;
      for (const auto i : c10::irange(a->dim())) {
        if (a->sizes()[i] <= 1) {
          continue;
        }
        if (a->strides()[i] == 0) {
          return MemOverlapStatus::TooHard;
        }
        if (b->strides()[i] == 0) {
          expanded_src = true;
          continue;
        }
        if (a->strides()[i] != b->strides()[i]) {
          different_mapping = true;
        }
      }
      if (expanded_src && !different_mapping) {
        return MemOverlapStatus::TooHard;
      }
      return different_mapping ? MemOverlapStatus::Partial
                               : MemOverlapStatus::Full;
    }
  }
  if (!a->is_non_overlapping_and_dense_or_false() || !b->is_non_overlapping_and_dense_or_false()) {
    return MemOverlapStatus::TooHard;
  }
  // Test for storage equality, rather than pointer equality.
  // This reduces precision, but if people are aliasing the
  // same pointer across multiple storages there are many
  // similar situations (e.g., storage().data() == storage().data()+1)
  // which we will miss.
  const auto& a_storage = a->unsafe_storage();
  if (a_storage && a_storage.is_alias_of(b->unsafe_storage())) {
    const auto a_begin = static_cast<const char*>(a->data());
    const auto a_end = a_begin + a->numel() * a->itemsize();
    const auto b_begin = static_cast<const char*>(b->data());
    const auto b_end = b_begin + b->numel() * b->itemsize();

    if (a_begin == b_begin && a_end == b_end) {
      return (a->strides() == b->strides()) ?
          MemOverlapStatus::Full : MemOverlapStatus::Partial;
    }
    if (a_begin < b_end && b_begin < a_end) {
      return MemOverlapStatus::Partial;
    }
  }
  return MemOverlapStatus::No;
}

void assert_no_partial_overlap(const TensorBase& a, const TensorBase& b) {
  assert_no_partial_overlap(a.unsafeGetTensorImpl(), b.unsafeGetTensorImpl());
}

void assert_no_partial_overlap(TensorImpl* a, TensorImpl* b) {
  TORCH_CHECK(get_overlap_status(a, b) != MemOverlapStatus::Partial,
    "unsupported operation: some elements of the input tensor and "
    "the written-to tensor refer to a single memory location. "
    "Please clone() the tensor before performing the operation.");
}

void assert_no_overlap(const TensorBase& a, const TensorBase& b) {
  assert_no_overlap(a.unsafeGetTensorImpl(), b.unsafeGetTensorImpl());
}

void assert_no_overlap(TensorImpl* a, TensorImpl* b) {
  const auto lap = get_overlap_status(a, b);
  TORCH_CHECK(lap != MemOverlapStatus::Partial && lap != MemOverlapStatus::Full,
    "unsupported operation: some elements of the input tensor and "
    "the written-to tensor refer to a single memory location. "
    "Please clone() the tensor before performing the operation.");
}

}
