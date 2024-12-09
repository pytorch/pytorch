#pragma once
#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/DimVector.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>

namespace c10 {

class C10_API SymbolicShapeMeta {
 public:
  // Basic metadata from which other quantities are derived
  SymDimVector sizes_ = {0};
  SymDimVector strides_ = {1};
  SymInt storage_offset_ = 0;

  bool strides_valid_ = true; // e.g. for sparse where there are no strides

  SymbolicShapeMeta() = default;
  ~SymbolicShapeMeta() = default;
  SymbolicShapeMeta(const SymbolicShapeMeta& other);
  SymbolicShapeMeta(SymbolicShapeMeta&& other) = delete;
  SymbolicShapeMeta& operator=(const SymbolicShapeMeta& other) = delete;
  SymbolicShapeMeta& operator=(SymbolicShapeMeta&& other) = delete;

  void refresh_numel() {
    // Non-const, don't need to hold mutables_ lock
    available_.fetch_and(~numel_avail);
    numel_ = 1;
  }

  void refresh_contiguous() {
    // Non-const, don't need to hold mutables_ lock
    available_.fetch_and(numel_avail);
    is_contiguous_ = false;
    is_channels_last_contiguous_ = false;
    is_channels_last_3d_contiguous_ = false;
    is_channels_last_ = false;
    is_channels_last_3d_ = false;
    is_non_overlapping_and_dense_ = false;
  }

  int64_t dim() const {
    return static_cast<int64_t>(sizes_.size());
  }

  // Accessors for derived quantities, computed lazily on first access

  bool has_numel() const {
    return available_.load() & numel_avail;
  }
  bool has_is_contiguous() const {
    return available_.load() & is_contiguous_avail;
  }
  bool has_is_channels_last_contiguous() const {
    return available_.load() & is_channels_last_contiguous_avail;
  }
  bool has_is_channels_last_3d_contiguous() const {
    return available_.load() & is_channels_last_3d_contiguous_avail;
  }
  bool has_is_channels_last() const {
    return available_.load() & is_channels_last_avail;
  }
  bool has_is_channels_last_3d() const {
    return available_.load() & is_channels_last_3d_avail;
  }
  bool has_is_non_overlapping_and_dense() const {
    return available_.load() & is_non_overlapping_and_dense_avail;
  }

  // Accessors to cached derived properties
  // DO NOT call with mutables_ lock held
  const SymInt& numel() const {
    if (C10_UNLIKELY(!has_numel())) {
      init_numel();
    }
    return numel_;
  }

  const SymBool& is_contiguous() const {
    if (C10_UNLIKELY(!has_is_contiguous())) {
      init_is_contiguous();
    }
    return is_contiguous_;
  }

  const SymBool& is_channels_last_contiguous() const {
    if (C10_UNLIKELY(!has_is_channels_last_contiguous())) {
      init_is_channels_last_contiguous();
    }
    return is_channels_last_contiguous_;
  }

  const SymBool& is_channels_last_3d_contiguous() const {
    if (C10_UNLIKELY(!has_is_channels_last_3d_contiguous())) {
      init_is_channels_last_3d_contiguous();
    }
    return is_channels_last_3d_contiguous_;
  }

  const SymBool& is_channels_last() const {
    if (C10_UNLIKELY(!has_is_channels_last())) {
      init_is_channels_last();
    }
    return is_channels_last_;
  }

  const SymBool& is_channels_last_3d() const {
    if (C10_UNLIKELY(!has_is_channels_last_3d())) {
      init_is_channels_last_3d();
    }
    return is_channels_last_3d_;
  }

  const SymBool& is_non_overlapping_and_dense() const {
    if (C10_UNLIKELY(!has_is_non_overlapping_and_dense())) {
      init_is_non_overlapping_and_dense();
    }
    return is_non_overlapping_and_dense_;
  }

  // Assumptions so we can short-circuit computation
  // NOTE: Don't need to lock mutables_ since these aren't const
  void assume_contiguous(SymBool val = true) {
    is_contiguous_ = std::move(val);
    available_.fetch_or(is_contiguous_avail);
  }
  void assume_channels_last_contiguous(SymBool val = true) {
    is_contiguous_ = std::move(val);
    available_.fetch_or(is_channels_last_contiguous_avail);
  }
  void assume_channels_last_3d_contiguous(SymBool val = true) {
    is_channels_last_3d_contiguous_ = std::move(val);
    available_.fetch_or(is_channels_last_3d_contiguous_avail);
  }
  void assume_channels_last(SymBool val = true) {
    is_channels_last_ = std::move(val);
    available_.fetch_or(is_channels_last_avail);
  }
  void assume_channels_last_3d(SymBool val = true) {
    is_channels_last_3d_ = std::move(val);
    available_.fetch_or(is_channels_last_3d_avail);
  }
  void assume_non_overlapping_and_dense(SymBool val = true) {
    is_non_overlapping_and_dense_ = std::move(val);
    available_.fetch_or(is_non_overlapping_and_dense_avail);
  }

 private:
  SymBool compute_contiguous() const;
  SymBool compute_channels_last_contiguous_2d() const;
  SymBool compute_channels_last_contiguous_3d() const;
  SymBool compute_strides_like_channels_last_2d() const;
  SymBool compute_strides_like_channels_last_3d() const;
  SymBool compute_non_overlapping_and_dense() const;

  // These are little wrappers over the real compute_ functions that
  // can make use of other contiguity fields to short circuit.
  // They need to be implemented separately for SymBool, as SymBool does
  // not short circuit.
  // TODO: should the SymBool cases avoid the short circuit?  Need to reason
  // if its correct, and reason if the simpler expressions are better for
  // analysis (maybe not!)

  SymBool compute_channels_last_contiguous_3d_dim5() const;
  SymBool compute_channels_last_2d_dim5() const;
  SymBool compute_channels_last_3d_dim5() const;
  SymBool compute_is_non_overlapping_and_dense_dim4() const;
  SymBool compute_is_non_overlapping_and_dense_dim5() const;
  SymBool compute_is_non_overlapping_and_dense_anydim() const;

  void init_numel() const;
  void init_is_contiguous() const;
  void init_is_channels_last_contiguous() const;
  void init_is_channels_last_3d_contiguous() const;
  void init_is_channels_last() const;
  void init_is_channels_last_3d() const;
  void init_is_non_overlapping_and_dense() const;

  // NOTE: These only set if !has_foo()
  void set_numel(SymInt val) const;
  void set_is_contiguous(SymBool val) const;
  void set_is_channels_last_contiguous(SymBool val) const;
  void set_is_channels_last_3d_contiguous(SymBool val) const;
  void set_is_channels_last(SymBool val) const;
  void set_is_channels_last_3d(SymBool val) const;
  void set_is_non_overlapping_and_dense(SymBool val) const;

  // Lazily initialized variables, with the corresponding available_ flag
  // indicating whether the value has been initialized
  mutable std::atomic<int> available_{0};
  enum avail {
    numel_avail = 1 << 0,
    is_contiguous_avail = 1 << 1,
    is_channels_last_contiguous_avail = 1 << 2,
    is_channels_last_3d_contiguous_avail = 1 << 3,
    is_channels_last_avail = 1 << 4,
    is_channels_last_3d_avail = 1 << 5,
    is_non_overlapping_and_dense_avail = 1 << 6,
  };

  // Mutex to prevent races when initializing the variable from const accessors
  mutable std::mutex mutables_;
  mutable SymInt numel_ = 1;
  mutable SymBool is_contiguous_{true};
  mutable SymBool is_channels_last_contiguous_{false};
  mutable SymBool is_channels_last_3d_contiguous_{false};
  mutable SymBool is_channels_last_{false};
  mutable SymBool is_channels_last_3d_{false};
  mutable SymBool is_non_overlapping_and_dense_{true};
};

} // namespace c10
