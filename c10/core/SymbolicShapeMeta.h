#pragma once
#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <c10/util/DimVector.h>

#include <bitset>

namespace c10 {

class C10_API SymbolicShapeMeta {
 public:
  // Basic metadata from which other quantities are derived
  SymDimVector sizes_ = {0};
  SymDimVector strides_ = {1};
  SymInt storage_offset_ = 0;

  bool strides_valid_ = true; // e.g. for sparse where there are no strides

  void refresh_numel() {
    available_.reset(numel_avail);
    numel_ = 1;
  }

  void refresh_contiguous() {
    if (strides_valid_) {
      available_.reset(is_contiguous_avail);
      is_contiguous_ = true;
      available_.reset(is_channels_last_contiguous_avail);
      is_channels_last_contiguous_ = false;
      available_.reset(is_channels_last_3d_contiguous_avail);
      is_channels_last_3d_contiguous_ = false;
      available_.reset(is_channels_last_avail);
      is_channels_last_ = false;
      available_.reset(is_channels_last_3d_avail);
      is_channels_last_3d_ = false;
      available_.reset(is_non_overlapping_and_dense_avail);
      is_non_overlapping_and_dense_ = true;
    } else {
      // Unstrided tensor types are never contiguous
      available_.set(is_contiguous_avail);
      is_contiguous_ = false;
      available_.set(is_channels_last_contiguous_avail);
      is_channels_last_contiguous_ = false;
      available_.set(is_channels_last_3d_contiguous_avail);
      is_channels_last_3d_contiguous_ = false;
      available_.set(is_channels_last_avail);
      is_channels_last_ = false;
      available_.set(is_channels_last_3d_avail);
      is_channels_last_3d_ = false;
      available_.set(is_non_overlapping_and_dense_avail);
      is_non_overlapping_and_dense_ = false;
    }
  }

  int64_t dim() const {
    return static_cast<int64_t>(sizes_.size());
  }

  // Accessors for derived quantities, computed lazily on first access

  SymInt numel() const {
    if (available_.test(numel_avail)) {
      return numel_;
    }
    numel_ = multiply_integers(sizes_);
    available_.set(numel_avail);
    return numel_;
  }

  SymBool is_contiguous() const {
    if (available_.test(is_contiguous_avail)) {
      return is_contiguous_;
    }

    is_contiguous_ = compute_contiguous();
    available_.set(is_contiguous_avail);
    return is_contiguous_;
  }

  SymBool is_channels_last_contiguous() const {
    if (available_.test(is_channels_last_contiguous_avail)) {
      return is_channels_last_contiguous_;
    }
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
    available_.set(is_channels_last_contiguous_avail);
    return is_channels_last_contiguous_;
  }

  SymBool is_channels_last_3d_contiguous() const {
    if (available_.test(is_channels_last_3d_contiguous_avail)) {
      return is_channels_last_3d_contiguous_;
    }
    is_channels_last_3d_contiguous_ = [&] {
      switch (dim()) {
        case 5:
          return compute_channels_last_contiguous_3d_dim5();
        default:
          return SymBool{false};
      }
    }();
    available_.set(is_channels_last_3d_contiguous_avail);
    return is_channels_last_3d_contiguous_;
  }

  SymBool is_channels_last() const {
    if (available_.test(is_channels_last_avail)) {
      return is_channels_last_;
    }
    is_channels_last_ = [&] {
      switch (dim()) {
        case 5:
          return compute_channels_last_2d_dim5();
        case 4:
          return compute_strides_like_channels_last_2d();
        default:
          return SymBool{false};
      }
    }();
    available_.set(is_channels_last_avail);
    return is_channels_last_;
  }

  SymBool is_channels_last_3d() const {
    if (available_.test(is_channels_last_3d_avail)) {
      return is_channels_last_3d_;
    }
    is_channels_last_3d_ = [&] {
      switch (dim()) {
        case 5:
          return compute_channels_last_3d_dim5();
        default:
          return SymBool{false};
      }
    }();
    available_.set(is_channels_last_3d_avail);
    return is_channels_last_3d_;
  }

  SymBool is_non_overlapping_and_dense() const {
    if (available_.test(is_non_overlapping_and_dense_avail)) {
      return is_non_overlapping_and_dense_;
    }
    is_non_overlapping_and_dense_ = [&] {
      switch (dim()) {
        case 5:
          return compute_is_non_overlapping_and_dense_dim5();
        case 4:
          return compute_is_non_overlapping_and_dense_dim4();
        default:
          return compute_is_non_overlapping_and_dense_anydim();
      }
    }();
    available_.set(is_non_overlapping_and_dense_avail);
    return is_non_overlapping_and_dense_;
  }

  // Assumptions so we can short-circuit computation
  void assume_contiguous() {
    available_.set(is_contiguous_avail);
    is_contiguous_ = true;
  }
  void assume_channels_last_contiguous() {
    available_.set(is_channels_last_contiguous_avail);
    is_channels_last_contiguous_ = true;
  }
  void assume_channels_last_3d_contiguous() {
    available_.set(is_channels_last_3d_contiguous_avail);
    is_channels_last_3d_contiguous_ = true;
  }
  void assume_channels_last() {
    available_.set(is_channels_last_avail);
    is_channels_last_ = true;
  }
  void assume_channels_last_3d() {
    available_.set(is_channels_last_3d_avail);
    is_channels_last_3d_ = true;
  }
  void assume_non_overlapping_and_dense() {
    available_.set(is_non_overlapping_and_dense_avail);
    is_non_overlapping_and_dense_ = true;
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

  // Lazily initialized variables, with the corresponding available_ flag
  // indicating whether the value has been initialized
  mutable std::bitset<7> available_;
  enum avail {
    numel_avail = 0,
    is_contiguous_avail,
    is_channels_last_contiguous_avail,
    is_channels_last_3d_contiguous_avail,
    is_channels_last_avail,
    is_channels_last_3d_avail,
    is_non_overlapping_and_dense_avail,
  };

  mutable SymInt numel_ = 1;
  mutable SymBool is_contiguous_{true};
  mutable SymBool is_channels_last_contiguous_{false};
  mutable SymBool is_channels_last_3d_contiguous_{false};
  mutable SymBool is_channels_last_{false};
  mutable SymBool is_channels_last_3d_{false};
  mutable SymBool is_non_overlapping_and_dense_{true};
};

} // namespace c10
