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
    numel_ = multiply_integers(sizes_);
  }

  void refresh_contiguous() {
    if (strides_valid_) {
      is_contiguous_ = true;
      is_channels_last_contiguous_ = false;
      is_channels_last_3d_contiguous_ = false;
      is_channels_last_ = false;
      is_channels_last_3d_ = false;
      is_non_overlapping_and_dense_ = true;
    } else {
      switch (dim()) {
        case 4: {
          is_contiguous_ = compute_contiguous();
          is_channels_last_contiguous_ = compute_channels_last_contiguous_2d();
          is_channels_last_3d_contiguous_ = false;
          is_channels_last_ = compute_strides_like_channels_last_2d();
          is_channels_last_3d_ = false;
          is_non_overlapping_and_dense_ =
              compute_is_non_overlapping_and_dense_dim4();
          break;
        }
        case 5: {
          is_contiguous_ = compute_contiguous();
          is_channels_last_contiguous_ = compute_channels_last_contiguous_2d();
          is_channels_last_3d_contiguous_ =
              compute_channels_last_contiguous_3d_dim5();
          is_channels_last_ = compute_channels_last_2d_dim5();
          is_channels_last_3d_ = compute_channels_last_3d_dim5();
          is_non_overlapping_and_dense_ =
              compute_is_non_overlapping_and_dense_dim5();
          break;
        }
        default: {
          // is_channels_last_ and is_channels_last_3d_ are suggested
          // memory_format. Being channels_last_contiguous doesn't necessarily
          // mean the tensor is strided like channels_last: for strides on
          // channel dimension could suggest desired memory_layout, but it
          // doesn't affect memory storage
          is_contiguous_ = compute_contiguous();
          is_channels_last_contiguous_ = false;
          is_channels_last_3d_contiguous_ = false;
          is_channels_last_ = false;
          is_channels_last_3d_ = false;
          is_non_overlapping_and_dense_ =
              compute_is_non_overlapping_and_dense_anydim();
          break;
        }
      }
    }
  }

  int64_t dim() const {
    return static_cast<int64_t>(sizes_.size());
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

 public:
  SymInt numel_ = 1;
  SymBool is_contiguous_{true};
  SymBool is_channels_last_contiguous_{false};
  SymBool is_channels_last_3d_contiguous_{false};
  SymBool is_channels_last_{false};
  SymBool is_channels_last_3d_{false};
  SymBool is_non_overlapping_and_dense_{true};
};

} // namespace c10
