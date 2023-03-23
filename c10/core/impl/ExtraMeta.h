#pragma once

#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <c10/core/impl/NamedTensorMetaInterface.h>
#include <c10/core/impl/SizesAndStrides.h>
#include <c10/util/DimVector.h>

#include <memory>

namespace c10::impl {

struct C10_API ExtraMeta {
  struct CompositeView {
    // The physical fields describe the tensor when the infallible
    // view was created.
    SizesAndStrides physical;
    std::int64_t physical_storage_offset;
    // The virtual sizes is the size of the infallible view.
    DimVector virtual_sizes;
  };

  using CompositeViews = std::vector<CompositeView>;

  SymDimVector sizes_ = {0};
  SymDimVector strides_ = {1};
  SymInt numel_ = 1;
  SymInt storage_offset_ = 0;
  SymBool is_contiguous_{true};
  SymBool is_channels_last_contiguous_{false};
  SymBool is_channels_last_3d_contiguous_{false};
  SymBool is_channels_last_{false};
  SymBool is_channels_last_3d_{false};
  SymBool is_non_overlapping_and_dense_{true};
  std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta_ = nullptr;
  CompositeViews composite_views;

  ExtraMeta() = default;

  ExtraMeta(
      SymDimVector sizes,
      SymDimVector strides,
      SymInt numel,
      SymInt storage_offset,
      SymBool is_contiguous,
      SymBool is_channels_last_contiguous,
      SymBool is_channels_last_3d_contiguous,
      SymBool is_channels_last,
      SymBool is_channels_last_3d,
      SymBool is_non_overlapping_and_dense,
      std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta,
      CompositeViews composite_views)
      : sizes_(std::move(sizes)),
        strides_(std::move(strides)),
        numel_(std::move(numel)),
        storage_offset_(std::move(storage_offset)),
        is_contiguous_(std::move(is_contiguous)),
        is_channels_last_contiguous_(std::move(is_channels_last_contiguous)),
        is_channels_last_3d_contiguous_(
            std::move(is_channels_last_3d_contiguous)),
        is_channels_last_(std::move(is_channels_last)),
        is_channels_last_3d_(std::move(is_channels_last_3d)),
        is_non_overlapping_and_dense_(std::move(is_non_overlapping_and_dense)),
        named_tensor_meta_(std::move(named_tensor_meta)),
        composite_views(std::move(composite_views)) {}

  std::unique_ptr<ExtraMeta> clone() const {
    return std::make_unique<ExtraMeta>(
        sizes_,
        strides_,
        numel_,
        storage_offset_,
        is_contiguous_,
        is_channels_last_contiguous_,
        is_channels_last_3d_contiguous_,
        is_channels_last_,
        is_channels_last_3d_,
        is_non_overlapping_and_dense_,
        named_tensor_meta_ ? named_tensor_meta_->clone() : nullptr,
        composite_views);
  }
};

} // namespace c10::impl
