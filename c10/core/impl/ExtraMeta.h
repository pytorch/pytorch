#pragma once

#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <c10/core/impl/NamedTensorMetaInterface.h>
#include <c10/util/DimVector.h>

#include <memory>

namespace c10::impl {

struct C10_API ExtraMeta {
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
      std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta)
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
        named_tensor_meta_(std::move(named_tensor_meta)) {}

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
        named_tensor_meta_ ? named_tensor_meta_->clone() : nullptr);
  }
};

} // namespace c10::impl
