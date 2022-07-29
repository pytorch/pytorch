#include <c10/core/impl/SizesAndStrides.h>

namespace c10 {
namespace impl {

void SizesAndStrides::resizeSlowPath(
    const size_t newSize,
    const size_t oldSize) {
  if (newSize <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        !isInline(),
        "resizeSlowPath called when fast path should have been hit!");
    SymInt* tempStorage = outOfLineStorage_;
    for (size_t i = 0; i < C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE; i++) {
      inlineStorage_[i] = std::move(tempStorage[i]);
      inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE + i] = std::move(tempStorage[oldSize]);
    }
    // CANNOT USE freeOutOfLineStorage() HERE! outOfLineStorage_
    // HAS BEEN OVERWRITTEN!
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    delete[] tempStorage;
  } else {
    if (isInline()) {
      // CANNOT USE allocateOutOfLineStorage(newSize) HERE! WOULD
      // OVERWRITE inlineStorage_!
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      SymInt* tempStorage = new SymInt[storageElems(newSize)];
      TORCH_CHECK(
          tempStorage,
          "Could not allocate memory to change Tensor SizesAndStrides!");
      const auto elemsToCopy = oldSize;
      const auto elemsToZero = (newSize > oldSize)
          ? (newSize - oldSize)
          : 0;
      for (size_t i = 0; i < elemsToCopy; i++) {
        tempStorage[i] = std::move(inlineStorage_[i]);
        tempStorage[newSize + i] = std::move(inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE + i]);
      }
      for (size_t i = 0; i < elemsToZero; i++) {
        tempStorage[oldSize + i] = 0;
        tempStorage[newSize + oldSize + i] = 0;
      }
      outOfLineStorage_ = tempStorage;
    } else {
      const bool isGrowing = oldSize < newSize;
      if (isGrowing) {
        // Resize before shifting so that we have room.
        resizeOutOfLineStorage(newSize);
      }
      // Shift the old strides to their new starting point. Note
      // that this does not occur in the inline path above because
      // the stride starting point is not moving.
      if (isGrowing) {
        std::move_backward(
          outOfLineStorage_ + oldSize,
          outOfLineStorage_ + oldSize + oldSize,
          outOfLineStorage_ + newSize
        );
      } else {
        std::move(
          outOfLineStorage_ + oldSize,
          outOfLineStorage_ + oldSize + newSize,
          outOfLineStorage_ + newSize
        );
      }
      if (!isGrowing) {
        // Resize after shifting so that we don't lose data.
        resizeOutOfLineStorage(newSize);
      } else {
        // Zero the end of the sizes portion.
        const auto elemsToZero = newSize - oldSize;
        for (size_t i = 0; i < elemsToZero; i++) {
          outOfLineStorage_[oldSize + i] = 0;
          outOfLineStorage_[newSize + oldSize + i] = 0;
        }
      }
    }
  }
  size_ = newSize;
}

} // namespace impl
} // namespace c10
