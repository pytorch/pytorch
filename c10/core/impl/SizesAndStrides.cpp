#include <c10/core/impl/SizesAndStrides.h>

namespace c10::impl {

void SizesAndStrides::resizeSlowPath(
    const size_t newSize,
    const size_t oldSize) {
  if (newSize <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        !isInline(),
        "resizeSlowPath called when fast path should have been hit!");
    int64_t* tempStorage = outOfLineStorage_;
    memcpy(
        &inlineStorage_[0],
        &tempStorage[0],
        C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE * sizeof(inlineStorage_[0]));
    memcpy(
        &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE],
        &tempStorage[oldSize],
        C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE * sizeof(inlineStorage_[0]));
    // CANNOT USE freeOutOfLineStorage() HERE! outOfLineStorage_
    // HAS BEEN OVERWRITTEN!
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    free(tempStorage);
  } else {
    if (isInline()) {
      // CANNOT USE allocateOutOfLineStorage(newSize) HERE! WOULD
      // OVERWRITE inlineStorage_!
      int64_t* tempStorage =
          // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
          static_cast<int64_t*>(malloc(storageBytes(newSize)));
      TORCH_CHECK(
          tempStorage,
          "Could not allocate memory to change Tensor SizesAndStrides!");
      const auto bytesToCopy = oldSize * sizeof(inlineStorage_[0]);
      const auto bytesToZero = (newSize > oldSize)
          ? (newSize - oldSize) * sizeof(tempStorage[0])
          : 0;
      memcpy(&tempStorage[0], &inlineStorage_[0], bytesToCopy);
      if (bytesToZero) {
        memset(&tempStorage[oldSize], 0, bytesToZero);
      }
      memcpy(
          &tempStorage[newSize],
          &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE],
          bytesToCopy);
      if (bytesToZero) {
        memset(&tempStorage[newSize + oldSize], 0, bytesToZero);
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
      memmove(
          outOfLineStorage_ + newSize,
          outOfLineStorage_ + oldSize,
          std::min(oldSize, newSize) * sizeof(outOfLineStorage_[0]));
      if (!isGrowing) {
        // Resize after shifting so that we don't lose data.
        resizeOutOfLineStorage(newSize);
      } else {
        // Zero the end of the sizes portion.
        const auto bytesToZero =
            (newSize - oldSize) * sizeof(outOfLineStorage_[0]);
        memset(&outOfLineStorage_[oldSize], 0, bytesToZero);
        memset(&outOfLineStorage_[newSize + oldSize], 0, bytesToZero);
      }
    }
  }
  size_ = newSize;
}

} // namespace c10::impl
