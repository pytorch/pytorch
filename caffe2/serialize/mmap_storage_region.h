#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/UniqueVoidPtr.h>

namespace caffe2 {
namespace serialize {

class TORCH_API MmapStorageRegion : public c10::intrusive_ptr_target {
 public:
  explicit MmapStorageRegion(const std::string& filename);
  ~MmapStorageRegion() override;
  /**
   * Returns a DataPtr to the specified offset in this object's mmap region
   * on the specified device
   */
  at::DataPtr getData(size_t offset, Device device);
  static void deleter(void* ctx);
  /**
   * Returns whether or not mmap (and therefore MmapStorageRegion) is supported
   * by the platform being used
   */
  static bool isSupportedByPlatform();
 private:
  uint8_t* region;
  size_t size;
};

} // namespace serialize
} // namespace caffe2
