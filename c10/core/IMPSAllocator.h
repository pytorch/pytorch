#pragma once

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include <c10/core/Allocator.h>
#include <c10/macros/Export.h>
#include <c10/util/ArrayRef.h>

namespace c10 {

// this is a public interface to access MPSAllocator.
// Do not declare methods that would depend on MPS or Metal frameworks.
class C10_API IMPSAllocator : public c10::Allocator {
 public:
  // see the comments in MPSAllocator.h for the description of these methods.
  virtual void emptyCache() const = 0;
  virtual void freeInactiveBuffers() const = 0;
  virtual ssize_t getUnalignedBufferSize(const void* ptr) const = 0;
  virtual IntArrayRef getBufferShape(const void* ptr) const = 0;
  virtual id_t getBufferId(const void* ptr) const = 0;
  virtual void setBufferShape(const void* ptr, const IntArrayRef& shape)
      const = 0;
  virtual bool isSharedBuffer(const void* ptr) const = 0;
  virtual bool isSharedBufferCPUPtr(const void* ptr) const = 0;
  virtual bool isSharedStorageSupported() const = 0;
  virtual c10::DataPtr allocScalarBufferWithValue(void* value, size_t size)
      const = 0;
  virtual std::string formatSize(size_t size) const = 0;
  virtual void setLowWatermarkRatio(double ratio) const = 0;
  virtual void setHighWatermarkRatio(double ratio) const = 0;
  virtual ssize_t getLowWatermarkValue() const = 0;
  virtual size_t getLowWatermarkLimit() const = 0;
  virtual size_t getHighWatermarkLimit() const = 0;
  virtual size_t getTotalAllocatedMemory() const = 0;
  virtual size_t getCurrentAllocatedMemory() const = 0;
  virtual size_t getDriverAllocatedMemory() const = 0;
  virtual size_t getRecommendedMaxMemory() const = 0;
  virtual std::pair<void*, uint32_t> getSharedCPUPtrFromDevicePtr(
      const void* ptr) const = 0;
  virtual std::pair<void*, uint32_t> getSharedDevicePtrFromCPUPtr(
      const void* ptr) const = 0;
  virtual bool recordEvents(c10::ArrayRef<const void*> buffers) const = 0;
  virtual bool waitForEvents(c10::ArrayRef<const void*> buffers) const = 0;
  virtual void* get_cpu_ptr_from_device_ptr(void* device_ptr) const = 0;
  virtual const void* get_cpu_ptr_from_device_ptr(
      const void* device_ptr) const = 0;
  virtual void* get_device_ptr_from_cpu_ptr(void* cpu_ptr) const = 0;
  virtual const void* get_device_ptr_from_cpu_ptr(
      const void* cpu_ptr) const = 0;
};

} // namespace c10
