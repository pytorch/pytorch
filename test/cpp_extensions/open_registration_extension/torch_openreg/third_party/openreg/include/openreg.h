#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum orError_t { orSuccess = 0, orErrorUnknown = 1 } orError_t;

typedef enum orMemcpyKind {
  orMemcpyHostToHost = 0,
  orMemcpyHostToDevice = 1,
  orMemcpyDeviceToHost = 2,
  orMemcpyDeviceToDevice = 3
} orMemcpyKind;

typedef enum orMemoryType {
  orMemoryTypeUnmanaged = 0,
  orMemoryTypeHost = 1,
  orMemoryTypeDevice = 2
} orMemoryType;

struct orPointerAttributes {
  orMemoryType type = orMemoryType::orMemoryTypeUnmanaged;
  int device;
  void* pointer;
  size_t size;
};

orError_t orMalloc(void** devPtr, size_t size);
orError_t orFree(void* devPtr);
orError_t orMallocHost(void** hostPtr, size_t size);
orError_t orFreeHost(void* hostPtr);
orError_t orMemcpy(void* dst, const void* src, size_t count, orMemcpyKind kind);
orError_t orMemoryUnprotect(void* devPtr);
orError_t orMemoryProtect(void* devPtr);

orError_t orGetDeviceCount(int* count);
orError_t orSetDevice(int device);
orError_t orGetDevice(int* device);

orError_t orPointerGetAttributes(
    orPointerAttributes* attributes,
    const void* ptr);

#ifdef __cplusplus
} // extern "C"
#endif
