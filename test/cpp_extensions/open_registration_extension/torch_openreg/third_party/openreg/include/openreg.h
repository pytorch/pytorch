#pragma once

#include <cstddef>

#ifdef _WIN32
  #define OPENREG_EXPORT __declspec(dllexport)
#else
  #define OPENREG_EXPORT __attribute__((visibility("default")))
#endif

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

OPENREG_EXPORT orError_t orMalloc(void** devPtr, size_t size);
OPENREG_EXPORT orError_t orFree(void* devPtr);
OPENREG_EXPORT orError_t orMallocHost(void** hostPtr, size_t size);
OPENREG_EXPORT orError_t orFreeHost(void* hostPtr);
OPENREG_EXPORT orError_t orMemcpy(void* dst, const void* src, size_t count, orMemcpyKind kind);
OPENREG_EXPORT orError_t orMemoryUnprotect(void* devPtr);
OPENREG_EXPORT orError_t orMemoryProtect(void* devPtr);

OPENREG_EXPORT orError_t orGetDeviceCount(int* count);
OPENREG_EXPORT orError_t orSetDevice(int device);
OPENREG_EXPORT orError_t orGetDevice(int* device);

OPENREG_EXPORT orError_t orPointerGetAttributes(
    orPointerAttributes* attributes,
    const void* ptr);

#ifdef __cplusplus
} // extern "C"
#endif
