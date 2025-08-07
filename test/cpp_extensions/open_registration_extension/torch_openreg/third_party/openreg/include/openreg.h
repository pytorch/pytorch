#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum orError_t {
  orSuccess = 0,
  orErrorUnknown = 1,
  orErrorNotReady = 2
} orError_t;

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

typedef enum orEventFlags {
  orEventDisableTiming = 0x0,
  orEventEnableTiming = 0x1,
} orEventFlags;

// Memory
orError_t orMalloc(void** devPtr, size_t size);
orError_t orFree(void* devPtr);
orError_t orMallocHost(void** hostPtr, size_t size);
orError_t orFreeHost(void* hostPtr);
orError_t orMemcpy(void* dst, const void* src, size_t count, orMemcpyKind kind);
orError_t orPointerGetAttributes(
    orPointerAttributes* attributes,
    const void* ptr);
orError_t orMemoryUnprotect(void* devPtr);
orError_t orMemoryProtect(void* devPtr);

// Device
orError_t orGetDeviceCount(int* count);
orError_t orSetDevice(int device);
orError_t orGetDevice(int* device);

struct orStream;
struct orEvent;
typedef struct orStream* orStream_t;
typedef struct orEvent* orEvent_t;

// Stream
orError_t orStreamCreateWithPriority(
    orStream_t* stream,
    unsigned int flags,
    int priority);
orError_t orStreamCreate(orStream_t* stream);
orError_t orStreamGetPriority(orStream_t stream, int* priority);
orError_t orStreamDestroy(orStream_t stream);
orError_t orStreamQuery(orStream_t stream);
orError_t orStreamSynchronize(orStream_t stream);
orError_t orStreamWaitEvent(
    orStream_t stream,
    orEvent_t event,
    unsigned int flags);

// Event
orError_t orEventCreateWithFlags(orEvent_t* event, unsigned int flags);
orError_t orEventDestroy(orEvent_t event);
orError_t orEventRecord(orEvent_t event, orStream_t stream);
orError_t orEventSynchronize(orEvent_t event);
orError_t orEventQuery(orEvent_t event);
orError_t orEventElapsedTime(float* ms, orEvent_t start, orEvent_t end);

orError_t orDeviceGetStreamPriorityRange(
    int* leastPriority,
    int* greatestPriority);
orError_t orDeviceSynchronize(void);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus

#define OPENREG_H

template <typename Func, typename... Args>
inline orError_t orLaunchKernel(
    orStream* stream,
    Func&& kernel_func,
    Args&&... args);

#include "openreg.inl"

#endif
