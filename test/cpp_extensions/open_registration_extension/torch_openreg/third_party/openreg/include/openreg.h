#pragma once

#include <cstddef>
#include <stdint.h>

#ifdef _WIN32
#define OPENREG_EXPORT __declspec(dllexport)
#else
#define OPENREG_EXPORT __attribute__((visibility("default")))
#endif

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
};

typedef enum orEventFlags {
  orEventDisableTiming = 0x0,
  orEventEnableTiming = 0x1,
} orEventFlags;

struct orStream;
struct orEvent;
typedef struct orStream* orStream_t;
typedef struct orEvent* orEvent_t;

// Memory
OPENREG_EXPORT orError_t orMalloc(void** devPtr, size_t size);
OPENREG_EXPORT orError_t orFree(void* devPtr);
OPENREG_EXPORT orError_t orMallocHost(void** hostPtr, size_t size);
OPENREG_EXPORT orError_t orFreeHost(void* hostPtr);
OPENREG_EXPORT orError_t
orMemcpy(void* dst, const void* src, size_t count, orMemcpyKind kind);
OPENREG_EXPORT orError_t orMemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    orMemcpyKind kind,
    orStream_t stream);
OPENREG_EXPORT orError_t
orPointerGetAttributes(orPointerAttributes* attributes, const void* ptr);
OPENREG_EXPORT orError_t orMemoryUnprotect(void* devPtr);
OPENREG_EXPORT orError_t orMemoryProtect(void* devPtr);

// IPC via POSIX shared memory (not available on Windows)
#ifndef _WIN32
// Max byte length of the shm name string (including null terminator).
#define OR_IPC_HANDLE_MAX_LEN 64

// Snapshot devPtr's allocation into a POSIX shm object; write its name into
// name_out (name_buf_len >= OR_IPC_HANDLE_MAX_LEN required) and the byte
// delta from the allocation base to devPtr into *offset_out.
OPENREG_EXPORT orError_t orGetIpcMemHandle(
    void* devPtr,
    char* name_out,
    size_t name_buf_len,
    ptrdiff_t* offset_out);

// Open the shm, mmap it, and unlink the name. *ptr_out is the mapped base;
// *size_out is the page-aligned block size. Call orCloseIpcMemHandle to unmap.
OPENREG_EXPORT orError_t orOpenIpcMemHandle(
    void** ptr_out,
    const char* name,
    size_t* size_out);

// Unmap the region. size must equal *size_out from orOpenIpcMemHandle.
OPENREG_EXPORT orError_t orCloseIpcMemHandle(void* ptr, size_t size);
#endif // !_WIN32

// Device
OPENREG_EXPORT orError_t orGetDeviceCount(int* count);
OPENREG_EXPORT orError_t orSetDevice(int device);
OPENREG_EXPORT orError_t orGetDevice(int* device);
OPENREG_EXPORT orError_t
orDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
OPENREG_EXPORT orError_t orDeviceSynchronize(void);

// Stream
OPENREG_EXPORT orError_t orStreamCreateWithPriority(
    orStream_t* stream,
    unsigned int flags,
    int priority);
OPENREG_EXPORT orError_t orStreamCreate(orStream_t* stream);
OPENREG_EXPORT orError_t orStreamGetPriority(orStream_t stream, int* priority);
OPENREG_EXPORT orError_t orStreamDestroy(orStream_t stream);
OPENREG_EXPORT orError_t orStreamQuery(orStream_t stream);
OPENREG_EXPORT orError_t orStreamSynchronize(orStream_t stream);
OPENREG_EXPORT orError_t
orStreamWaitEvent(orStream_t stream, orEvent_t event, unsigned int flags);

// Event
OPENREG_EXPORT orError_t
orEventCreateWithFlags(orEvent_t* event, unsigned int flags);
OPENREG_EXPORT orError_t orEventCreate(orEvent_t* event);
OPENREG_EXPORT orError_t orEventDestroy(orEvent_t event);
OPENREG_EXPORT orError_t orEventRecord(orEvent_t event, orStream_t stream);
OPENREG_EXPORT orError_t orEventSynchronize(orEvent_t event);
OPENREG_EXPORT orError_t orEventQuery(orEvent_t event);
OPENREG_EXPORT orError_t
orEventElapsedTime(float* ms, orEvent_t start, orEvent_t end);

// Activity tracing
OPENREG_EXPORT orError_t orActivityEnableTracing(void);
OPENREG_EXPORT orError_t orActivityDisableTracing(void);
OPENREG_EXPORT orError_t orActivityPushExternalCorrelationId(uint64_t id);
OPENREG_EXPORT orError_t orActivityPopExternalCorrelationId(uint64_t* id);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus

#define OPENREG_H
#include "openreg.inl"

#endif
