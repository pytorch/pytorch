// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/distributed/c10d/nccl2/CudaApi.hpp>

namespace c10d::nccl2 {

// DefaultCudaApi implementation

cudaError_t DefaultCudaApi::setDevice(int device) {
  return cudaSetDevice(device);
}

cudaError_t DefaultCudaApi::getDevice(int* device) {
  return cudaGetDevice(device);
}

cudaError_t DefaultCudaApi::getDeviceProperties(
    cudaDeviceProp* prop,
    int device) {
  return cudaGetDeviceProperties(prop, device);
}

cudaError_t DefaultCudaApi::memGetInfo(size_t* free, size_t* total) {
  return cudaMemGetInfo(free, total);
}

cudaError_t DefaultCudaApi::getDeviceCount(int* count) {
  return cudaGetDeviceCount(count);
}

cudaError_t DefaultCudaApi::getStreamPriorityRange(
    int* leastPriority,
    int* greatestPriority) {
  return cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
}

cudaError_t DefaultCudaApi::streamCreateWithPriority(
    cudaStream_t* pStream,
    unsigned int flags,
    int priority) {
  return cudaStreamCreateWithPriority(pStream, flags, priority);
}

cudaError_t DefaultCudaApi::streamDestroy(cudaStream_t stream) {
  return cudaStreamDestroy(stream);
}

cudaError_t DefaultCudaApi::streamWaitEvent(
    cudaStream_t stream,
    cudaEvent_t event,
    unsigned int flags) {
  return cudaStreamWaitEvent(stream, event, flags);
}

cudaStream_t DefaultCudaApi::getCurrentCUDAStream(int device_index) {
  return at::cuda::getCurrentCUDAStream(device_index).stream();
}

cudaError_t DefaultCudaApi::streamSynchronize(cudaStream_t stream) {
  return cudaStreamSynchronize(stream);
}

cudaError_t DefaultCudaApi::streamIsCapturing(
    cudaStream_t stream,
    cudaStreamCaptureStatus* pCaptureStatus) {
  return cudaStreamIsCapturing(stream, pCaptureStatus);
}

cudaError_t DefaultCudaApi::streamGetCaptureInfo(
    cudaStream_t stream,
    cudaStreamCaptureStatus* pCaptureStatus,
    unsigned long long* pId) {
  return cudaStreamGetCaptureInfo(stream, pCaptureStatus, pId);
}

cudaError_t DefaultCudaApi::userObjectCreate(
    cudaUserObject_t* object_out,
    void* ptr,
    cudaHostFn_t destroy,
    unsigned int initialRefcount,
    unsigned int flags) {
  return cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags);
}

cudaError_t DefaultCudaApi::graphRetainUserObject(
    cudaGraph_t graph,
    cudaUserObject_t object,
    unsigned int count,
    unsigned int flags) {
  return cudaGraphRetainUserObject(graph, object, count, flags);
}

cudaError_t DefaultCudaApi::userObjectRelease(
    cudaUserObject_t object,
    unsigned int count) {
  return cudaUserObjectRelease(object, count);
}

cudaError_t DefaultCudaApi::launchHostFunc(
    cudaStream_t stream,
    cudaHostFn_t fn,
    void* userData) {
  return cudaLaunchHostFunc(stream, fn, userData);
}

cudaError_t DefaultCudaApi::streamGetCaptureInfo_v2(
    cudaStream_t stream,
    cudaStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out,
    cudaGraph_t* graph_out,
    const cudaGraphNode_t** dependencies_out,
    size_t* numDependencies_out) {
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 13000)
  return cudaStreamGetCaptureInfo(
      stream,
      captureStatus_out,
      id_out,
      graph_out,
      dependencies_out,
      nullptr,
      numDependencies_out);
#else
  return cudaStreamGetCaptureInfo_v2(
      stream,
      captureStatus_out,
      id_out,
      graph_out,
      dependencies_out,
      numDependencies_out);
#endif
}

cudaError_t DefaultCudaApi::threadExchangeStreamCaptureMode(
    enum cudaStreamCaptureMode* mode) {
  return cudaThreadExchangeStreamCaptureMode(mode);
}

cudaError_t DefaultCudaApi::hostAlloc(
    void** pHost,
    size_t size,
    unsigned int flags) {
  return cudaHostAlloc(pHost, size, flags);
}

cudaError_t DefaultCudaApi::hostFree(void* ptr) {
  return cudaFreeHost(ptr);
}

cudaError_t DefaultCudaApi::malloc(void** devPtr, size_t size) {
  return cudaMalloc(devPtr, size);
}

cudaError_t DefaultCudaApi::free(void* devPtr) {
  return cudaFree(devPtr);
}

cudaError_t DefaultCudaApi::memcpy(
    void* dst,
    const void* src,
    size_t count,
    cudaMemcpyKind kind) {
  return cudaMemcpy(dst, src, count, kind);
}

cudaError_t DefaultCudaApi::memcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    cudaMemcpyKind kind,
    cudaStream_t stream) {
  return cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t DefaultCudaApi::eventCreate(cudaEvent_t* event) {
  return cudaEventCreate(event);
}

cudaError_t DefaultCudaApi::eventCreateWithFlags(
    cudaEvent_t* event,
    unsigned int flags) {
  return cudaEventCreateWithFlags(event, flags);
}

cudaError_t DefaultCudaApi::eventDestroy(cudaEvent_t event) {
  return cudaEventDestroy(event);
}

cudaError_t DefaultCudaApi::eventRecord(
    cudaEvent_t event,
    cudaStream_t stream) {
  return cudaEventRecord(event, stream);
}

cudaError_t DefaultCudaApi::eventRecordWithFlags(
    cudaEvent_t event,
    cudaStream_t stream,
    unsigned int flags) {
  return cudaEventRecordWithFlags(event, stream, flags);
}

cudaError_t DefaultCudaApi::eventQuery(cudaEvent_t event) {
  return cudaEventQuery(event);
}

const char* DefaultCudaApi::getErrorString(cudaError_t error) {
  return cudaGetErrorString(error);
}

} // namespace c10d::nccl2
