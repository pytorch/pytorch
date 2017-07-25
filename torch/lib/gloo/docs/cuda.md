# NVIDIA GPU support
Gloo includes several collective algorithm implementations that work directly with NVIDIA GPU buffers. These take advantage of overlapping host and GPU operations to decrease overall latency.

GPU-aware algorithms require CUDA 7 or newer for various CUDA and NCCL features.

## Serializing GPU device operations
Gloo leverages CUDA streams to sequence operations on a single GPU device without blocking other concurrent activity. Before calling any of the gloo collective functions that operate on GPU buffers, the calling code should
* Ensure the GPU buffer inputs are synchronized and valid, or
* Pass the associated `cudaStream_t`(s) to the gloo collective function so that it can serialize its usage of the inputs.

If no `cudaStream_t`(s) are passed to the gloo collective function, GPU buffer outputs are valid when the gloo collective function returns. Otherwise, the calling code must synchronize with the streams before using the GPU buffer outputs, i.e., explicitly with `cudaStreamSynchronize()` or inserting dependent operations in the stream.

See CUDA documentation for additional information about using streams.

```cpp
void broadcastZeros(
    std::shared_ptr<::gloo::Context>& context,
    int rank,
    float* devicePtr,
    size_t count) {
  // Allocate a stream to serialize GPU device operations
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Zero local GPU device buffer asynchronously
  cudaMemsetAsync(devicePtr, 0, count, stream);

  // Broadcast the buffer to participating machines
  gloo::CudaBroadcastOneToAll<float> broadcast(
    context, devicePtr, count, rank, stream);
  broadcast.run();

  // Wait until the broadcast is complete
  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);
}
```

## Synchronizing GPU memory allocation
Overlapping calls to `cudaMalloc()` or `cudaFree()` may result in deadlock. Gloo and any calling code must coordinate memory allocations. Calling code should
1. Pass a shared `std::mutex` into `gloo::CudaShared::setMutex()` before calling any other gloo functions.
2. Always acquire the mutex before calling CUDA memory allocation functions.

```cpp
// Define a mutex to synchronize calls to cudaMalloc/cudaFree
std::mutex m;

// Share the mutex with gloo
gloo::CudaShared::setMutex(&m);

// Always call cudaMalloc/cudaFree while holding the mutex
void* allocateCudaMemory(size_t bytes) {
  std::lock_guard<std::mutex> lock(m);
  void* ptr;
  cudaMalloc(&ptr, bytes);
  return ptr;
}
```
