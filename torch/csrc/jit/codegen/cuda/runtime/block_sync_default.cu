
// Default block synchronization. Just use __barrier_sync
namespace block_sync {

__forceinline__ __device__ void init() {}

// Thread-block synchronization
__forceinline__ __device__ void sync() {
#ifdef __HIP_PLATFORM_HCC__
  __syncthreads();
#else
  __barrier_sync(0);
#endif
}

} // namespace block_sync
