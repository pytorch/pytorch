#pragma once
#include <ATen/ATen.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
namespace {
static CONSTEXPR_EXCEPT_WIN_CUDA int64_t kILP = 4;
static CONSTEXPR_EXCEPT_WIN_CUDA int64_t kChunkSize = 65536;
static CONSTEXPR_EXCEPT_WIN_CUDA int64_t kBlockSize = 1024;

template<typename T>
__device__ __forceinline__ bool is_aligned(T* p){
  return ((uint64_t)p) % (kILP * sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  using LT = at::native::memory::aligned_vector<T, kILP>;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

} // namespace
