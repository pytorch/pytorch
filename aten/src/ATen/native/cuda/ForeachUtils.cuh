#pragma once
#include <ATen/ATen.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
namespace at { 
namespace native {
namespace {

static constexpr int64_t kILP = 4;
static constexpr int64_t kChunkSize = 65536;
static constexpr int64_t kBlockSize = 512;

template<typename T>
__device__ __forceinline__ bool is_aligned(T* p){
  return ((uint64_t)p) % (kILP * sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  using LT = at::native::memory::aligned_vector<T, kILP>;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

}

bool check_fast_route(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  auto expected_dtype = tensors[0].dtype();
  auto expected_device = tensors[0].device();

  for (auto t : tensors) {
    if (t.dtype() != expected_dtype) {
      return false;
    }

    if (t.device() != expected_device) {
      return false;
    }

    if (t.layout() != at::kStrided) {
      return false;
    }

    if (!t.is_non_overlapping_and_dense()) {
      return false;
    }

    if ((at::isIntegralType(t.scalar_type(), true) && scalar.isFloatingPoint()) || 
        t.scalar_type() == at::kBool) {
     return false;
    }
  }

  return true;
}
}} // at::native
