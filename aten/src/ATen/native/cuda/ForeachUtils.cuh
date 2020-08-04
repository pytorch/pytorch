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


bool check_fast_route(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  auto expected_dtype = tensors[0].dtype();

  for (auto t : tensors) {
    if (t.dtype() != expected_dtype) {
      return false;
    }

    if (t.layout() != at::kStrided) {
      return false;
    }

    if (!t.is_non_overlapping_and_dense()) {
      return false;
    }
  }

  return true;
}

bool check_fast_route(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  if (!check_fast_route(tensors)) {
    return false;
  }

  for (auto t : tensors) {
    if ((at::isIntegralType(t.scalar_type(), true) && scalar.isFloatingPoint()) || 
        t.scalar_type() == at::kBool) {
     return false;
    }
  }

  return true;
}

bool check_fast_route(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  auto expected_dtype = tensors1[0].dtype();
  auto expected_device = tensors1[0].device();

  for (int i = 0; i < tensors1.size(); i++) {
    TORCH_CHECK(tensors1[i].sizes() == tensors2[i].sizes(), "Corresponding tensors from tensor lists have different size.");

    if (tensors1[i].dtype() != expected_dtype || 
        tensors2[i].dtype() != expected_dtype) {
      return false;
    }

    if (tensors1[i].device() != expected_device || 
        tensors2[i].device() != expected_device) {
      return false;
    }

    if (tensors1[i].layout() != at::kStrided || 
        tensors2[i].layout() != at::kStrided) {
      return false;
    }

    if (tensors1[i].strides() != tensors2[i].strides()) {
      return false;
    }

    if (!tensors1[i].is_non_overlapping_and_dense() || 
        !tensors2[i].is_non_overlapping_and_dense()) {
      return false;
    }
  }

  return true;
}

}
}} // at::native
