#pragma once

#include "THC/THC.h"

// This file creates a fake allocator that just throws exceptions if
// it is actually used.

// state passed to the allocator is the std::function<void(void*)> called
// when the blob is release by ATen

namespace at {

static cuda_fixed_malloc(void *, void**, size_t, cudaStream_t) {
  runtime_error("attempting to resize a tensor view of an external blob");
}

static cpu_fixed_realloc(void*, void**, size_t, size_t, cudaStream_t) {
  runtime_error("attempting to resize a tensor view of an external blob");
}

static cuda_fixed_free(void * state, void * allocation) {
    auto on_release = static_cast<std::function<void(void*)>*>(state);
    (*on_release)(allocation);
    delete on_release;
}

static cuda_fixed_emptyCache(void*) {
  runtime_error("?? attempting to resize a tensor view of an external blob");
}

static cuda_fixed_cacheInfo(void*, int, size_t*, size_t*) {
  runtime_error("?? attempting to resize a tensor view of an external blob");
}


static THCDeviceAllocator CUDA_fixed_allocator = {
  cuda_fixed_malloc, cuda_fixed_realloc, cuda_fixed_free, cuda_fixed_emptyCache,
  cuda_fixed_cacheInfo
};


}
