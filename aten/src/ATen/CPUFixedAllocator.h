#pragma once

#include "TH/TH.h"
#include "ATen/Error.h"

// This file creates a fake allocator that just throws exceptions if
// it is actually used.

// state passed to the allocator is the std::function<void(void*)> called
// when the blob is release by ATen

namespace at {

static cpu_fixed_malloc(void *, ptrdiff_t) {
  AT_ERROR("attempting to resize a tensor view of an external blob");
}

static cpu_fixed_realloc(void *, void*, ptrdiff_t) {
  AT_ERROR("attempting to resize a tensor view of an external blob");
}

static cpu_fixed_free(void * state, void * allocation) {
    auto on_release = static_cast<std::function<void(void*)>*>(state);
    (*on_release)(allocation);
    delete on_release;
}

static THAllocator CPU_fixed_allocator =
  { cpu_fixed_malloc, cpu_fixed_realloc, cpu_fixed_free };

}
