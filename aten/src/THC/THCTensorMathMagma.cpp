#include <THC/THCGeneral.h>
#include <ATen/cuda/detail/CUDAHooks.h>

#ifdef USE_MAGMA
#include <magma_v2.h>
#endif

namespace {
void _THCMagma_init() {
#ifdef USE_MAGMA
  magma_init();
#endif
}

struct Initializer {
  Initializer() {
    ::at::cuda::detail::THCMagma_init = _THCMagma_init;
  };
} initializer;
} // anonymous namespace
