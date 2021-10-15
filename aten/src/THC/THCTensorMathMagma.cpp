#include <THC/THCGeneral.h>
#include <ATen/cuda/detail/CUDAHooks.h>
#include <ATen/cuda/CUDAConfig.h>

#if AT_MAGMA_ENABLED()
#include <magma_v2.h>
#endif

namespace {
void _THCMagma_init() {
#if AT_MAGMA_ENABLED()
  magma_init();
#endif
}

struct Initializer {
  Initializer() {
    ::at::cuda::detail::THCMagma_init = _THCMagma_init;
  };
} initializer;
} // anonymous namespace
