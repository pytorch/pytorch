#include <THC/THCGeneral.h>
#include <THC/THCTensorMath.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCTensorMathMagma.cuh>
#include <THC/THCTensor.hpp>
#include <THC/THCStorage.hpp>
#include <algorithm>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/cuda/detail/CUDAHooks.h>

#ifdef USE_MAGMA
#include <magma_v2.h>
#endif

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#define NoMagma(name) "No CUDA implementation of '" #name "'. Install MAGMA and rebuild cutorch (http://icl.cs.utk.edu/magma/)"

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

#include <THC/generic/THCTensorMathMagma.cu>
#include <THC/THCGenerateAllTypes.h>
