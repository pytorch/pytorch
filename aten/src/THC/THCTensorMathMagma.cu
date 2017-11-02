#include "THCGeneral.h"
#include "THCTensorMath.h"
#include "THCTensorCopy.h"
#include "THCTensorMathMagma.cuh"
#include <algorithm>

#ifdef USE_MAGMA
#include <magma.h>
#else
#include "THCBlas.h"
#endif

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#define NoMagma(name) "No CUDA implementation of '" #name "'. Install MAGMA and rebuild cutorch (http://icl.cs.utk.edu/magma/)"

void THCMagma_init(THCState *state)
{
#ifdef USE_MAGMA
  magma_init();
#endif
}

#include "generic/THCTensorMathMagma.cu"
#include "THCGenerateAllTypes.h"
