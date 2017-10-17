#include "torch/csrc/cuda/THCP.h"

// Declare/Define the expansion functions that have THCState.  Note that we
// still need to define the CPU-type versions because the copy functions that
// copy from GPU to CPU type have a THCState.

#define CUDA_EXPAND 1

#include "torch/csrc/expand_utils.h"
#include "torch/csrc/generic/expand_utils-inl.h"

#undef CUDA_EXPAND
