#pragma once

#ifdef AOTI_STANDALONE
#include <torch/csrc/inductor/aoti_standalone/cuda/utils.h>
#else
#include <torch/csrc/inductor/aoti_runtime/utils_cuda.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>
#endif
