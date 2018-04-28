#pragma once
#include <ATen/Config.h>
#if AT_CUDA_ENABLED()
#define WITH_CUDA
#endif

#include "torch/containers.h"
#include "torch/optimizers.h"
#include "torch/serialization.h"
