#pragma once
#include <ATen/Config.h>
#if AT_CUDA_ENABLED()
#define WITH_CUDA
#endif

#include "containers.h"
#include "optimizers.h"
#include "serialization.h"
