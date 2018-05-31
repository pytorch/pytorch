#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include "THCStorage.h"

#include "ATen/ScalarType.h"
#include "ATen/ScalarTypeUtils.h"
#include "ATen/cuda/CUDATypeConversion.cuh"
#include <atomic>

#include "generic/THCStorage.hpp"
#include "THCGenerateAllTypes.h"
