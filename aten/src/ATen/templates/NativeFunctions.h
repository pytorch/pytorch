#pragma once

#include "ATen/ATen.h"
#include <tuple>
#include <vector>

#if AT_CUDNN_ENABLED()
#include "ATen/cudnn/AffineGridGenerator.h"
#include "ATen/cudnn/BatchNorm.h"
#include "ATen/cudnn/Conv.h"
#include "ATen/cudnn/GridSampler.h"
#include "ATen/cudnn/Types.h"
#endif

namespace at {
namespace native {

${native_function_declarations}

}
}

#include "ATen/native/NativeFunctions-inl.h"
