#pragma once

#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/native/DispatchStub.h"
#include "ATen/cpu/vml.h"

namespace at { namespace native {

DECLARE_DISPATCH(Tensor(*)(const Tensor &, const Tensor &, int64_t, int64_t), grid_sampler_2d_cpu_kernel);

}}  // namespace at::native
