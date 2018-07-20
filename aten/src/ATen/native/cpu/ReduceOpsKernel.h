#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/optional.h>

namespace at {
namespace native {

using reduce_fn = void(*)(Tensor &, const Tensor &, at::optional<int64_t>);

extern DispatchStub<reduce_fn> sum_kernel;
extern DispatchStub<reduce_fn> prod_kernel;

}
}
