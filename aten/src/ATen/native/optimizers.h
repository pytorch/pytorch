#pragma once

#include <ATen/ATen.h>
#include <ATen/Generator.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

using sparse_adam_step_fn = void (*)(
    double /* alpha */,
    double /* beta1 */,
    double /* beta2 */,
    double /* eps */,
    int64_t /* step */,
    TensorIterator* /* it */);

DECLARE_DISPATCH(sparse_adam_step_fn, sparse_adam_step_stub);

} // namespace native
} // namespace at
