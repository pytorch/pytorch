#pragma once

#include <ATen/ATen.h>

// Prefer not to use these functions directly.  Instead prefer to use the
// corresponding mobile operators that this module registers with, and exposes
// through, c10 whenever possible in order to decouple op creation from op
// execution.  The underlying computation in many NN operators is performed in
// phases, all of which do not need to run at the same frequency.  Decoupling op
// creation from execution allows for one-time computations to be cached and
// factored out resulting in time savings per calls to forward() - something
// this API cannot do.  Furthermore, this API does not allow for fusion of
// non-linear operators that, again, is something that the exposed c10 mobile
// operators can handle.

namespace at {
namespace native {
namespace mobile {
namespace cpu {

bool available();
bool initialize();
bool deinitialize();

//
// Add
//

bool use_add(
    const Tensor& input1,
    const Tensor& input2);

Tensor& add(
    Tensor& output,
    const Tensor& input1,
    const Tensor& input2);

Tensor add(
    const Tensor& input1,
    const Tensor& input2);

//
// Clamp
//

bool use_clamp(
    const Tensor& input,
    Scalar output_min,
    Scalar output_max);

Tensor& clamp(
    Tensor& output,
    const Tensor& input,
    Scalar output_min,
    Scalar output_max);

Tensor clamp(
    const Tensor& input,
    Scalar output_min,
    Scalar output_max);

//
// Convolution
//

bool use_convolution(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool transposed);

Tensor convolution(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool transposed);

//
// Linear
//

bool use_linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias);

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias);

//
// Pooling
//

bool use_max_pool(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    bool ceil_mode);

Tensor max_pool(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    bool ceil_mode);

//
// ReLU
//

bool use_relu(
    const Tensor& input);

Tensor& relu(
    Tensor& result,
    const Tensor& input);

Tensor relu(
    const Tensor& input);

} // namespace cpu
} // namespace mobile
} // namespace native
} // namespace at
