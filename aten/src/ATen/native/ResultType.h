#pragma once

#include <c10/core/ScalarType.h>

// Type determination rules for mixed-type operations.
//
// The result type is computed using the operands with the following precedence:
//
// 1) Tensors with dim 1 or higher
// 2) Tensors with dim 0 that aren't wrapped numbers (e.g. `tensor(5)`)
// 3) Tensors with dim 0 that are wrapped numbers (e.g. `5`)
//
// So if there are any tensors of dim 1 or higher, then 0-dim tensors do not
// affect the result type. This behavior was chosen to preserve backwards
// compatibility and is *likely to change* in the near future.
// (See https://github.com/pytorch/pytorch/issues/9515)
//
// If an operation needs to produce an output of a specific type, this will
// raise an error if the operation arguments yield a different type.

namespace at {

struct Type;

CAFFE2_API Type& resultType(TensorList tensors);

CAFFE2_API Type& resultTypeForOutput(Tensor output, TensorList inputs);

}  // namespace at
