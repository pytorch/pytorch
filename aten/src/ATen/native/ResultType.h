#pragma once

#include <ATen/core/Type.h>

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

namespace at {

struct Type;

CAFFE2_API Type& resultType(TensorList tensors);

}  // namespace at
