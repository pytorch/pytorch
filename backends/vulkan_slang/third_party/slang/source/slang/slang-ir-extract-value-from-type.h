// slang-ir-extract-value-from-type.h
#pragma once

#include "slang-ir.h"
#include "slang-type-layout.h"

namespace Slang
{

// Emit code using builder that yields an `IRInst` representing a value of `size` bytes
// starting at `offset` in `src`. `src` must be a value of `struct`, array, vector or basic type.
// `size` can be either 1, 2 or 4. The resulting `IRInst` value will have an `uint` type.
IRInst* extractValueAtOffset(
    IRBuilder& builder,
    TargetProgram* targetReq,
    IRInst* src,
    uint32_t offset,
    uint32_t size);

} // namespace Slang
