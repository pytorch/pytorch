// slang-ir-legalize-array-return-type.h
#pragma once

#include "slang-ir-insts.h"

namespace Slang
{
struct IRModule;

// Turn array-typed return values into `out` parameters for backends that does not
// support arrays in return values.
void legalizeArrayReturnType(IRModule* module);
} // namespace Slang
