// slang-ir-legalize-mesh-outputs.h
#pragma once

#include "slang-ir-insts.h"

namespace Slang
{
struct IRModule;

// Turn opaque mesh output types into regular arrays
void legalizeMeshOutputTypes(IRModule* module);
} // namespace Slang
