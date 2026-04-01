// slang-ir-legalize-uniform-buffer-load.h
#pragma once

#include "slang-ir-insts.h"

namespace Slang
{
struct IRModule;

// Legalize a load(IRUniformParameterGroupType) into a makeStruct(load(fieldAddr),...) for glsl.
void legalizeUniformBufferLoad(IRModule* module);
} // namespace Slang
