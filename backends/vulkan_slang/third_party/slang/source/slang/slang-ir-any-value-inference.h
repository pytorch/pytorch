// slang-ir-any-value-inference.h
#pragma once

#include "../core/slang-common.h"
#include "slang-compiler.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
void inferAnyValueSizeWhereNecessary(TargetProgram* targetProgram, IRModule* module);
}
