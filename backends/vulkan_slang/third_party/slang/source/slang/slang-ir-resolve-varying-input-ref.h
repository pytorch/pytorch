#pragma once

#include "slang-ir.h"

namespace Slang
{
void resolveVaryingInputRef(IRFunc* func);
void resolveVaryingInputRef(IRModule* module);

} // namespace Slang
