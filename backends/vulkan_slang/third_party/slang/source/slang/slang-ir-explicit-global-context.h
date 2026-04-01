// slang-ir-explicit-global-context.h
#pragma once

#include "slang-compiler.h"

namespace Slang
{
struct IRModule;

/// Collect global-scope variables/paramters to form an explicit context that gets threaded through
void introduceExplicitGlobalContext(IRModule* module, CodeGenTarget target);

} // namespace Slang
