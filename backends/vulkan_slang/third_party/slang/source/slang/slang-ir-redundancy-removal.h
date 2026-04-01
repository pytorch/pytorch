// slang-ir-redundancy-removal.h
#pragma once
#include "slang-compiler.h"

namespace Slang
{
struct IRModule;
struct IRGlobalValueWithCode;

bool removeRedundancy(IRModule* module, bool hoistLoopInvariantInsts);
bool removeRedundancyInFunc(IRGlobalValueWithCode* func, bool hoistLoopInvariantInsts);

bool eliminateRedundantLoadStore(IRGlobalValueWithCode* func);

void removeAvailableInDownstreamModuleDecorations(CodeGenTarget target, IRModule* module);
} // namespace Slang
