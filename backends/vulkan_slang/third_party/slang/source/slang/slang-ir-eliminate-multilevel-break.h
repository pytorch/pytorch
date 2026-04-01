// slang-ir-eliminate-multi-level-break.h
#pragma once

namespace Slang
{
struct IRModule;
struct IRGlobalValueWithCode;

void eliminateMultiLevelBreak(IRModule* module);
void eliminateMultiLevelBreakForFunc(IRModule* module, IRGlobalValueWithCode* func);

} // namespace Slang
