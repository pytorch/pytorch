// slang-ir-single-return.h
#pragma once

namespace Slang
{
struct IRModule;
struct IRGlobalValueWithCode;

// Convert the CFG of `func` to have only a single `return` at the end.
void convertFuncToSingleReturnForm(IRModule* module, IRGlobalValueWithCode* func);

bool isSingleReturnFunc(IRGlobalValueWithCode* func);

int getReturnCount(IRGlobalValueWithCode* func);
} // namespace Slang
