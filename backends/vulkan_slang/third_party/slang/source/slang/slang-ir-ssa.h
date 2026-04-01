// slang-ir-ssa.h
#pragma once

namespace Slang
{
struct IRModule;
struct IRGlobalValueWithCode;
struct IRInst;

bool constructSSA(IRModule* module, IRGlobalValueWithCode* globalVal);
bool constructSSA(IRModule* module);
bool constructSSA(IRInst* globalVal);
} // namespace Slang
