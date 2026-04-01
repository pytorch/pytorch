// slang-ir-composite-reg-to-mem.h
#pragma once

namespace Slang
{
struct IRModule;
struct IRCall;
struct IRInst;
struct IRFunc;

/// Convert parameters of composite type into pointers and modify the callsites accordingly.
void convertCompositeTypeParametersToPointers(IRModule* module);
} // namespace Slang
