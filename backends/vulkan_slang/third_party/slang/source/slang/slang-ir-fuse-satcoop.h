#pragma once

namespace Slang
{
struct CodeGenContext;
struct IRModule;
struct IRType;

/// Fuse adjacent calls to saturated_cooperation
void fuseCallsToSaturatedCooperation(IRModule* module);
} // namespace Slang
