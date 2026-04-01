// slang-ir-explicit-global-init.h
#pragma once

namespace Slang
{
struct IRModule;
class TargetProgram;

/// Move initialization logic off of global variables and onto each entry point
void moveGlobalVarInitializationToEntryPoints(IRModule* module, TargetProgram* targetProgram);
} // namespace Slang
