// slang-ir-entry-point-uniform.h
#pragma once

#include "slang-compiler.h"

namespace Slang
{
struct IRModule;

struct CollectEntryPointUniformParamsOptions
{
    // TODO(JS): Not sure if it makes sense to initialize to true or false. Go with false as
    // seems to fit usage.
    bool alwaysCreateCollectedParam = false;
    TargetRequest* targetReq = nullptr;
};

/// Collect entry point uniform parameters into a wrapper `struct` and/or buffer
void collectEntryPointUniformParams(
    IRModule* module,
    CollectEntryPointUniformParamsOptions const& options);

/// Move any uniform parameters of entry points to the global scope instead.
void moveEntryPointUniformParamsToGlobalScope(IRModule* module);

} // namespace Slang
