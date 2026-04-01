// slang-ir-diff-call.h
#pragma once

namespace Slang
{
struct IRModule;

struct IRDerivativeCallProcessOptions
{
    // Nothing for now..
};

bool processDerivativeCalls(
    IRModule* module,
    IRDerivativeCallProcessOptions const& options = IRDerivativeCallProcessOptions());

} // namespace Slang