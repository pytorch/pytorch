// slang-ir-specialize-dispatch.h
#pragma once

namespace Slang
{
struct SharedGenericsLoweringContext;

/// Modifies the body of interface dispatch functions to use branching instead
/// of function pointer calls to implement the dynamic dispatch logic.
/// This is only used on GPU targets where function pointers are not supported
/// or are not efficient.
void specializeDispatchFunctions(SharedGenericsLoweringContext* sharedContext);
} // namespace Slang
