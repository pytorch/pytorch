// slang-ir-any-value-marshalling.h
#pragma once

#include "../core/slang-common.h"

namespace Slang
{
struct IRType;
struct SharedGenericsLoweringContext;

/// Generates functions that pack and unpack `AnyValue`s, and replaces
/// all `IRPackAnyValue` and `IRUnpackAnyValue` instructions with calls
/// to these packing/unpacking functions.
/// This is a sub-pass of lower-generics.
void generateAnyValueMarshallingFunctions(SharedGenericsLoweringContext* sharedContext);


/// Get the AnyValue size required to hold a value of `type`.
SlangInt getAnyValueSize(IRType* type);
} // namespace Slang
