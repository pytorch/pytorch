// slang-ir-lower-enum-type.h
#pragma once

#include "slang-ir.h"

namespace Slang
{
struct IRModule;
class DiagnosticSink;

/// Lower `IREnumType` to their underlying integer types.
void lowerEnumType(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
