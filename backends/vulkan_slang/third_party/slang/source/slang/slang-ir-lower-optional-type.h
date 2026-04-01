// slang-ir-lower-optional-type.h
#pragma once

#include "slang-ir.h"

namespace Slang
{
struct IRModule;
class DiagnosticSink;

/// Lower `IROptionalType<T,E>` types to ordinary `struct`s.
void lowerOptionalType(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
