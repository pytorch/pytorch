// slang-ir-lower-result-type.h
#pragma once

#include "slang-ir.h"

namespace Slang
{
struct IRModule;
class DiagnosticSink;

/// Lower `IRResultType<T,E>` types to ordinary `struct`s.
void lowerResultType(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
