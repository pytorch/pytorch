// slang-ir-lower-tuple-types.h
#pragma once

#include "slang-ir.h"

namespace Slang
{
struct IRModule;
class DiagnosticSink;

/// Lower tuple types to ordinary `struct`s.
void lowerTuples(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
