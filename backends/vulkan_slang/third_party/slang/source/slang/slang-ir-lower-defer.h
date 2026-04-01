// slang-ir-lower-defer.h
#pragma once

#include "slang-ir.h"

namespace Slang
{
struct IRModule;
class DiagnosticSink;

/// Lower the `defer` statements.
///
/// Duplicates the child instructions of each `defer` to the end of each
/// dominated block whose terminator jumps to a location that is not dominated
/// by the `defer`. Also removes all `IRDefer` instructions after that.
void lowerDefer(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
