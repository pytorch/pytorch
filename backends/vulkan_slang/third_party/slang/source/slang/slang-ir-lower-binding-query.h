// slang-ir-lower-binding-query.h
#pragma once

#include "slang-ir.h"

namespace Slang
{
struct IRModule;
class DiagnosticSink;

/// Lower the `getRegisterIndex` and `getRegisterSpace` intrinsics.
///
/// These operations semantically return binding information on their
/// argument, which must be a value of an opaque type (resource,
/// sampler, etc.). These operations can only ever work on values that
/// derive (in one way or another) from a global shader parameter.
///
void lowerBindingQueries(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
