// slang-ir-lower-error-handling.h
#pragma once

namespace Slang
{

struct IRModule;
class DiagnosticSink;

/// Lower error handling related opcodes and function calls to use standard control flow.
/// A function with an error code type will be translated into a function that returns
/// `Result<T,E>`, which can be further lowered to standard return values and `out` parameters in a
/// separate pass. Call sites (`IRTryCall`) to error-throwing function will be rewritten to conform
/// to the new function signature. `IRThrow` will be replaced with `IRReturn(IRMakeErrorResult(e))`.
///
void lowerErrorHandling(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
