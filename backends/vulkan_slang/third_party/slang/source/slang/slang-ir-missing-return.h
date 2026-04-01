// slang-ir-missing-return.h
#pragma once

namespace Slang
{
class DiagnosticSink;
struct IRModule;
enum class CodeGenTarget;

/// This IR check pass is performed twice, once during IR lowering(with no code gen target
/// specified) and once during linking(with code gen target specified).
///
/// Some code gen targets allow missing returns while some do not. The first pass, where no target
/// is specified, emit warnings on missing returns, and the second pass may additionally emit errors
/// when compiling for targets that do not support missing returns.
///
/// On the second pass, `diagnoseWarning` is set to false to suppress warnings, ensuring that only
/// errors are emitted. This prevents duplicate warnings from appearing in both passes.
void checkForMissingReturns(
    IRModule* module,
    DiagnosticSink* sink,
    CodeGenTarget target,
    bool diagnoseWarning);

} // namespace Slang
