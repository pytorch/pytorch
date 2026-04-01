// slang-ir-lower-generics.h
#pragma once

#include "slang-ir.h"

namespace Slang
{
struct IRModule;
class DiagnosticSink;
class TargetProgram;

/// Lower generic and interface-based code to ordinary types and functions using
/// dynamic dispatch mechanisms.
void lowerGenerics(TargetProgram* targetReq, IRModule* module, DiagnosticSink* sink);

// Clean up any generic-related IR insts that are no longer needed. Called when
// it has been determined that no more dynamic dispatch code will be generated.
void cleanupGenerics(TargetProgram* targetReq, IRModule* module, DiagnosticSink* sink);
} // namespace Slang
