#pragma once

namespace Slang
{

struct IRModule;
class TargetProgram;
class DiagnosticSink;

/// Replace `GetDynamicResourceHeap` insts with an actual array of resources.
void lowerDynamicResourceHeap(TargetProgram* targetProgram, IRModule* module, DiagnosticSink* sink);

} // namespace Slang
