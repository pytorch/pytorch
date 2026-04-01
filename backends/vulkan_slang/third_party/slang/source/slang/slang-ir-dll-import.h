// slang-ir-dll-import.h
#pragma once

namespace Slang
{
struct IRModule;
class DiagnosticSink;
class TargetProgram;
/// Generate implementations for functions marked as [DllImport].
void generateDllImportFuncs(TargetProgram* targetReq, IRModule* module, DiagnosticSink* sink);
} // namespace Slang
