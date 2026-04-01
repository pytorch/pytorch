// slang-ir-dll-export.h
#pragma once

namespace Slang
{
struct IRModule;
class DiagnosticSink;
/// Generate wrappers for functions marked as [DllExport].
void generateDllExportFuncs(IRModule* module, DiagnosticSink* sink);
} // namespace Slang
