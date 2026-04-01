#pragma once

namespace Slang
{
struct IRModule;
class DiagnosticSink;
class TargetRequest;

void checkForRecursiveTypes(IRModule* module, DiagnosticSink* sink);

void checkForRecursiveFunctions(TargetRequest* target, IRModule* module, DiagnosticSink* sink);

} // namespace Slang
