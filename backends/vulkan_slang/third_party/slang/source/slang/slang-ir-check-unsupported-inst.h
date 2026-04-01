#pragma once

namespace Slang
{
struct IRModule;
class DiagnosticSink;
class TargetRequest;

void checkUnsupportedInst(TargetRequest* target, IRModule* module, DiagnosticSink* sink);
} // namespace Slang
