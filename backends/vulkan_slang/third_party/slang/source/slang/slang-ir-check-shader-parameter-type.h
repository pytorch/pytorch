#pragma once

namespace Slang
{
struct IRModule;
class DiagnosticSink;
class TargetRequest;

void checkForInvalidShaderParameterType(
    TargetRequest* targetReq,
    IRModule* module,
    DiagnosticSink* sink);
} // namespace Slang
