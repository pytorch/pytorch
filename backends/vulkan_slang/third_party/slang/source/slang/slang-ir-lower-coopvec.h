#pragma once

namespace Slang
{

struct IRModule;
class DiagnosticSink;

/// Lower Cooperative Vectors to ordinary arrays
void lowerCooperativeVectors(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
