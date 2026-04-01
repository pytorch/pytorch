#pragma once

#include "slang-ir.h"

namespace Slang
{
struct IRModule;
class DiagnosticSink;

void lowerGLSLShaderStorageBufferObjectsToPointers(IRModule* module, DiagnosticSink* sink);

void lowerGLSLShaderStorageBufferObjectsToStructuredBuffers(IRModule* module, DiagnosticSink* sink);
} // namespace Slang
