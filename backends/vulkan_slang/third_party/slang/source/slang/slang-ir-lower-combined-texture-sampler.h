#pragma once

#include "slang-ir.h"

namespace Slang
{
struct CodeGenContext;
struct IRModule;
class DiagnosticSink;

// Lower combined texture sampler types to structs.
void lowerCombinedTextureSamplers(
    CodeGenContext* codeGenContext,
    IRModule* module,
    DiagnosticSink* sink);
} // namespace Slang
