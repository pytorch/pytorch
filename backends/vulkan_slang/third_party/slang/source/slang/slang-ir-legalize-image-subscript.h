#pragma once

#include "slang-compiler.h"
#include "slang-ir.h"

namespace Slang
{
class DiagnosticSink;

void legalizeImageSubscript(TargetRequest* target, IRModule* module, DiagnosticSink* sink);
} // namespace Slang
