#pragma once

#include "slang-ir.h"

namespace Slang
{
class DiagnosticSink;

void legalizeIRForWGSL(IRModule* module, DiagnosticSink* sink);
} // namespace Slang
