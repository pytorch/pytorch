// slang-ir-check-differentiability.h
#pragma once

#include "slang-ir.h"

namespace Slang
{
struct IRModule;
class DiagnosticSink;

// Check all auto diff usages are valid.
void checkAutoDiffUsages(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
