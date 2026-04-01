// slang-ir-cleanup-void.h
#pragma once

#include "slang-ir.h"

namespace Slang
{
struct IRModule;
class DiagnosticSink;

/// Cleanup all uses of void and void types.
void cleanUpVoidType(IRModule* module);

} // namespace Slang
