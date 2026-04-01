// slang-ir-use-uninitialized-out-param.h
#pragma once

namespace Slang
{
class DiagnosticSink;
struct IRModule;

void checkForUsingUninitializedValues(IRModule* module, DiagnosticSink* sink);
} // namespace Slang
