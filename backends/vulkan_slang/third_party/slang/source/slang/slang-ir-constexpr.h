// slang-ir-constexpr.h
#pragma once

namespace Slang
{
class DiagnosticSink;
struct IRModule;

void propagateConstExpr(IRModule* module, DiagnosticSink* sink);
} // namespace Slang
