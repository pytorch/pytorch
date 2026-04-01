// slang-ir-uniformity.h
#pragma once

namespace Slang
{
struct IRModule;
class DiagnosticSink;

void validateUniformity(IRModule* module, DiagnosticSink* sink);
} // namespace Slang
