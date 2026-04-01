// slang-ir-lower-bit-cast.h
#pragma once

// This file defines an IR pass that lowers a BitCast<T>(U) operation, where T and U are struct
// types, into a series of bit-cast operations on basic-typed elements.

namespace Slang
{

struct IRModule;
class DiagnosticSink;
class TargetProgram;

void lowerBitCast(TargetProgram* targetReq, IRModule* module, DiagnosticSink* sink);

} // namespace Slang
