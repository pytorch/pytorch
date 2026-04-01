// slang-ir-early-raytracing-intrinsic-simplification.h
#pragma once

#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
struct IRModule;
struct IRGlobalValueWithCode;
class DiagnosticSink;
class TargetProgram;

void replaceLocationIntrinsicsWithRaytracingObject(
    TargetProgram* target,
    IRModule* module,
    DiagnosticSink* sink);
} // namespace Slang