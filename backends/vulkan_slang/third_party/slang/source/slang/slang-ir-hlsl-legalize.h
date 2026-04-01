// slang-ir-hlsl-legalize.h
#pragma once
#include "../core/slang-list.h"
#include "slang-compiler.h"

namespace Slang
{

class DiagnosticSink;
class Session;

struct IRFunc;
struct IRModule;

void legalizeNonStructParameterToStructForHLSL(IRModule* module);

} // namespace Slang