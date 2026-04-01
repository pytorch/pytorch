// slang-ir-addr-inst-elimination.h
#pragma once

#include "slang-ir.h"

namespace Slang
{
class DiagnosticSink;

SlangResult eliminateAddressInsts(IRFunc* func, DiagnosticSink* sink);

} // namespace Slang
