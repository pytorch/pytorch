#pragma once

#include "slang/slang-diagnostics.h"

namespace CppParse
{
using namespace Slang;

namespace CPPDiagnostics
{

#define DIAGNOSTIC(id, severity, name, messageFormat) extern const DiagnosticInfo name;
#include "diagnostic-defs.h"

} // namespace CPPDiagnostics
} // namespace CppParse
