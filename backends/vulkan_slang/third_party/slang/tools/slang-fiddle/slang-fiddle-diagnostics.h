// slang-fiddle-diagnostics.h
#pragma once

#include "compiler-core/slang-diagnostic-sink.h"
#include "slang/slang-diagnostics.h"


namespace fiddle
{
using namespace Slang;

namespace Diagnostics
{

#define DIAGNOSTIC(id, severity, name, messageFormat) extern const DiagnosticInfo name;
#include "slang-fiddle-diagnostic-defs.h"
} // namespace Diagnostics
} // namespace fiddle
