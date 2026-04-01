#ifndef SLANG_CORE_DIAGNOSTICS_H
#define SLANG_CORE_DIAGNOSTICS_H

#include "../../source/compiler-core/slang-diagnostic-sink.h"
#include "../../source/compiler-core/slang-source-loc.h"
#include "../../source/core/slang-basic.h"
#include "../../source/core/slang-writer.h"
#include "slang.h"

namespace Slang
{

DiagnosticsLookup* getCoreDiagnosticsLookup();

namespace RenderTestDiagnostics
{
#define DIAGNOSTIC(id, severity, name, messageFormat) extern const DiagnosticInfo name;
#include "diagnostic-defs.h"
} // namespace RenderTestDiagnostics

} // namespace Slang

#endif
