#ifndef SLANG_JSON_DIAGNOSTICS_H
#define SLANG_JSON_DIAGNOSTICS_H

#include "../core/slang-basic.h"
#include "../core/slang-writer.h"
#include "slang-diagnostic-sink.h"
#include "slang-source-loc.h"
#include "slang-token.h"
#include "slang.h"

namespace Slang
{

DiagnosticsLookup* getJSONDiagnosticsLookup();

namespace JSONDiagnostics
{
#define DIAGNOSTIC(id, severity, name, messageFormat) extern const DiagnosticInfo name;
#include "slang-json-diagnostic-defs.h"
} // namespace JSONDiagnostics

} // namespace Slang

#endif
