#ifndef SLANG_CORE_DIAGNOSTICS_H
#define SLANG_CORE_DIAGNOSTICS_H

#include "../core/slang-basic.h"
#include "../core/slang-writer.h"
#include "slang-diagnostic-sink.h"
#include "slang-source-loc.h"
#include "slang-token.h"
#include "slang.h"

namespace Slang
{

DiagnosticsLookup* getCoreDiagnosticsLookup();

namespace MiscDiagnostics
{
#define DIAGNOSTIC(id, severity, name, messageFormat) extern const DiagnosticInfo name;
#include "slang-misc-diagnostic-defs.h"
} // namespace MiscDiagnostics

namespace LexerDiagnostics
{
#define DIAGNOSTIC(id, severity, name, messageFormat) extern const DiagnosticInfo name;
#include "slang-lexer-diagnostic-defs.h"
} // namespace LexerDiagnostics

} // namespace Slang

#endif
