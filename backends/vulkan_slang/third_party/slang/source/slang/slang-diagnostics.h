#ifndef SLANG_DIAGNOSTICS_H
#define SLANG_DIAGNOSTICS_H

#include "../compiler-core/slang-diagnostic-sink.h"
#include "../compiler-core/slang-source-loc.h"
#include "../compiler-core/slang-token.h"
#include "../core/slang-basic.h"
#include "../core/slang-writer.h"
#include "slang.h"

namespace Slang
{
DiagnosticInfo const* findDiagnosticByName(UnownedStringSlice const& name);
const DiagnosticsLookup* getDiagnosticsLookup();
SlangResult overrideDiagnostic(
    DiagnosticSink* sink,
    DiagnosticSink* outDiagnostic,
    const UnownedStringSlice& identifier,
    Severity originalSeverity,
    Severity overrideSeverity);
SlangResult overrideDiagnostics(
    DiagnosticSink* sink,
    DiagnosticSink* outDiagnostic,
    const UnownedStringSlice& identifierList,
    Severity originalSeverity,
    Severity overrideSeverity);

namespace Diagnostics
{
#define DIAGNOSTIC(id, severity, name, messageFormat) extern const DiagnosticInfo name;
#include "slang-diagnostic-defs.h"
} // namespace Diagnostics
} // namespace Slang

#ifdef _DEBUG
#define SLANG_INTERNAL_ERROR(sink, pos)             \
    (sink)->diagnose(                               \
        Slang::SourceLoc(__LINE__, 0, 0, __FILE__), \
        Slang::Diagnostics::internalCompilerError)
#define SLANG_UNIMPLEMENTED(sink, pos, what)        \
    (sink)->diagnose(                               \
        Slang::SourceLoc(__LINE__, 0, 0, __FILE__), \
        Slang::Diagnostics::unimplemented,          \
        what)

#else
#define SLANG_INTERNAL_ERROR(sink, pos) \
    (sink)->diagnose(pos, Slang::Diagnostics::internalCompilerError)
#define SLANG_UNIMPLEMENTED(sink, pos, what) \
    (sink)->diagnose(pos, Slang::Diagnostics::unimplemented, what)

#endif

#define SLANG_DIAGNOSE_UNEXPECTED(sink, pos, message) \
    (sink)->diagnose(pos, Slang::Diagnostics::unexpected, message)

#endif
