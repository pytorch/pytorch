// slang-core-diagnostics.cpp
#include "slang-core-diagnostics.h"

namespace Slang
{

namespace MiscDiagnostics
{
#define DIAGNOSTIC(id, severity, name, messageFormat) \
    const DiagnosticInfo name = {id, Severity::severity, #name, messageFormat};
#include "slang-misc-diagnostic-defs.h"
#undef DIAGNOSTIC
} // namespace MiscDiagnostics

static const DiagnosticInfo* const kMiscDiagnostics[] = {
#define DIAGNOSTIC(id, severity, name, messageFormat) &MiscDiagnostics::name,
#include "slang-misc-diagnostic-defs.h"
#undef DIAGNOSTIC
};


namespace LexerDiagnostics
{
#define DIAGNOSTIC(id, severity, name, messageFormat) \
    const DiagnosticInfo name = {id, Severity::severity, #name, messageFormat};
#include "slang-lexer-diagnostic-defs.h"
#undef DIAGNOSTIC
} // namespace LexerDiagnostics

static const DiagnosticInfo* const kLexerDiagnostics[] = {
#define DIAGNOSTIC(id, severity, name, messageFormat) &LexerDiagnostics::name,
#include "slang-lexer-diagnostic-defs.h"
#undef DIAGNOSTIC
};

static DiagnosticsLookup* _newCoreDiagnosticsLookup()
{
    auto lookup = new DiagnosticsLookup;
    lookup->add(kMiscDiagnostics, SLANG_COUNT_OF(kMiscDiagnostics));
    lookup->add(kLexerDiagnostics, SLANG_COUNT_OF(kLexerDiagnostics));

    return lookup;
}

DiagnosticsLookup* getCoreDiagnosticsLookup()
{
    static RefPtr<DiagnosticsLookup> s_lookup = _newCoreDiagnosticsLookup();
    return s_lookup;
}

} // namespace Slang
