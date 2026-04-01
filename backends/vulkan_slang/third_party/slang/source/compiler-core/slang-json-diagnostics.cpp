// slang-json-diagnostics.cpp
#include "slang-json-diagnostics.h"

namespace Slang
{

namespace JSONDiagnostics
{
#define DIAGNOSTIC(id, severity, name, messageFormat) \
    const DiagnosticInfo name = {id, Severity::severity, #name, messageFormat};
#include "slang-json-diagnostic-defs.h"
#undef DIAGNOSTIC
} // namespace JSONDiagnostics

static const DiagnosticInfo* const kJSONDiagnostics[] = {
#define DIAGNOSTIC(id, severity, name, messageFormat) &JSONDiagnostics::name,
#include "slang-json-diagnostic-defs.h"
#undef DIAGNOSTIC
};

static DiagnosticsLookup* _newJSONDiagnosticsLookup()
{
    auto lookup = new DiagnosticsLookup;
    lookup->add(kJSONDiagnostics, SLANG_COUNT_OF(kJSONDiagnostics));
    return lookup;
}

DiagnosticsLookup* getJSONDiagnosticsLookup()
{
    static RefPtr<DiagnosticsLookup> s_lookup = _newJSONDiagnosticsLookup();
    return s_lookup;
}

} // namespace Slang
