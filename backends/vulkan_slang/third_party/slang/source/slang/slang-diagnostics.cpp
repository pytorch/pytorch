// slang-diagnostics.cpp
#include "slang-diagnostics.h"

#include "../compiler-core/slang-core-diagnostics.h"
#include "../compiler-core/slang-name.h"
#include "../core/slang-char-util.h"
#include "../core/slang-dictionary.h"
#include "../core/slang-memory-arena.h"
#include "../core/slang-string-util.h"
namespace Slang
{

namespace Diagnostics
{
#define DIAGNOSTIC(id, severity, name, messageFormat) \
    const DiagnosticInfo name = {id, Severity::severity, #name, messageFormat};
#include "slang-diagnostic-defs.h"
#undef DIAGNOSTIC
} // namespace Diagnostics

static const DiagnosticInfo* const kCompilerDiagnostics[] = {
#define DIAGNOSTIC(id, severity, name, messageFormat) &Diagnostics::name,
#include "slang-diagnostic-defs.h"
#undef DIAGNOSTIC
};

static DiagnosticsLookup* _newDiagnosticsLookup()
{
    DiagnosticsLookup* lookup =
        new DiagnosticsLookup(kCompilerDiagnostics, SLANG_COUNT_OF(kCompilerDiagnostics));

    // Add all the diagnostics in 'core'
    DiagnosticsLookup* coreLookup = getCoreDiagnosticsLookup();
    if (coreLookup)
    {
        for (auto diagnostic : coreLookup->getDiagnostics())
        {
            lookup->add(diagnostic);
        }
    }

    // Add the alias
    lookup->addAlias("overlappingBindings", "parameterBindingsOverlap");
    return lookup;
}

static DiagnosticsLookup* _getDiagnosticLookupSingleton()
{
    static RefPtr<DiagnosticsLookup> s_lookup = _newDiagnosticsLookup();
    return s_lookup;
}

DiagnosticInfo const* findDiagnosticByName(UnownedStringSlice const& name)
{
    return _getDiagnosticLookupSingleton()->findDiagnosticByName(name);
}

const DiagnosticsLookup* getDiagnosticsLookup()
{
    return _getDiagnosticLookupSingleton();
}


SlangResult overrideDiagnostic(
    DiagnosticSink* sink,
    DiagnosticSink* outDiagnostic,
    const UnownedStringSlice& identifier,
    Severity originalSeverity,
    Severity overrideSeverity)
{
    auto diagnosticsLookup = getDiagnosticsLookup();

    const DiagnosticInfo* diagnostic = nullptr;
    Int diagnosticId = -1;

    // If it starts with a digit we assume it a number
    if (identifier.getLength() > 0 && (CharUtil::isDigit(identifier[0]) || identifier[0] == '-'))
    {
        if (SLANG_FAILED(StringUtil::parseInt(identifier, diagnosticId)))
        {
            outDiagnostic->diagnose(SourceLoc(), Diagnostics::unknownDiagnosticName, identifier);
            return SLANG_FAIL;
        }

        // If we use numbers, we don't worry if we can't find a diagnostic
        // and silently ignore. This was the previous behavior, and perhaps
        // provides a way to safely disable warnings, without worrying about
        // the version of the compiler.
        diagnostic = diagnosticsLookup->getDiagnosticById(diagnosticId);
    }
    else
    {
        diagnostic = diagnosticsLookup->findDiagnosticByName(identifier);
        if (!diagnostic)
        {
            outDiagnostic->diagnose(SourceLoc(), Diagnostics::unknownDiagnosticName, identifier);
            return SLANG_FAIL;
        }
        diagnosticId = diagnostic->id;
    }

    // If we are only allowing certain original severities check it's the right type
    if (diagnostic && originalSeverity != Severity::Disable &&
        diagnostic->severity != originalSeverity)
    {
        // Strictly speaking the diagnostic name is known, but it's not the right severity
        // to be converted from, so it is an 'unknown name' in the context of severity...
        // Or perhaps we want another diagnostic
        outDiagnostic->diagnose(SourceLoc(), Diagnostics::unknownDiagnosticName, identifier);
        return SLANG_FAIL;
    }

    // Override the diagnostic severity in the sink
    sink->overrideDiagnosticSeverity(int(diagnosticId), overrideSeverity, diagnostic);

    return SLANG_OK;
}

SlangResult overrideDiagnostics(
    DiagnosticSink* sink,
    DiagnosticSink* outDiagnostic,
    const UnownedStringSlice& identifierList,
    Severity originalSeverity,
    Severity overrideSeverity)
{
    List<UnownedStringSlice> slices;
    StringUtil::split(identifierList, ',', slices);

    for (const auto& slice : slices)
    {
        SLANG_RETURN_ON_FAIL(
            overrideDiagnostic(sink, outDiagnostic, slice, originalSeverity, overrideSeverity));
    }
    return SLANG_OK;
}

} // namespace Slang
