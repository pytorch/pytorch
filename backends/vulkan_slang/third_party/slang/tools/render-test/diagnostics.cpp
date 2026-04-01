// diagnostics.cpp
#include "diagnostics.h"

namespace Slang
{

namespace RenderTestDiagnostics
{
#define DIAGNOSTIC(id, severity, name, messageFormat) \
    const DiagnosticInfo name = {id, Severity::severity, #name, messageFormat};
#include "diagnostic-defs.h"
#undef DIAGNOSTIC
} // namespace RenderTestDiagnostics

static const DiagnosticInfo* const kDiagnostics[] = {
#define DIAGNOSTIC(id, severity, name, messageFormat) &RenderTestDiagnostics::name,
#include "diagnostic-defs.h"
#undef DIAGNOSTIC
};


} // namespace Slang
