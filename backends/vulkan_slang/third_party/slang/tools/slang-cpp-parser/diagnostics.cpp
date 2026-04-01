#include "diagnostics.h"

namespace CppParse
{

namespace CPPDiagnostics
{
using namespace Slang;

#define DIAGNOSTIC(id, severity, name, messageFormat) \
    const DiagnosticInfo name = {id, Severity::severity, #name, messageFormat};
#include "diagnostic-defs.h"
} // namespace CPPDiagnostics

} // namespace CppParse
