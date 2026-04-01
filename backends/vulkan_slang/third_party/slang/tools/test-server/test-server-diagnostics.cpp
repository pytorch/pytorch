#include "test-server-diagnostics.h"

namespace TestServer
{

namespace ServerDiagnostics
{
using namespace Slang;

#define DIAGNOSTIC(id, severity, name, messageFormat) \
    const DiagnosticInfo name = {id, Severity::severity, #name, messageFormat};
#include "test-server-diagnostic-defs.h"
} // namespace ServerDiagnostics

} // namespace TestServer
