// slang-fiddle-diagnostics.cpp
#include "slang-fiddle-diagnostics.h"

namespace fiddle
{
namespace Diagnostics
{
using namespace Slang;

#define DIAGNOSTIC(id, severity, name, messageFormat) \
    const DiagnosticInfo name = {id, Severity::severity, #name, messageFormat};
#include "slang-fiddle-diagnostic-defs.h"
} // namespace Diagnostics
} // namespace fiddle
