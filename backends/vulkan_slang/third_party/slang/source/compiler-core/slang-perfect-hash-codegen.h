#pragma once

#include "slang-diagnostic-sink.h"
#include "slang-perfect-hash.h"

namespace Slang
{
SlangResult writePerfectHashLookupCppFile(
    String fileName,
    List<String> opnames,
    String enumName,
    String enumPrefix,
    String enumHeaderFile,
    DiagnosticSink* sink);
}
