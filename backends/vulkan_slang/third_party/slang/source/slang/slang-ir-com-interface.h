// slang-ir-com-interface.cpp
#pragma once

#include "../compiler-core/slang-artifact.h"

namespace Slang
{

struct IRModule;
class DiagnosticSink;

/// Lower com interface types.
/// A use of `IRInterfaceType` with `IRComInterfaceDecoration` will be translated into a `IRComPtr`
/// type. A use of `IRThisType` with a COM interface will also be translated into a `IRComPtr` type.
void lowerComInterfaces(IRModule* module, ArtifactStyle artifactStyle, DiagnosticSink* sink);

} // namespace Slang
