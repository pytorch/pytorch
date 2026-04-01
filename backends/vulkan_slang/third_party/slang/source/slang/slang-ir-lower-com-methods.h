// slang-ir-lower-com-methods.h
#pragma once

namespace Slang
{

struct IRModule;
class DiagnosticSink;

/// Lower the signature of COM interface methods out of types that
/// cannot appear in a COM interface. For example, String, List, ComPtr, Result all need to be
/// translated.
void lowerComMethods(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
