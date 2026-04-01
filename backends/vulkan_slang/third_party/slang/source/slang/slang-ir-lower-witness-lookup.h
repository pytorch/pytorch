// slang-ir-lower-witness-lookup.h
#pragma once

namespace Slang
{
struct IRModule;
class DiagnosticSink;

/// Lower calls to a witness lookup into a call to a dispatch function.
/// For example, if we see call(witnessLookup(wt, key)), we will create a
/// dispatch function that calls into different implementations based on witness table
/// ID. The dispatch function will be called instead of witnessLookup.
bool lowerWitnessLookup(IRModule* module, DiagnosticSink* sink);

} // namespace Slang
