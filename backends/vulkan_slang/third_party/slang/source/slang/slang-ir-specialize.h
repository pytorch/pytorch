// slang-ir-specialize.h
#pragma once

namespace Slang
{
struct IRModule;
class DiagnosticSink;
class TargetProgram;

struct SpecializationOptions
{
    // Option that allows specializeModule to generate dynamic-dispatch code
    // wherever possible to open up more specialization opportunities.
    //
    bool lowerWitnessLookups = false;
};

/// Specialize generic and interface-based code to use concrete types.
bool specializeModule(
    TargetProgram* target,
    IRModule* module,
    DiagnosticSink* sink,
    SpecializationOptions options);

void finalizeSpecialization(IRModule* module);

} // namespace Slang
