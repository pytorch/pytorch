// slang-ir-ssa-simplification.h
#pragma once

#include "slang-ir-dce.h"
#include "slang-ir-peephole.h"
#include "slang-ir-simplify-cfg.h"

namespace Slang
{
struct IRModule;
struct IRGlobalValueWithCode;
class DiagnosticSink;
class TargetProgram;

struct IRSimplificationOptions
{
    CFGSimplificationOptions cfgOptions;
    PeepholeOptimizationOptions peepholeOptions;
    IRDeadCodeEliminationOptions deadCodeElimOptions;

    bool minimalOptimization = false;
    bool removeRedundancy = false;
    bool hoistLoopInvariantInsts = false;

    static IRSimplificationOptions getDefault(TargetProgram* targetProgram);

    static IRSimplificationOptions getFast(TargetProgram* targetProgram);
};

// Run a combination of SSA, SCCP, SimplifyCFG, and DeadCodeElimination pass
// until no more changes are possible.
void simplifyIR(
    TargetProgram* target,
    IRModule* module,
    IRSimplificationOptions options,
    DiagnosticSink* sink = nullptr);

// Run simplifications on IR that is out of SSA form.
void simplifyNonSSAIR(TargetProgram* target, IRModule* module, IRSimplificationOptions options);

void simplifyFunc(
    TargetProgram* target,
    IRGlobalValueWithCode* func,
    IRSimplificationOptions options,
    DiagnosticSink* sink = nullptr);
} // namespace Slang
