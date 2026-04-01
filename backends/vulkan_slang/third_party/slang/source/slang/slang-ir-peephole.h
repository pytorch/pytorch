// slang-ir-peephole.h
#pragma once

namespace Slang
{
struct IRModule;
struct IRCall;
struct IRInst;
class TargetProgram;

struct PeepholeOptimizationOptions
{
    bool isPrelinking = false;
    static PeepholeOptimizationOptions getPrelinking()
    {
        PeepholeOptimizationOptions result;
        result.isPrelinking = true;
        return result;
    }
};

/// Apply peephole optimizations.
bool peepholeOptimize(TargetProgram* target, IRModule* module, PeepholeOptimizationOptions options);
bool peepholeOptimize(TargetProgram* target, IRInst* func);
bool peepholeOptimizeInst(TargetProgram* target, IRModule* module, IRInst* inst);
bool peepholeOptimizeGlobalScope(TargetProgram* target, IRModule* module);
bool tryReplaceInstUsesWithSimplifiedValue(TargetProgram* target, IRModule* module, IRInst* inst);
} // namespace Slang
