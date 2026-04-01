// slang-ir-simplify-cfg.h
#pragma once

namespace Slang
{
struct IRModule;
struct IRGlobalValueWithCode;
struct IRLoop;
struct IRDominatorTree;

struct CFGSimplificationOptions
{
    bool removeTrivialSingleIterationLoops = true;
    bool removeSideEffectFreeLoops = true;
    static CFGSimplificationOptions getDefault() { return CFGSimplificationOptions(); }
    static CFGSimplificationOptions getFast() { return CFGSimplificationOptions{false, false}; }
};

bool isTrivialSingleIterationLoop(
    IRDominatorTree* domTree,
    IRGlobalValueWithCode* func,
    IRLoop* loop);

/// Simplifies control flow graph by merging basic blocks that
/// forms a simple linear chain.
/// Returns true if changed.
bool simplifyCFG(IRModule* module, CFGSimplificationOptions options);

bool simplifyCFG(IRGlobalValueWithCode* func, CFGSimplificationOptions options);

} // namespace Slang
