// slang-ir-reachability.h
#pragma once

#include "slang-ir.h"

namespace Slang
{

// A context for computing and caching reachability between blocks on the CFG.
struct ReachabilityContext
{
    Dictionary<IRBlock*, int> mapBlockToId;
    List<IRBlock*> allBlocks;
    List<UIntSet> sourceBlocks; // sourcesBlocks[i] stores the set of blocks from which block i can
                                // be reached.

    ReachabilityContext() = default;
    ReachabilityContext(IRGlobalValueWithCode* code);

    bool isInstReachable(IRInst* from, IRInst* to);
    bool isBlockReachable(IRBlock* from, IRBlock* to);
};

} // namespace Slang
