#include "slang-ir-reachability.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"

namespace Slang
{
// Computes whether block1 can reach block2.
// A block is considered not reachable from itself unless there is a backedge in the CFG.
ReachabilityContext::ReachabilityContext(IRGlobalValueWithCode* code)
{
    int id = 0;
    for (auto block : code->getBlocks())
    {
        mapBlockToId[block] = id++;
        allBlocks.add(block);
    }

    sourceBlocks.setCount(allBlocks.getCount());
    for (auto& srcBlock : sourceBlocks)
        srcBlock.resizeAndClear(allBlocks.getCount());

    if (allBlocks.getCount() == 0)
        return;

    List<IRBlock*> workList;
    List<IRBlock*> pendingWorkList;

    workList.add(allBlocks[0]);
    while (workList.getCount())
    {
        pendingWorkList.clear();
        for (Index i = 0; i < workList.getCount(); i++)
        {
            auto src = workList[i];
            auto srcId = mapBlockToId.getValue(src);
            for (auto successor : src->getSuccessors())
            {
                auto successorId = mapBlockToId.getValue(successor);
                auto& blockSet = sourceBlocks[successorId];
                bool changed = false;
                if (!blockSet.contains(srcId))
                {
                    blockSet.add(srcId);
                    changed = true;
                }
                if (!blockSet.contains(sourceBlocks[srcId]))
                {
                    blockSet.unionWith(sourceBlocks[srcId]);
                    changed = true;
                }
                if (changed)
                    pendingWorkList.add(successor);
            }
        }
        workList.swapWith(pendingWorkList);
    }
}

bool ReachabilityContext::isInstReachable(IRInst* from, IRInst* to)
{
    // If inst1 and inst2 are in the same block,
    // we test if inst2 appears after inst1.
    if (getBlock(from) == getBlock(to))
    {
        for (auto inst = from->getNextInst(); inst; inst = inst->getNextInst())
        {
            if (inst == to)
                return true;
        }
    }

    return isBlockReachable(getBlock(from), getBlock(to));
}

bool ReachabilityContext::isBlockReachable(IRBlock* from, IRBlock* to)
{
    if (!from)
        return false;

    if (!to)
        return false;

    int* fromId = mapBlockToId.tryGetValue(from);
    int* toId = mapBlockToId.tryGetValue(to);
    if (!fromId || !toId)
        return true;

    return sourceBlocks[*toId].contains(*fromId);
}
} // namespace Slang
