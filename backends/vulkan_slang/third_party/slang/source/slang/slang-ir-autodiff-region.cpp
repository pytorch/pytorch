// slang-ir-autodiff-region.cpp
#include "slang-ir-autodiff-region.h"

namespace Slang
{
RefPtr<IndexedRegionMap> buildIndexedRegionMap(IRGlobalValueWithCode* func)
{
    RefPtr<IndexedRegionMap> regionMap = new IndexedRegionMap;

    List<IRBlock*> workList;

    regionMap->mapBlock(func->getFirstBlock(), nullptr);
    workList.add(func->getFirstBlock());

    while (workList.getCount() > 0)
    {
        auto currentBlock = workList.getLast();
        workList.removeLast();

        auto terminator = currentBlock->getTerminator();
        auto currentRegion = regionMap->getRegion(currentBlock);

        switch (terminator->getOp())
        {
        case kIROp_loop:
            {
                auto loopRegion = regionMap->newRegion(as<IRLoop>(terminator), currentRegion);
                auto condBlock = as<IRLoop>(terminator)->getTargetBlock();

                regionMap->mapBlock(condBlock, loopRegion);
                workList.add(condBlock);

                auto ifElse = as<IRIfElse>(condBlock->getTerminator());
                SLANG_RELEASE_ASSERT(ifElse);

                // TODO: this is one of the places we'll need to change if we support loops that
                // loop on either the true or false side. For now, we assume the loop is on the
                // true side only.
                //
                regionMap->mapBlock(ifElse->getFalseBlock(), currentRegion);
                workList.add(ifElse->getFalseBlock());
            }
        }

        for (auto successor : currentBlock->getSuccessors())
        {
            // If already mapped, skip.
            if (regionMap->hasMapping(successor))
                continue;
            regionMap->mapBlock(successor, currentRegion);
            workList.add(successor);
        }
    }

    return regionMap;
}
}; // namespace Slang
