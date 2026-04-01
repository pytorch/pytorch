// slang-ir-autodiff-region.h
#pragma once

#include "slang-ir-autodiff.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
struct IndexedRegion : public RefObject
{
    IRLoop* loop;
    IndexedRegion* parent;

    IndexedRegion(IRLoop* loop, IndexedRegion* parent)
        : loop(loop), parent(parent)
    {
    }

    IRBlock* getInitializerBlock() { return as<IRBlock>(loop->getParent()); }
    IRBlock* getConditionBlock()
    {
        auto condBlock = as<IRBlock>(loop->getTargetBlock());
        SLANG_RELEASE_ASSERT(as<IRIfElse>(condBlock->getTerminator()));
        return condBlock;
    }

    IRBlock* getBreakBlock() { return loop->getBreakBlock(); }

    IRBlock* getUpdateBlock()
    {
        auto initBlock = getInitializerBlock();

        auto condBlock = getConditionBlock();

        IRBlock* lastLoopBlock = nullptr;

        for (auto predecessor : condBlock->getPredecessors())
        {
            if (predecessor != initBlock)
                lastLoopBlock = predecessor;
        }

        // Should find atleast one predecessor that is _not_ the
        // init block (that contains the loop info). This
        // predecessor would be the last block in the loop
        // before looping back to the condition.
        //
        SLANG_RELEASE_ASSERT(lastLoopBlock);

        return lastLoopBlock;
    }
};

struct IndexedRegionMap : public RefObject
{
    Dictionary<IRBlock*, IndexedRegion*> map;
    List<RefPtr<IndexedRegion>> regions;

    IndexedRegion* newRegion(IRLoop* loop, IndexedRegion* parent)
    {
        auto region = new IndexedRegion(loop, parent);
        regions.add(region);

        return region;
    }

    void mapBlock(IRBlock* block, IndexedRegion* region) { map.add(block, region); }

    bool hasMapping(IRBlock* block) { return map.containsKey(block); }

    IndexedRegion* getRegion(IRBlock* block) { return map[block]; }

    List<IndexedRegion*> getAllAncestorRegions(IRBlock* block)
    {
        List<IndexedRegion*> regionList;

        IndexedRegion* region = getRegion(block);
        for (; region; region = region->parent)
            regionList.add(region);

        return regionList;
    }
};

RefPtr<IndexedRegionMap> buildIndexedRegionMap(IRGlobalValueWithCode* func);


}; // namespace Slang
