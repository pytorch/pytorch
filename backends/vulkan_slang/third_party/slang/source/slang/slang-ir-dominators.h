// slang-ir-dominators.h
#pragma once

#include "../core/slang-basic.h"

namespace Slang
{
struct IRBlock;
struct IRGlobalValueWithCode;
struct IRInst;

/// The computed dominator tree for an IR control flow graph.
struct IRDominatorTree : public RefObject
{
    /// The function or other code-bearing value for which the dominator tree was computed.
    IRGlobalValueWithCode* code;

    /// Does the first block dominate the second?
    ///
    /// A block A dominates block B iff every control-flow path
    /// that starts at the entry block of the CFG and passes
    /// through B must first pass through A.
    ///
    bool dominates(IRBlock* dominator, IRBlock* dominated);

    bool dominates(IRInst* dominator, IRInst* dominated);

    /// Does the first block properly dominate the second?
    ///
    /// Block A properly dominates block B iff A dominates B
    /// and A != B.
    ///
    bool properlyDominates(IRBlock* dominator, IRBlock* dominated);

    /// Does the first block immediately dominate the second?
    ///
    /// Block A immediately dominates block B iff A dominates B
    /// and for any block X that dominates B, X also dominates A.
    ///
    bool immediatelyDominates(IRBlock* dominator, IRBlock* dominated);

    /// Get the immediate dominator (idom) of a block.
    ///
    /// This is the parent of `block` in the dominator tree.
    IRBlock* getImmediateDominator(IRBlock* block);

    /// An iterable collection of the blocks dominated by a specific block
    struct DominatedList;

    /// Get the blocks that a block immediately dominates.
    ///
    /// These are the children of the block in the dominator tree.
    DominatedList getImmediatelyDominatedBlocks(IRBlock* block);

    /// Get the blocks that a block properly dominates.
    ///
    /// These are the descendents of the block in the dominator tree.
    DominatedList getProperlyDominatedBlocks(IRBlock* block);

    /// Is `block` unrechable in the control flow graph?
    bool isUnreachable(IRBlock* block);

    struct DominatedList
    {
    public:
        DominatedList();

        struct Iterator
        {
        public:
            Iterator();

            IRBlock* operator*() const;
            void operator++();
            bool operator==(Iterator const& that) const;
            bool operator!=(Iterator const& that) const;

        private:
            friend struct DominatedList;
            Iterator(IRDominatorTree* tree, Int index);

            IRDominatorTree* mTree;
            Int mIndex;
        };

        Iterator begin() const;
        Iterator end() const;

        Count getCount() const { return Count(mEnd - mBegin); }

    private:
        friend struct IRDominatorTree;
        DominatedList(IRDominatorTree* tree, Int begin, Int end);

        IRDominatorTree* mTree;
        Int mBegin;
        Int mEnd;
    };

private:
    //
    // The layout of an `IRDominatorTree` uses a dense array for all of the nodes in the CFG.
    // We therefore need a way to map an `IRBlock*` pointer over to an index in this array:
    //

    /// Map a block to its index in the `nodes` array
    Int getBlockIndex(IRBlock* block);

    /// Dictionary used to accelerate `getBlockIndex`
    Dictionary<IRBlock*, Int> mapBlockToIndex;

    /// Reachability information for the CFG
    HashSet<IRBlock*> reachableSet;

    //
    // In order to accelerate queries on the tree structure, we will order the tree nodes
    // carefully, so that all of the descendants of a node are contiguous, with all of
    // the immediate children coming first.
    //
    // Each node thus needs to remember its parent (immediate dominator), and the range
    // of indices that represent children and descendents (respectively), with the knowledge
    // that the first child and first descendent share the same index.
    //

    /// Information about one node in the dominator tree
    struct Node
    {
        /// The block associated with this tree node
        IRBlock* block;

        /// Index of the parent node or -1 if no parent
        Int parent;

        /// Index of first descendent
        Int beginDescendents;

        /// "One after the end" value for range of child node indices.
        Int endChildren;

        /// "One after the end" value for range of descendent node indices.
        Int endDescendents;
    };

    /// Storage for the dominator tree itself
    List<Node> nodes;

    /// Value to use for invalid node indices (e.g.,
    /// when a node has no parent).
    static const Int kInvalidIndex = -1;

    //
    // The `DominatedList` type needs direct access to all of this
    // data in order to provide iteration.
    //
    friend struct DominatedList;
    friend struct DominatedList::Iterator;
    //
    // The context type we will use to compute the dominator tree
    // also needs to be able to access all the fields to initialze
    // an `IRDominatorTree`
    //
    friend struct DominatorTreeComputationContext;

    // TODO: we should probably build/store a postdominator
    // tree in the same structure, just to make life simpler.
};

RefPtr<IRDominatorTree> computeDominatorTree(IRGlobalValueWithCode* code);

void computePostorder(IRGlobalValueWithCode* code, List<IRBlock*>& outOrder);
void computePostorder(
    IRGlobalValueWithCode* code,
    List<IRBlock*>& outOrder,
    HashSet<IRBlock*>& outReachableSet);
void computePostorderOnReverseCFG(IRGlobalValueWithCode* code, List<IRBlock*>& outOrder);

inline List<IRBlock*> getPostorder(IRGlobalValueWithCode* code)
{
    List<IRBlock*> result;
    computePostorder(code, result);
    return result;
}

inline List<IRBlock*> getPostorderOnReverseCFG(IRGlobalValueWithCode* code)
{
    List<IRBlock*> result;
    computePostorderOnReverseCFG(code, result);
    return result;
}

inline List<IRBlock*> getReversePostorder(IRGlobalValueWithCode* code)
{
    List<IRBlock*> result;
    computePostorder(code, result);
    result.reverse();
    return result;
}

inline List<IRBlock*> getReversePostorderOnReverseCFG(IRGlobalValueWithCode* code)
{
    List<IRBlock*> result;
    computePostorderOnReverseCFG(code, result);
    result.reverse();
    return result;
}
} // namespace Slang
