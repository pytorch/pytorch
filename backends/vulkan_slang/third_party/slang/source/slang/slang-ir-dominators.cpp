// slang-ir-dominators.cpp
#include "slang-ir-dominators.h"

//
// This file implements the public interface of the `IRDominatorTree` type,
// to enable queries on dominance relationships in a control-flow graph.
//
// It also implements computation of the dominator tree for a CFG using
// the algorithm presented in "A Simple, Fast Dominance Algorithm" by
// Keith D. Cooper, Timothy J. Harvey, and Ken Kennedy.
//
// The algorithm is *not* the most efficinet one, asymptotically, but
// it is one that is easy to implement and explain, and so we favor it
// in order to get something up and running with a reasonable level of
// confidence that the results are correct.
//

#include "slang-ir.h"

namespace Slang
{

//
// Let's start with the implementation of the public API for `IRDominatorTree`
//

// IRDominatorTree

bool IRDominatorTree::immediatelyDominates(IRBlock* dominator, IRBlock* dominated)
{
    // To test if block A immediately dominates block B, we just
    // check if A is the (one and only) immediate dominator of B.
    return dominator == getImmediateDominator(dominated);
}

bool IRDominatorTree::properlyDominates(IRBlock* dominator, IRBlock* dominated)
{
    // We need to deal with the cases where `dominator` and/or
    // `dominated` are unreachable, and thus not represtend
    // in the nodes of the dominator tree we constructed.
    //
    // If `dominated` is unreachable, then there are zero
    // control flow paths that can reach it, so that *all*
    // of those (zero) control flow paths go through
    // `dominator`.
    //
    if (isUnreachable(dominated))
        return true;

    // If `dominated` is reachable then there must exist at least
    // one control-flow path to it. Thus if `dominator` is not
    // reachable, it cannot be on that path, and thus must
    // not be a dominator.
    //
    if (isUnreachable(dominator))
        return false;


    // Because of how we laid out the tree, we can test if one node
    // properly dominates another in constant time.
    //
    // We simply need to test if the node index for `dominated` falls
    // in the range of indices for the descendents of `dominator`.
    //

    Int dominatorIndex = getBlockIndex(dominator);
    Int dominatedIndex = getBlockIndex(dominated);
    Node& dominatorNode = nodes[dominatorIndex];

    return (dominatedIndex >= dominatorNode.beginDescendents) &&
           (dominatedIndex < dominatorNode.endDescendents);
}

bool IRDominatorTree::dominates(IRBlock* dominator, IRBlock* dominated)
{
    // We need to check two cases here.
    //
    // First, a node always dominated itself, so if the blocks are
    // the the same, then we are done:
    //
    if (dominator == dominated)
        return true;
    //
    // Otherwise, for distinct blocks we just check for
    // proper dominance:
    //
    return properlyDominates(dominator, dominated);
}

bool IRDominatorTree::dominates(IRInst* dominator, IRInst* dominated)
{
    auto dominatorBlock = as<IRBlock>(dominator);
    if (!dominatorBlock)
        dominatorBlock = as<IRBlock>(dominator->getParent());

    auto dominatedBlock = as<IRBlock>(dominated);
    if (!dominatedBlock)
        dominatedBlock = as<IRBlock>(dominated->getParent());

    if (dominatorBlock == dominatedBlock)
    {
        for (auto inst = dominator; inst; inst = inst->getNextInst())
        {
            if (inst == dominated)
                return true;
        }
        return false;
    }
    else
    {
        return dominates(dominatorBlock, dominatedBlock);
    }
}

IRBlock* IRDominatorTree::getImmediateDominator(IRBlock* block)
{
    // An unreachable block has no immediate dominator.
    //
    if (isUnreachable(block))
        return nullptr;

    // The immediate dominator of a block is its parent
    // in the dominator tree. Looking this up is straightforward,
    // and we just need to be a bit careful to deal with
    // invalid node indices.

    Int blockIndex = getBlockIndex(block);
    if (blockIndex == kInvalidIndex)
        return nullptr;

    Int parentIndex = nodes[blockIndex].parent;
    if (parentIndex == kInvalidIndex)
        return nullptr;

    return nodes[parentIndex].block;
}

IRDominatorTree::DominatedList IRDominatorTree::getImmediatelyDominatedBlocks(IRBlock* block)
{
    // An unreachable block doesn't immediately dominate anything.
    //
    if (isUnreachable(block))
        return DominatedList();

    // Because of our representation, the immediately dominated blocks
    // for a node are contiguous, and we store their range in the
    // node already.

    Int blockIndex = getBlockIndex(block);
    if (blockIndex == kInvalidIndex)
        return DominatedList();

    Node& node = nodes[blockIndex];
    return DominatedList(this, node.beginDescendents, node.endChildren);
}

IRDominatorTree::DominatedList IRDominatorTree::getProperlyDominatedBlocks(IRBlock* block)
{
    // Technically each unreachable block dominates all the other
    // unreachable blocks, but setting things up to answer that
    // query "correctly" would be a hassle.
    //
    if (isUnreachable(block))
        return DominatedList();

    // Because of our representation, the properly dominated blocks
    // for a node are contiguous, and we store their range in the
    // node already.

    Int blockIndex = getBlockIndex(block);
    if (blockIndex == kInvalidIndex)
        return DominatedList();

    Node& node = nodes[blockIndex];
    return DominatedList(this, node.beginDescendents, node.endDescendents);
}

Int IRDominatorTree::getBlockIndex(IRBlock* block)
{
    Int index = kInvalidIndex;
    if (!mapBlockToIndex.tryGetValue(block, index))
    {
        SLANG_UNEXPECTED("block was not present in dominator tree");
    }
    return index;
}

bool IRDominatorTree::isUnreachable(IRBlock* block)
{
    return !reachableSet.contains(block);
}


// IRDominatorTree::DominatedList

IRDominatorTree::DominatedList::DominatedList()
    : mTree(nullptr), mBegin(0), mEnd(0)
{
}

IRDominatorTree::DominatedList::DominatedList(IRDominatorTree* tree, Int begin, Int end)
    : mTree(tree), mBegin(begin), mEnd(end)
{
}

IRDominatorTree::DominatedList::Iterator IRDominatorTree::DominatedList::begin() const
{
    return Iterator(mTree, mBegin);
}

IRDominatorTree::DominatedList::Iterator IRDominatorTree::DominatedList::end() const
{
    return Iterator(mTree, mEnd);
}


// IRDominatorTree::DominatedList::Iterator

IRDominatorTree::DominatedList::Iterator::Iterator()
    : mTree(nullptr), mIndex(0)
{
}

IRDominatorTree::DominatedList::Iterator::Iterator(IRDominatorTree* tree, Int index)
    : mTree(tree), mIndex(index)
{
}

IRBlock* IRDominatorTree::DominatedList::Iterator::operator*() const
{
    return mTree->nodes[mIndex].block;
}

void IRDominatorTree::DominatedList::Iterator::operator++()
{
    mIndex++;
}

bool IRDominatorTree::DominatedList::Iterator::operator==(Iterator const& that) const
{
    SLANG_ASSERT(mTree == that.mTree);
    return mIndex == that.mIndex;
}

bool IRDominatorTree::DominatedList::Iterator::operator!=(Iterator const& that) const
{
    SLANG_ASSERT(mTree == that.mTree);
    return mIndex != that.mIndex;
}

//
// The dominance computation algorithm we are using relies on being able to compute
// a reverse postorder traversal of the nodes in the CFG, which is done using a depth-first
// search (DFS). We don't currently have infrastructure for DFS in the compiler, so
// we will implement it here for now, and plan to move it into its own file once
// we have a second use case.
//

/// A base "visitor" class for use in depth-first search algorithms on an IR CFG.
struct DepthFirstSearchContext
{
    /// The blocks in the CFG that we've already visited.
    HashSet<IRBlock*> visited;

    /// Walk a (previously unvisited) block.
    ///
    /// This will perform any pre-order actions on the block,
    /// then recursively visit its (unvisited) successors, and
    /// then perform any post-actions.
    ///
    template<typename SuccessorFunc>
    void walk(IRBlock* block, const SuccessorFunc& getSuccessors)
    {
        List<IRBlock*> nodeStack;
        nodeStack.add(block);
        visited.add(block);
        preVisit(block);

        while (nodeStack.getCount())
        {
            auto curNode = nodeStack.getLast();
            bool pushedChild = false;
            for (auto succ : getSuccessors(curNode))
            {
                if (!visited.contains(succ))
                {
                    pushedChild = true;
                    nodeStack.add(succ);
                    visited.add(succ);

                    preVisit(succ);
                    break;
                }
            }
            if (!pushedChild)
            {
                postVisit(curNode);
                nodeStack.removeLast();
            }
        }
    }

    /// Overridable action to perform on first entering a CFG node.
    virtual void preVisit(IRBlock* /*block*/) {}

    /// Overridable action to perform on exiting a CFG node
    virtual void postVisit(IRBlock* /*block*/) {}
};

//
// With DFS traversal factored out, computing a post-order walk
// of the CFG is a simple matter of defining a visitor that appends
// to an order as a post-action:
//

/// A visitor that computes a postorder traversal for a CFG.
struct PostorderComputationContext : public DepthFirstSearchContext
{
    /// List to append the computed order onto
    List<IRBlock*>* order;

    virtual void postVisit(IRBlock* block) SLANG_OVERRIDE { order->add(block); }
};

void computeReachableSet(IRGlobalValueWithCode* code, HashSet<IRBlock*>& outSet)
{
    DepthFirstSearchContext context;
    if (code->getFirstBlock())
        context.walk(code->getFirstBlock(), [](IRBlock* block) { return block->getSuccessors(); });
    outSet = _Move(context.visited);
}

/// Compute a postorder traversal of the blocks in `code`, writing the resulting order to
/// `outOrder`.
void computePostorder(IRGlobalValueWithCode* code, List<IRBlock*>& outOrder)
{
    HashSet<IRBlock*> reachableSet;
    computePostorder(code, outOrder, reachableSet);
}

/// Compute a postorder traversal of the blocks in `code`, writing the resulting order to
/// `outOrder`.
void computePostorder(
    IRGlobalValueWithCode* code,
    List<IRBlock*>& outOrder,
    HashSet<IRBlock*>& outReachableSet)
{
    PostorderComputationContext context;
    context.order = &outOrder;
    if (code->getFirstBlock())
        context.walk(code->getFirstBlock(), [](IRBlock* block) { return block->getSuccessors(); });

    // Append unvisited blocks (unreachable blocks) to the begining of postOrder.
    List<IRBlock*> prefix;
    for (auto block : code->getBlocks())
    {
        if (!context.visited.contains(block))
        {
            prefix.add(block);
        }
    }
    prefix.addRange(outOrder);
    outOrder = _Move(prefix);
    outReachableSet = _Move(context.visited);
}

void computePostorderOnReverseCFG(IRGlobalValueWithCode* code, List<IRBlock*>& outOrder)
{
    PostorderComputationContext context;
    context.order = &outOrder;
    for (auto block = code->getLastBlock(); block; block = block->getPrevBlock())
    {
        auto terminator = block->getTerminator();
        switch (terminator->getOp())
        {
        case kIROp_Return:
        case kIROp_MissingReturn:
        case kIROp_Unreachable:
            context.walk(block, [](IRBlock* b) { return b->getPredecessors(); });
            break;
        }
    }
    return;
}

//
// With the preliminaries out of the way, we are ready to implement
// the dominator tree construction algorithm as described by Cooper, Harvey, and Kennedy.
// The actual code for the algorithm is given in Figure 3 of the paper.
//
// We will wrap the subroutines of their algorithm in a `struct` type
// to allow the temporary structures to be shared.
//
struct DominatorTreeComputationContext
{
    // We will use signed integers to represent the "name" of a block.
    // The integers will reflect the a postorder traversal, and this
    // property will be exploited in the `intersect()` function.
    //
    typedef Int BlockName;
    //
    // An invalid/undefined block name will be represented as -1.
    //
    static const BlockName kUndefined = BlockName(-1);
    //
    // We will explicitly store the blocks visited in the postorder
    // traversal, so that we can look up a block based on its "name"
    //
    List<IRBlock*> postorder;
    //
    // Also maintain a set of reachable blocks.
    //
    HashSet<IRBlock*> reachableSet;

    //
    // We need a way to map our actual IR blocks to their names for
    // the purpose of this algorithm. This mapping step adds overhead,
    // but it seems unavoidable unless we also translate the CFG itself
    // to an index-based representation.
    //
    Dictionary<IRBlock*, BlockName> mapBlockToName;
    BlockName getBlockName(IRBlock* block) { return mapBlockToName[block]; }

    //
    // The algorithm iteratively builds up an array `doms` that upon
    // completion will directly encode the immediate dominator for each
    // node. During the iterative steps it is used to implicitly encode
    // a representation of the set of dominators for each node.
    //
    List<BlockName> doms;


    //
    // Here we get to the meat of the algorithm presented in Cooper et al.
    // Figure 3:
    //
    void iterativelyComputeImmediateDominators(IRGlobalValueWithCode* code)
    {
        // First we compute the postorder traversal order for the blocks in the CFG.
        computePostorder(code, postorder, reachableSet);

        // We will initialize our map from the block objects to their "name"
        // (index in the traversal order), before moving on.
        BlockName blockCount = BlockName(postorder.getCount());
        for (BlockName bb = 0; bb < blockCount; ++bb)
        {
            mapBlockToName[postorder[bb]] = bb;
        }

        // Next we initialize the `doms` array that we will iteratively turn
        // into an encoding of the dominator tree.
        doms.setCount(blockCount);
        for (BlockName bb = 0; bb < blockCount; ++bb)
        {
            doms[bb] = kUndefined;
        }

        // The start node is special, since it is the root of the dominator tree.
        // Technically it doesn't have an immediate dominator, but we will set
        // its entry in `doms` to refer to itself, to indicate that we are done
        // processing the given node.
        //
        BlockName startNode = getBlockName(code->getFirstBlock());
        doms[startNode] = startNode;

        // Given that we computed a postorder traversal of the graph, we know
        // that the start node should be the last one in the computed order.
        //
        SLANG_ASSERT(startNode == blockCount - 1);

        // We are using an iterative algorithm, so we will detect that we
        // have reached a fixed point when we hit an iteration where nothing
        // changes.
        //
        bool changed = true;
        while (changed)
        {
            changed = false;

            // The algorithm specifies that we should walk through the blocks
            // in *reverse* postorder, since this speeds up convergence.
            // Because we've numbered the blocks in postorder, walking them
            // in reverse numerical order will do the trick.
            //
            // We don't want to include the start node in our iteration
            // (since we already know its dominators), and because we know
            // that the start node is always the last in the order (`blockCount - 1`)
            // we can just start at the next node after it (`blockCount - 2`).
            //
            // Note: it is important that we are using signed integers for
            // block numbers here, since we will drop below zero before exiting
            // the loop, and if the CFG had only a single block, then our *starting*
            // block index would be `-1`.
            //
            for (auto b = blockCount - 2; b >= 0; --b)
            {
                // We are walking through block indices, but the predecessor
                // lists are encoded in the IR blocks themselves.
                //
                IRBlock* block = postorder[b];

                // The algorithm description in the paper says to pick the
                // initial value for the `new_idom` variable from the "first
                // (processed) predecessor of b (pick one)".
                // After that step, the algorithm walks over the remaining
                // predecessors, and for the ones that have a valid entry
                // in the `doms` array, performs an intersection of their
                // implicitly-represented dominator sets.
                //
                // The paper doesn't precisely clarify what they mean by
                // a "processed" predecessor, but it seems to mean one that
                // has a valid value in the `doms` array, which is what
                // the subsequent loop is already checking.
                //
                // We are going to fold this logic together into a single loop.
                // We will start with an invalid/undefined value for
                // `new_idom`, which represents our best guess at the
                // immediate dominator for block `b`:
                //
                BlockName new_idom = kUndefined;

                // Now we will loop over *all* of the predecessors, ...
                for (auto pred : block->getPredecessors())
                {
                    // ... and skip those that haven't been "processed".
                    BlockName p = getBlockName(pred);
                    BlockName dominatorOfPredecessor = doms[p];
                    if (dominatorOfPredecessor == kUndefined)
                        continue;

                    // When we encounter the first "processed" predecessor,
                    // we can initialize the variable tracking our best
                    // guess at the immediate dominator.
                    //
                    if (new_idom == kUndefined)
                    {
                        new_idom = p;
                    }
                    //
                    // Otherwise, we need to merge information between
                    // the predecessor `p` and our best-guess immediate
                    // dominator `new_idom`. We need a node that dominates
                    // both of them to be the immediate dominator of `b`.
                    //
                    else
                    {
                        new_idom = intersect(p, new_idom);
                    }
                }

                // After we've computed a new best guess at the immediate
                // dominator for `b`, we need to see if the computed
                // value differs from what we'd previously stored in the
                // `doms` array. If anything changed, then we haven't
                // converged yet, and we need to keep going.
                //
                BlockName oldDominator = doms[b];
                if (oldDominator != new_idom)
                {
                    doms[b] = new_idom;
                    changed = true;
                }
            }
        }

        // Upon exiting the loop, things should have converged with
        // the `doms` array being an explicit encoding of the immediate
        // dominator for each node, with one small error: there is no
        // immediate dominator for the start node:
        doms[startNode] = kUndefined;
    }

    //
    // The algorithm above relied on a utility routine `intersect()` that
    // is implicitly used to compute intersections between sets of nodes,
    // but explicitly takes the form of a routine that computes a common
    // parent in the dominator tree for two nodes.
    //
    // We present that subroutine here, almost identical to how it
    // is presented in Cooper et al. Figure 3:
    //
    BlockName intersect(BlockName b1, BlockName b2)
    {
        // We need to find a common ancestor of both `b1` and `b2`,
        // and will do this by tracking two "fingers," each initially
        // pointing at one node, and then iteratively move the finger
        // that is furthest to the "left" (earlier in the postorder
        // traversal to the left until) to the "right" (by moving
        // the immediate dominator of the node we are pointing at),
        // until the two fingers are pointing at the same place.
        //
        // Termination is guaranteed because we are always moving the
        // fingers from a node to its immediate dominator, and the
        // entry node is guaranteed to be at the root of the dominator
        // tree.
        //
        // The use of the postorder here relies on the (subtle) fact
        // that the immediate dominator of a node must come later
        // in a postorder traversal.
        //
        BlockName finger1 = b1;
        BlockName finger2 = b2;

        while (finger1 != finger2)
        {
            while (finger1 < finger2)
                finger1 = doms[finger1];
            while (finger2 < finger1)
                finger2 = doms[finger2];
        }
        return finger1;
    }

    //
    // Now that we've implemented Cooper et al. fairly close to how
    // it was presented, we can build an array encoding the immediate
    // dominator relationship. We still need to expand that array
    // into an encoding that lets us efficiently answer queries
    // about dominance.
    //
    // In order to do that, we need to expand the information we
    // have built on each block (currently just an immediate dominator)
    // into a bit more detail:
    //
    struct BlockInfo
    {
        // How many children does this node/block have in the dominator tree?
        Int childCount = 0;

        // How many indirect (non-child) descendents?
        Int indirectDescendentCount = 0;

        // What is the 0-based offset of this node among all the children of its parent?
        Int childOffsetInParent = 0;

        // What is the 0-based offset for this node's descendent list,
        // among all the children in its parent?
        Int descendentOffsetInParent = 0;

        Int nodeIndex = 0;
        Int firstDescendentIndex = 0;
    };
    //

    RefPtr<IRDominatorTree> createDominatorTree(IRGlobalValueWithCode* code)
    {
        if (code->getFirstBlock() == nullptr)
            return nullptr;

        // We first run the Cooper et al. algorithm to compute the `doms` array
        // which encodes immediate dominators.
        //
        iterativelyComputeImmediateDominators(code);

        // We will build some intermediate information on each
        // block to help us fill out the tree.
        BlockName blockCount = BlockName(doms.getCount());
        List<BlockInfo> blockInfos;
        for (BlockName bb = 0; bb < blockCount; ++bb)
        {
            blockInfos.add(BlockInfo());
        }

        // We will propagate layout information in two passes over the tree.
        //
        // First we will perform a "bottom up" pass that will accumulate
        // the number of children and the total number of descendents for
        // each node, and also assign each child its relative offsets within
        // the storage for its parent.
        //
        // Because our blocks are ordered in postorder, we can do this
        // bottom-up walk just by iterating over them in the given order.
        //
        for (BlockName bb = 0; bb < blockCount; ++bb)
        {
            BlockName parent = doms[bb];
            if (parent == kUndefined)
                continue;

            // For our iteration order to make sense, we need to be certain
            // that parent nodes come after their child nodes in the postorder traversal.
            SLANG_ASSERT(parent > bb);

            // Compute the 0-based index of this child among all the children
            // with the same parent, and increment its child count.
            blockInfos[bb].childOffsetInParent = blockInfos[parent].childCount;
            blockInfos[parent].childCount++;

            // Our layout for the descendents of a node will put all the immediate
            // child nodes contiguously first, followed by their descendents (in contiguous blocks).
            //
            // We need to compute an offset for where the descendents of this node will
            // be stored, within the overall space carved out for the "indirect" descendents
            // of the parent node.
            //
            blockInfos[bb].descendentOffsetInParent = blockInfos[parent].indirectDescendentCount;
            //
            // When adding up the indirect descendents of `parent`, we need to include both
            // the direct and indirect descendents of our node `bb`.
            blockInfos[parent].indirectDescendentCount +=
                blockInfos[bb].childCount + blockInfos[bb].indirectDescendentCount;
        }
        //
        // The next pass is a top-down pass that uses the accumulated
        // information to assign absolute indices to each node.
        //
        // For each node, we want to compute its absolute index in
        // the overall array of nodes, and then we also want to compute
        // the index where its first descendent node will be placed
        // (which can then be used by child nodes to compute their
        // index).
        //
        // The start node in the CFG is special, and will always get
        // index zero, with its first desecendent at index 1.
        //
        BlockName startBlock = getBlockName(code->getFirstBlock());
        blockInfos[startBlock].nodeIndex = 0;
        blockInfos[startBlock].firstDescendentIndex = 1;
        //
        // For the remaining nodes, we'll compute them in a top-down
        // pass (using reverse postorder).
        //
        for (BlockName bb = blockCount - 1; bb >= 0; --bb)
        {
            // We will skip nodes without a parent in the dominator tree.
            // This should really only be the start node, but it might
            // happen that we have some unreachable nodes that shouldn't
            // appear in the dominator tree at all.
            //
            // TODO: make sure we either handle those correctly, or
            // else add a pass to eliminate unreachable blocks first.
            //
            BlockName parent = doms[bb];
            if (parent == kUndefined)
                continue;

            // The absolute index of a node is the absolute index for its
            // parent's descendent list, plus the relative offset of this
            // child node in its parent.
            //
            blockInfos[bb].nodeIndex =
                blockInfos[parent].firstDescendentIndex + blockInfos[bb].childOffsetInParent;

            // The other descendents of a node are always laid out in the space
            // after its immediate children. Thus, the index for where this node
            // will place its descendents (direct + indirect) must come after
            // the storage for the children of the parent.
            //
            blockInfos[bb].firstDescendentIndex = blockInfos[parent].firstDescendentIndex +
                                                  blockInfos[parent].childCount +
                                                  blockInfos[bb].descendentOffsetInParent;
        }

        // We now have all the information we need, and can start to fill in
        // the actual `IRDominatorTree` structure with the encoded information.
        //
        RefPtr<IRDominatorTree> dominatorTree = new IRDominatorTree();
        dominatorTree->code = code;
        dominatorTree->nodes.setCount(blockCount);
        dominatorTree->reachableSet = _Move(reachableSet);

        // We will iterate over all of the blocks, and fill in the corresponding
        // dominator tree node for each.
        //
        // Note that the number of the blocks (in postorder) and the numbering
        // of the nodes (in breadth-first order) will not match, so we have
        // to be careful around whehter we are working with a block index/name,
        // or a node index.
        //
        for (BlockName bb = 0; bb < blockCount; ++bb)
        {
            // Find the IR block, look up our pre-computed information,
            // and find the corresponding node in the dominator tree.
            //
            IRBlock* block = postorder[bb];
            BlockInfo const& blockInfo = blockInfos[bb];
            Int nodeIndex = blockInfo.nodeIndex;
            IRDominatorTree::Node& node = dominatorTree->nodes[nodeIndex];

            // We will now start filling in the node. Filling in the block is
            // trial, and while we are at it we can add an entry to the mapping
            // from the block to  the node index.
            //
            node.block = block;
            dominatorTree->mapBlockToIndex.add(block, nodeIndex);

            // Filling in the parent is easy enough, just with the detail that
            // we need to handle the invalid case explicitly (for a node with
            // no parent), and need to carefully map the block index `parent`
            // over to its corresponding node index.
            //
            BlockName parent = doms[bb];
            node.parent = parent == kUndefined ? IRDominatorTree::kInvalidIndex
                                               : blockInfos[parent].nodeIndex;

            // Finally we need to compute the range information to use for the
            // descendents (both immediate children and indirect descendents).
            //
            // All of the relevant information was computed in our two passes
            // above, so all that has to happen here is adding together the
            // absolute start index for the descendent range with the counts
            // we accumulated.
            //
            Int beginDescendents = blockInfo.firstDescendentIndex;
            Int endChildren = beginDescendents + blockInfo.childCount;
            //
            // The indirect descendents of a node will always come after
            // its direct descenents.
            //
            Int endDescendents = endChildren + blockInfo.indirectDescendentCount;
            node.beginDescendents = beginDescendents;
            node.endChildren = endChildren;
            node.endDescendents = endDescendents;
        }

#if 0
        // Let's do some ad hoc validation here, just to be sure we built the
        // data structure reasonably.
        for(BlockName ii = 0; ii < blockCount; ++ii)
        {
            for(BlockName jj = 0; jj < blockCount; ++jj)
            {
                IRBlock* i = postorder[ii];
                IRBlock* j = postorder[jj];

                SLANG_RELEASE_ASSERT(dominatorTree->immediatelyDominates(i, j) == (ii == doms[jj]));

                Int dd = jj;
                while(dd != kUndefined)
                {
                    if(dd == ii)
                        break;
                    dd = doms[dd];
                }
                SLANG_RELEASE_ASSERT(dominatorTree->dominates(i, j) == (dd != kUndefined));

            }
        }
#endif

        return dominatorTree;
    }
};


RefPtr<IRDominatorTree> computeDominatorTree(IRGlobalValueWithCode* code)
{
    DominatorTreeComputationContext context;
    return context.createDominatorTree(code);
}

} // namespace Slang
