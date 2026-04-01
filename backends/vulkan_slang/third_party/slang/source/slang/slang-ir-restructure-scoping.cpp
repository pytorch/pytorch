// slang-ir-restructure-scoping.cpp
#include "slang-ir-restructure-scoping.h"

#include "slang-ir-insts.h"
#include "slang-ir-restructure.h"
#include "slang-ir.h"

namespace Slang
{

/// Try to find the first structured region that represents `block`
///
/// In general the same block may appear as multiple regions,
/// so this will return the first region in the linked list.
static SimpleRegion* getFirstRegionForBlock(RegionTree* regionTree, IRBlock* block)
{
    SimpleRegion* region = nullptr;
    if (regionTree->mapBlockToRegion.tryGetValue(block, region))
    {
        return region;
    }
    return nullptr;
}

/// Try to find the first structured region that contains `inst`.
static SimpleRegion* getFirstRegionForInst(RegionTree* regionTree, IRInst* inst)
{
    auto ii = inst;
    while (ii)
    {
        if (auto block = as<IRBlock>(ii))
            return getFirstRegionForBlock(regionTree, block);

        ii = ii->getParent();
    }

    return nullptr;
}

/// Compute the depth of a node in the region tree.
///
/// This is the number of nodes (including `region`)
/// on a path from `region` to the root.
///
static Int computeDepth(Region* region)
{
    Int depth = 0;
    for (Region* rr = region; rr; rr = rr->getParent())
    {
        depth++;
    }
    return depth;
}

/// Get the `n`th ancestor of `region`.
///
/// When `n` is zero, this returns `region`.
/// When `n` is one, this returns the parent of `region`, and so forth.
///
static Region* getAncestor(Region* region, Int n)
{
    Region* rr = region;
    for (Int ii = 0; ii < n; ++ii)
    {
        SLANG_ASSERT(rr);
        rr = rr->getParent();
    }
    return rr;
}

/// Find a region that is an ancestor of both `left` and `right`.
static Region* findCommonAncestorRegion(Region* left, Region* right)
{
    // Rather than blinding search through each ancestor of `left`
    // and see if it is also an ancestor of `right` and vice-versa,
    // let's try to be smart about this.
    //
    // We will start by computing the depth of `left` and `right`:
    //
    Int leftDepth = computeDepth(left);
    Int rightDepth = computeDepth(right);

    // Whatever the common ancestor is, it can't be any deeper
    // than the minimum of these two depths.
    //
    Int minDepth = Math::Min(leftDepth, rightDepth);

    // Let's fetch the ancestor of each of `left` and `right`
    // corresponding to that depth:
    //
    Region* leftAncestor = getAncestor(left, leftDepth - minDepth);
    Region* rightAncestor = getAncestor(right, rightDepth - minDepth);

    // Now we know that `leftAncestor` and `rightAncestor`
    // must have the same depth. Let's go ahead and assert
    // it just to be safe:
    //
    SLANG_ASSERT(computeDepth(leftAncestor) == minDepth);
    SLANG_ASSERT(computeDepth(rightAncestor) == minDepth);

    // If `leftAncestor` and `rightAncestor` are the same node,
    // then we've found a common ancestor, otherwise we should
    // look at their parents. Because the depth must match
    // on both sides, we will never risk missing an ancestor.
    //
    while (leftAncestor != rightAncestor)
    {
        leftAncestor = leftAncestor->getParent();
        rightAncestor = rightAncestor->getParent();
    }

    // Okay, we've found a common ancestor.
    //
    Region* commonAncestor = leftAncestor;
    return commonAncestor;
}

/// Find a simple region that is an ancestor of both `left` and `right`.
static SimpleRegion* findSimpleCommonAncestorRegion(Region* left, Region* right)
{
    // Start by finding a common ancestor without worrying about it being simple.
    Region* ancestor = findCommonAncestorRegion(left, right);

    // Now search for a simple region up the tree.
    while (ancestor)
    {
        if (ancestor->getFlavor() == Region::Flavor::Simple)
            return (SimpleRegion*)ancestor;

        ancestor = ancestor->getParent();
    }

    // This shouldn't ever occur. The root of the region tree should
    // be a simple regions that represents the entry block of the
    // function.
    //
    SLANG_UNEXPECTED("no common ancestor found in region tree");
    UNREACHABLE_RETURN(nullptr);
}

IRInst* getDefaultInitVal(IRBuilder* builder, IRType* type)
{
    switch (type->getOp())
    {
    default:
        return nullptr;

    case kIROp_BoolType:
        return builder->getBoolValue(false);

    case kIROp_IntType:
    case kIROp_UIntType:
    case kIROp_UInt64Type:
        return builder->getIntValue(type, 0);

    case kIROp_HalfType:
    case kIROp_FloatType:
    case kIROp_DoubleType:
        return builder->getFloatValue(type, 0.0);

        // TODO: handle vector/matrix types here, by
        // creating an appropriate scalar value and
        // then "splatting" it.
    }
}

/// Initialize a variable to a sane default value, if possible.
void defaultInitializeVar(IRBuilder* builder, IRVar* var, IRType* type)
{
    IRInst* initVal = nullptr;
    switch (type->getOp())
    {
    case kIROp_VoidType:
    default:
        // By default, see if we can synthesize an IR value
        // to be used as the default, and allow the logic
        // below to store it into the variable.
        initVal = getDefaultInitVal(builder, type);
        break;

        // TODO: Handle aggregate types (structures, arrays)
        // explicitly here, since they need to be careful about
        // the cases where an element/field type might not
        // be something we can default-initialize.
    }

    if (initVal)
    {
        builder->emitStore(var, initVal);
    }
}

/// Detect and fix any structured scoping issues for a given `def` instruction.
///
/// The `defRegion` should be the region that contains `def`, and `regionTree`
/// should be the region tree for the function that contains `def`.
///
static void fixValueScopingForInst(
    IRInst* def,
    SimpleRegion* defRegion,
    RegionTree* regionTree,
    bool isInstAlwaysFolded)
{
    // This algorithm should not consider "phi nodes" for now,
    // because the emit logic will already create variables for them.
    // We could consider folding the logic to move out of SSA form
    // into this function, but that would add a lot of complexity for now.
    if (def->getOp() == kIROp_Param)
        return;

    // We would have a scoping violation if there exists some
    // use `u` of `def` such that the region containing `u`
    // (call it `useRegion`) is not a descendent of `defRegion`
    // in the region tree.
    //
    // If there are no scoping violations, we don't want to do
    // anything. If there *are* any scoping violations, then
    // we ill need to introduce a temporary `tmp`, store into
    // it right after `def`, and then load from it at any "bad"
    // use sites.
    //
    // Of course, for the whole thing to work, we also need
    // to put `tmp` into a block somwhere, and it needs to
    // be a block that is visible to all of the uses, or we
    // are just back int the same mess again.
    //
    // The right block to use for inserting `tmp` is the least
    // common ancestor of `def` and all the "bad" uses, so
    // we will get a bit "clever" and fold in the search for
    // bad uses with the computation of the region we should
    // insert `tmp` into (to avoid looping over the uses
    // twice).
    //
    SimpleRegion* insertRegion = defRegion;
    IRVar* tmp = nullptr;

    // If we end up needing to insert code we'll need an IR builder,
    // so we will go ahead and create one now.
    //
    // TODO: the logic to compute `module` here could be hoisted
    // out earlier, rather than being done per-instruction.
    //
    IRModule* module = regionTree->irCode->getModule();

    IRBuilder builder(module);

    // Because we will be changing some of the uses of `def`
    // to use other values while we iterate the list, we
    // need to be a bit careful and extract the next use
    // in the linked list *before* we operator on `u`.
    //
    IRUse* nextUse = nullptr;
    for (auto u = def->firstUse; u; u = nextUse)
    {
        nextUse = u->nextUse;

        // Looking at the use site `u`, we'd like to check if
        // it violates our scoping rules.
        //
        // As a simple early-exit case, if the user is in
        // the same block as the definition, there are no problems.
        //
        IRInst* user = u->getUser();
        if (user->getParent() == defRegion->block)
            continue;

        // Otherwise, let's find the structures control-flow
        // region that holds the user. We expect to always
        // find one, because the use site must be in the same
        // function.
        //
        // TODO: Double check that logic if we ever introduce
        // things like nested function.
        //
        SimpleRegion* useRegion = getFirstRegionForInst(regionTree, user);

        // If there is no region associated with the use, then
        // the use must be in unreachable code (part of the CFG,
        // but not part of the region tree). We will skip
        // such uses for now, since they won't even appear in
        // the output.
        //
        if (!useRegion)
            continue;

        // Now we want to check if `useRegion` is a child/descendent
        // of a region that has the same block as `defRegion`.
        // If it is, then there is no scoping problem with this use.
        //
        if (useRegion->isDescendentOf(defRegion->block))
            continue;

        // If we've gotten this far, we know that `u` is a "bad"
        // use of `def`, and needs fixing.
        //
        // For insts that are always fold into use sites, we try to hoist them
        // to as early as possible, and then leave it there.
        //
        if (isInstAlwaysFolded)
        {
            def->removeFromParent();
            addHoistableInst(&builder, def);
            continue;
        }

        // For non-hoistable insts, we will use a temporary variable to resolve
        // the bad scoping, creating it on-demand when we ecounter a first "bad" use, and
        // then re-using that temporary for any subsequent bad uses.
        //
        if (!tmp)
        {
            // If the value is *already* a temporary variable, then
            // we are really just trying to fix the scoping of the
            // variable declaration itself, and the variable can
            // effectively be its own temporary.
            //
            if (auto varDef = as<IRVar>(def))
            {
                tmp = varDef;
            }
            else
            {
                // We will create a temporary to represent `def`,
                // and insert a `store` into it right after `def`.
                //
                // Note: we are inserting the new variable right
                // after `def` for now, just because we don't
                // yet know the final region that it should be
                // placed into. We will move it to the correct
                // location when we are done.
                //
                builder.setInsertBefore(def->getNextInst());
                tmp = builder.emitVar(def->getDataType());
                builder.emitStore(tmp, def);
                //
                // Note: the lifetime for the new variable starts
                // right after the store we have emitted.
            }
        }

        // In order to know where `tmp` should be defined
        // at the end of the algorithm, we need to compute
        // a valid `insertRegion` that is an ancestor of
        // all of the use sites (and it also a simple region
        // so that we can insert into its IR block).
        //
        // We need to deal with one complexity in our restructuring
        // process, which is that a block may be duplicated into
        // one or more regions, so we loop over all the regions
        // for the same block as `useRegion`.
        //
        for (auto rr = useRegion; rr; rr = rr->nextSimpleRegionForSameBlock)
        {
            insertRegion = findSimpleCommonAncestorRegion(insertRegion, rr);
        }

        // We need to fix up the use `u`, but the way we fix
        // it depends on whether we moving `def` itself (in which
        // case `tmp` and `def` are the same), or if we have
        // introduced an intermediate temporary.
        //
        if (def == tmp)
        {
            // If we are moving the definition itself, we don't
            // need to do any kind of fix-up work at use sites.
        }
        else
        {
            // Othwerise we need to fix up the use `use` so
            // that it uses a value loaded from `tmp` instead
            // of `def`.
            //
            builder.setInsertBefore(user);
            IRInst* tmpVal = builder.emitLoad(tmp);

            // We are clobbering the value used by the `IRUse` `u`,
            // while will cut it out of the list of uses for `def`.
            // We need to be careful when doing this to not disrupt
            // our iteration of the uses of `def`, so we carefully
            // used the `nextUse` temporary at the start of the loop.
            //
            // Note(tfoley): This is more subtle than the comment makes
            // it out to be, because we are *also* injecting a new use
            // of `def` in the logic that creates `tmp` (because `def`
            // is used as an operand to the `store` that initializes
            // `tmp`). We really ought to work on a copy of the use-def
            // information.
            //
            u->set(tmpVal);
        }
    }

    // At the end of the loop, the `tmp` variable will have
    // been created if and only if we fixed up anything.
    //
    if (tmp)
    {
        // If we created a temporary, then now we need to move
        // its definition to the right place, which is the
        // `insertRegion` that we computed during the loop.
        //
        // We'd like to insert our temporary near the top
        // of the region, since that is the conventional
        // place for local variables to go.
        //
        tmp->insertBefore(insertRegion->block->getFirstOrdinaryInst());

        // The whole point of the transformation we are doing
        // here is that `def` is not on the "obvious" control
        // flow path to one or more uses (which are now using
        // `tmp`), but that means that it might not be "obvious"
        // to a downstream compiler that `tmp` always gets
        // initialized (by the code we inserted after `def`)
        // before each of these use sites.
        //
        // We *know* that things are valid as long as our
        // dominator tree was valid - there is no way to
        // get to the block that loads from `tmp` without passing
        // through the block that computes `def` (and then
        // stores it into `tmp`) first.
        //
        // To avoid warnings/errros, we will go ahead and try
        // to emit logic to "default initialize" the `tmp`
        // variable if possible.
        //
        builder.setInsertBefore(tmp->getNextInst());
        defaultInitializeVar(&builder, tmp, def->getDataType());
    }
}

void fixValueScoping(RegionTree* regionTree, const Func<bool, IRInst*>& shouldAlwaysFoldInst)
{
    // We are going to have to walk through every instruction
    // in the code of the function to detect an bad cases.
    //
    auto code = regionTree->irCode;
    for (auto block : code->getBlocks())
    {
        // All of the instruction in `block` will have the same
        // parent region, so we will look it up now rather than
        // have to re-do this work on a per-instruction basis.
        //
        auto parentRegion = getFirstRegionForBlock(regionTree, block);

        // If a block has no region then it must be unreachable,
        // so we will skip it entirely for this pass.
        //
        // TODO: we should be eliminating unrechable blocks anyway.
        //
        if (!parentRegion)
            continue;

        // Note: This pass will end up modifying the IR while also
        // iterating over it. As such, we need to be careful not
        // to let our iteration logic get confused.
        //
        // In particular, it is possible that `inst` will get moved
        // to another block, as a way to resolve scoping issues, and
        // if we did not account for that result, we might end up
        // walking to the next instruction after `inst` even though
        // it isn't inside `block`.
        //
        // We defensively cache the next instruction to visit so that
        // we can continue our iteration after `inst` even if it gets
        // moved. For now we are confident that the operations on
        // `inst` won't affect `nextInst`, since the pass is not supposed
        // to move or delete any *other* instructions.
        //
        IRInst* nextInst = nullptr;
        for (auto inst = block->getFirstOrdinaryInst(); inst; inst = nextInst)
        {
            nextInst = inst->getNextInst();
            bool isInstAlwaysFolded = shouldAlwaysFoldInst(inst);
            fixValueScopingForInst(inst, parentRegion, regionTree, isInstAlwaysFolded);
        }
    }
}

} // namespace Slang
