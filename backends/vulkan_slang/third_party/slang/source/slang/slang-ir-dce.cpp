// slang-ir-dce.cpp
#include "slang-ir-dce.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
struct DeadCodeEliminationContext
{
    // This type implements a simple global DCE pass over
    // an entire module.
    //
    // We start with member variables to stand in for
    // the parameters that were passed to the top-level
    // `eliminateDeadCode` function.
    //
    IRModule* module;

    IRDeadCodeEliminationOptions options;

    // If we removed an inst, there may be still "weak references" to the inst.
    // These uses will be replaced with `undefInst`.
    IRInst* undefInst = nullptr;

    // Track if we have removed any phi parameters.
    // If so we need to rerun dce pass because after removing them
    // there could be new DCE opportunities.
    bool phiRemoved = false;

    // Querying whether an instruction has been
    // determined to be live is easy.
    // To speedup the test, we use the
    // `scratchData` field of each inst as the marker.
    //
    bool isInstAlive(IRInst* inst)
    {
        if (!inst)
            return false;
        return inst->scratchData != 0;
    }

    // We are going to do an iterative analysis
    // where we mark instructions we know are
    // live, and then see if that can help us
    // identify any other instructions that
    // must also be live.
    //
    // For this, we will use a work list of
    // instructions that have been marked
    // as live, but for which we haven't
    // looked at their impact on other
    // instructions.
    //
    List<IRInst*> workList;

    // When we discover that an instruction seems
    // to be live, we will add it to our set,
    // and also the work list, but only if we
    // haven't done so previously.
    //
    void markInstAsLive(IRInst* inst)
    {
        // Again, we safeguard against null instructions
        // just in case.
        //
        if (!inst)
            return;

        if (!inst->scratchData)
        {
            inst->scratchData = 1;
            workList.add(inst);
        }
    }

    IRInst* getUndefInst()
    {
        if (!undefInst)
        {
            IRBuilder builder(module);
            if (auto firstChild = module->getModuleInst()->getFirstChild())
                builder.setInsertBefore(firstChild);
            else
                builder.setInsertInto(module->getModuleInst());
            undefInst = Slang::getUndefInst(builder, module);
        }
        return undefInst;
    }

    bool processInst(IRInst* root)
    {
        bool result = false;

        module->invalidateAllAnalysis();

        for (;;)
        {
            // Clear the `alive` bits by initializing all scratchData to 0.
            initializeScratchData(root);

            workList.clear();

            // First of all, we know that the root instruction
            // should be considered as live, because otherwise
            // we'd end up eliminating it, so that is a
            // good place to start.
            //
            markInstAsLive(root);

            // Ensure there is a global undef inst that is always alive.
            // This undef inst will be used to fill in weak-referencing uses
            // whose used value is marked as dead and eliminated.
            // We always make sure this undef inst is available to prevent
            // infiniate oscilating loops.
            markInstAsLive(getUndefInst());

            // Marking the module as live should have
            // seeded our work list, so we can now start
            // processing entries off of our work list
            // until it goes dry.
            //
            while (workList.getCount())
            {
                auto inst = workList.getLast();
                workList.removeLast();

                if (!isChildInstOf(inst, root))
                    continue;

                // At this point we know that `inst` is live,
                // and we want to start considering which other
                // instructions must be live because of that
                // knowlege.
                //
                // A first easy case is that the parent (if any)
                // of a live instruction had better be live, or
                // else we might delete the parent, and
                // the child with it.
                //
                markInstAsLive(inst->getParent());

                // Next the type of a live instruction, and all
                // of its operands must also be live, or else
                // we won't be able to compute its value.
                //
                markInstAsLive(inst->getFullType());
                UInt operandCount = inst->getOperandCount();
                for (UInt ii = 0; ii < operandCount; ++ii)
                {
                    // There are some type of operands that needs to be treated as
                    // "weak" references -- they can never hold things alive, and
                    // whenever we delete the referenced value, these operands needs
                    // to be replaced with `undef`.
                    if (!isWeakReferenceOperand(inst, ii))
                        markInstAsLive(inst->getOperand(ii));
                }

                // Finally, we need to consider the children
                // and decorations of the instruction.
                //
                // Note that just because an instruction is
                // live doesn't mean its children must be, or
                // else we'd never eliminate *anything* (we
                // marked the whole module as live, and everything
                // is a transitive child of the module).
                //
                // Decorations, in contrast, are always live if their
                // parents are (because we don't want to silently drop
                // decorations). It is still important to *mark*
                // decorations as live, because they have operands,
                // and those operands need to be marked as live.
                // We will fold decorations into the same loop
                // as children for simplicity.
                //
                // To keep the code here simple, we'll defer the
                // decision of whether a child (or decoration)
                // should be live when its parent is to a subroutine.
                //

                for (auto child : inst->getDecorationsAndChildren())
                {
                    if (shouldInstBeLiveIfParentIsLive(child))
                    {
                        // In this case, we know `inst` is live and
                        // its `child` should be live if its parent is,
                        // so the `child` must be live too.
                        //
                        markInstAsLive(child);
                    }
                }
            }

            // If our work list runs dry, that means we've reached a steady
            // state where everything that is transitively relevant to
            // the "outputs" of the module has been marked as live.
            //
            // Now we can simply walk through all of our instructions
            // recursively and eliminate those that are "dead" by
            // virtue of not having been found live.
            //
            phiRemoved = false;
            result |= eliminateDeadInstsRec(root);


            if (!phiRemoved)
                break;
        }
        return result;
    }

    // Given the basic infrastructrure above, let's
    // dive into the task of actually finding all
    // the live code in a module.
    //
    bool processModule() { return processInst(module->getModuleInst()); }

    bool eliminateDeadInstsRec(IRInst* inst)
    {
        bool changed = false;
        // Given the instruction `inst` we need to eliminate
        // any dead code at, or under it.
        //
        // The easy case is if `inst` is dead (that is, not live).
        //
        if (!isInstAlive(inst))
        {
            // We can simply remove and deallocate `inst` because it is
            // dead, and not worry about any of its descendents,
            // because they must have been dead too (since we always
            // mark the parent of a live instruction as live).
            //
            if (inst->hasUses())
            {
                inst->replaceUsesWith(getUndefInst());
            }

            if (inst->getOp() == kIROp_Param)
            {
                // For Phi parameters, we need to update all branch arguments.
                removePhiArgs(inst);
                phiRemoved = true;
            }
            inst->removeAndDeallocate();
            changed = true;
        }
        else
        {
            // If `inst` is live, then we need to deal with the possibility
            // that its children/decorations (or descendents in general)
            // might still be dead.
            //
            // The biggest wrinkle is that we walk the linked list of
            // children/decorations a bit carefully, because eliminating one inst
            // may cause the other nodes to be hoisted out of the current scope.
            // We need to cache all children in a work list to ensure they are
            // properly traversed.
            //
            List<IRInst*> children;
            for (auto child : inst->getDecorationsAndChildren())
                children.add(child);
            for (IRInst* child : children)
            {
                changed |= eliminateDeadInstsRec(child);
            }
            if (changed)
            {
                // If the function body is changed, invalidate its dominator tree.
                if (auto func = as<IRGlobalValueWithCode>(inst))
                    module->invalidateAnalysisForInst(func);
            }
        }
        return changed;
    }

    // Now we come to the decision procedure we put off before:
    // should a given `inst` be live if its parent is?
    //
    bool shouldInstBeLiveIfParentIsLive(IRInst* inst)
    {
        return Slang::shouldInstBeLiveIfParentIsLive(inst, options);
    }
};

bool isPtrUsed(IRInst* ptrInst)
{
    for (auto use = ptrInst->firstUse; use; use = use->nextUse)
    {
        if (as<IRLoad>(use->getUser()))
            return true;
        else if (as<IRCall>(use->getUser())) // TODO: narrow this case to 'inout' parameters only.
            return true;
        else if (as<IRPtrTypeBase>(use->getUser()->getDataType()) && isPtrUsed(use->getUser()))
            return true;
    }

    return false;
}

bool isFieldUsed(IRStructField* fieldInst)
{
    auto structKey = fieldInst->getKey();
    for (auto use = structKey->firstUse; use; use = use->nextUse)
    {
        if (as<IRPtrTypeBase>(use->getUser()->getDataType()) && isPtrUsed(use->getUser()))
            return true;

        if (as<IRFieldExtract>(use->getUser()))
            return true;
    }

    // Check fields that have this field as a sub-field.
    auto parentType = cast<IRStructType>(fieldInst->getParent());

    if (as<IRModuleInst>(parentType->getParent()))
    {
        for (auto use = parentType->firstUse; use; use = use->nextUse)
        {
            auto useField = as<IRStructField>(use->getUser());
            if (useField && isFieldUsed(useField))
                return true;
        }
    }
    else if (as<IRBlock>(parentType->getParent()))
    {
        if (auto genericParentType = as<IRGeneric>(parentType->getParent()))
        {
            List<IRSpecialize*> specInsts;
            for (auto use = genericParentType->firstUse; use; use = use->nextUse)
            {
                if (auto specInst = as<IRSpecialize>(use->getUser()))
                    specInsts.add(specInst);
            }

            for (auto specInst : specInsts)
            {
                for (auto use = specInst->firstUse; use; use = use->nextUse)
                {
                    auto useField = as<IRStructField>(use->getUser());
                    if (useField && isFieldUsed(useField))
                        return true;
                }
            }
        }
    }

    return false;
}

bool removeStoresIntoInst(IRInst* ptrInst)
{
    bool changed = false;

    List<IRInst*> storesToRemove;
    for (auto use = ptrInst->firstUse; use; use = use->nextUse)
    {
        // If this is a store, remove it.
        if (auto store = as<IRStore>(use->getUser()))
        {
            if (store->getPtr() == ptrInst)
                storesToRemove.add(store);
        }

        // If there are any stores into a 'sub-object' of the pointer,
        // remove them.
        //

        if (auto subAddr = as<IRFieldAddress>(use->getUser()))
            changed |= removeStoresIntoInst(subAddr);

        if (auto subIndex = as<IRGetElementPtr>(use->getUser()))
            changed |= removeStoresIntoInst(subIndex);
    }

    for (auto store : storesToRemove)
    {
        changed = true;
        store->removeAndDeallocate();
    }

    return changed;
}

bool removeStoresIntoField(IRStructField* field)
{
    return removeStoresIntoInst(field->getKey());
}

bool trimMakeStructOperands(IRStructField* field)
{
    // TODO: This can be sped up by considering the full set of fields instead
    // of one at a time.

    bool changed = false;
    auto structType = cast<IRStructType>(field->getParent());

    UIndex indexInStruct = 0;
    for (auto _field : structType->getFields())
    {
        if (field == _field)
            break;
        indexInStruct++;
    }

    List<IRInst*> workList;
    for (auto use = structType->firstUse; use; use = use->nextUse)
    {
        if (use->getUser()->getOp() == kIROp_MakeStruct)
        {
            workList.add(use->getUser());
        }
    }

    IRBuilder builder(field->getModule());

    for (auto makeStruct : workList)
    {
        // Make a replacement list of operands.
        List<IRInst*> newOperands;
        for (UInt index = 0; index < makeStruct->getOperandCount(); ++index)
        {
            if (index == indexInStruct)
            {
                // skip..
                changed = true;
                continue;
            }
            else
            {
                newOperands.add(makeStruct->getOperand(index));
            }
        }

        builder.setInsertAfter(makeStruct);
        auto newMakeStruct = builder.emitMakeStruct(makeStruct->getFullType(), newOperands);
        makeStruct->replaceUsesWith(newMakeStruct);
    }

    for (auto makeStruct : workList)
    {
        makeStruct->removeAndDeallocate();
    }

    return changed;
}

bool isStructEmpty(IRType* type)
{
    auto structType = as<IRStructType>(type);
    if (!structType)
        return false;

    UCount nonEmptyFieldCount = 0;
    for (auto field : structType->getFields())
    {
        if (as<IRVoidType>(field->getFieldType()))
            continue;
        if (isStructEmpty(field->getFieldType()))
            continue;
        nonEmptyFieldCount++;
    }

    return nonEmptyFieldCount == 0;
}

bool trimOptimizableType(IRStructType* type)
{
    bool changed = false;
    List<IRStructField*> fieldsToRemove;
    for (auto field : type->getFields())
    {
        // We'll ignore void-type fields, since they're handled differently.
        if (as<IRVoidType>(field->getFieldType()))
            continue;

        // ... same for empty struct fields.
        if (as<IRStructType>(field->getFieldType()) && isStructEmpty(field->getFieldType()))
            continue;

        if (!isFieldUsed(field))
            fieldsToRemove.add(field);
    }

    for (auto field : fieldsToRemove)
    {
        changed |= removeStoresIntoField(field);
        changed |= trimMakeStructOperands(field);
        field->removeFromParent();
    }

    for (auto field : fieldsToRemove)
    {
        changed = true;
        field->removeAndDeallocate();
    }

    return changed;
}

bool trimOptimizableTypes(IRModule* module)
{
    bool changed = false;
    for (auto inst : module->getGlobalInsts())
    {
        if (auto type = as<IRStructType>(inst))
        {
            if (type->findDecoration<IROptimizableTypeDecoration>())
                changed |= trimOptimizableType(type);
        }
    }
    return changed;
}

bool shouldInstBeLiveIfParentIsLive(IRInst* inst, IRDeadCodeEliminationOptions options)
{
    // The main source of confusion/complexity here is that
    // we are using the same routine to decide:
    //
    // * Should some ordinary instruction in a basic block be kept around?
    // * Should a basic block in some function be kept around?
    // * Should a function/type/variable in a module be kept around?
    //
    // Still, there are a few basic patterns we can observe.
    // First, if `inst` is an instruction that might have some effects
    // when it is executed, then we should keep it around.
    //
    SideEffectAnalysisOptions sideEffectOptions = options.useFastAnalysis
                                                      ? SideEffectAnalysisOptions::None
                                                      : SideEffectAnalysisOptions::UseDominanceTree;

    if (inst->mightHaveSideEffects(sideEffectOptions))
    {
        return true;
    }
    //
    // The `mightHaveSideEffects` query is conservative, and will
    // return `true` as its default mode, so once we are past that
    // query we know that `inst` is either something "structural"
    // (that makes up the program) rather than executable, or it
    // is executable but was on an allow-list of things that are
    // safe to eliminate.

    // Most top-level objects (functions, types, etc.) obviously
    // do *not* have side effects. That creates the risk that
    // we'll just go ahead and eliminate every single function/type
    // in a module. There needs to be a way to identify the
    // functions we want to keep around, and for right now
    // that is handled with the `[keepAlive]` decoration.
    //
    if (inst->findDecorationImpl(kIROp_KeepAliveDecoration))
        return true;
    //
    // We also consider anything with an `[export(...)]` as live,
    // when the appropriate option has been set.
    //
    // Note: our current approach to linking for back-end compilation
    // leaves many linakge decorations in place that we seemingly
    // don't need/want, so this option currently can't be enabled
    // unconditionally.
    //
    if (options.keepExportsAlive)
    {
        bool isImported = false;
        bool shouldKeptAliveIfImported = false;
        IRInst* innerInst = inst;
        if (auto genInst = as<IRGeneric>(inst))
        {
            innerInst = findInnerMostGenericReturnVal(genInst);
        }
        for (auto decor : inst->getDecorations())
        {
            switch (decor->getOp())
            {
            case kIROp_ExportDecoration:
                return true;
            case kIROp_ImportDecoration:
                isImported = true;
                break;
            }
        }
        for (auto decor : innerInst->getDecorations())
        {
            switch (decor->getOp())
            {
            case kIROp_ForwardDerivativeDecoration:
            case kIROp_UserDefinedBackwardDerivativeDecoration:
            case kIROp_PrimalSubstituteDecoration:
                shouldKeptAliveIfImported = true;
                break;
            }
        }
        if (isImported && shouldKeptAliveIfImported)
            return true;
    }

    if (options.keepLayoutsAlive && inst->findDecoration<IRLayoutDecoration>())
    {
        return true;
    }

    // A basic block is an interesting case. Knowing that a function
    // is live means that its entry block is live, but the liveness
    // of any other blocks is determined by whether they are referenced
    // by other instructions (e.g., a branch from one block to
    // another).
    //
    if (auto block = as<IRBlock>(inst))
    {
        // To determine whether this is the first block in its
        // parent function (or what-have-you) we can simply
        // check if there is a previous block before it.
        //
        auto prevBlock = block->getPrevBlock();
        return prevBlock == nullptr;
    }

    // There are a few special cases of "structural" instructions
    // that we don't want to eliminate, so we'll check for those next.
    //
    switch (inst->getOp())
    {
        // Function parameters obviously shouldn't get eliminated,
        // even if nothing references them.
        //
    case kIROp_Param:
        return isFirstBlock(inst->getParent());

        // IR struct types and witness tables are currently kludged
        // so that they have child instructions that represent their
        // entries (effectively `(key,value)` pairs), and those child
        // instructions are never directly referenced (e.g., an access
        // to a struct field references the *key* but not the `(key,value)`
        // pair that is the `IRField` instruction.
        //
        // TODO: at some point the IR should use a different representation
        // for struct types and witness tables that does away with
        // this problem.
        //
    case kIROp_StructField:
    case kIROp_WitnessTableEntry:
        return true;

    case kIROp_GlobalParam:
        return options.keepGlobalParamsAlive;
    default:
        break;
    }

    // If none of the explicit cases above matched, then we will consider
    // the instruction to not be live just because its parent is. Further
    // analysis could still lead to a change in the status of `inst`, if
    // an instruction that uses it as an operand is marked live.
    //
    return false;
}

bool isWeakReferenceOperand(IRInst* inst, UInt operandIndex)
{
    // There are some type of operands that needs to be treated as
    // "weak" references -- they can never hold things alive, and
    // whenever we delete the referenced value, these operands needs
    // to be replaced with `undef`.
    switch (inst->getOp())
    {
    case kIROp_BoundInterfaceType:
        if (inst->getOperand(operandIndex)->getOp() == kIROp_WitnessTable)
            return true;
        break;
    case kIROp_SpecializationDictionaryItem:
        // Ignore all operands of SpecializationDictionaryItem.
        // This inst is used as a cache and shouldn't hold anything alive.
        return true;
    default:
        break;
    }
    return false;
}

// The top-level function for invoking the DCE pass
// is straighforward. We set up the context object
// and then defer to it for the real work.
//
bool eliminateDeadCode(IRModule* module, IRDeadCodeEliminationOptions const& options)
{
    DeadCodeEliminationContext context;
    context.module = module;
    context.options = options;
    return context.processModule();
}

bool eliminateDeadCode(IRInst* root, IRDeadCodeEliminationOptions const& options)
{
    DeadCodeEliminationContext context;
    context.module = root->getModule();
    context.options = options;
    return context.processInst(root);
}

} // namespace Slang
