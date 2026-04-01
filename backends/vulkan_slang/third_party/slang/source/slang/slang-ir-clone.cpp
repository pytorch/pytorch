// slang-ir-clone.cpp
#include "slang-ir-clone.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"
#include "slang-mangle.h"


namespace Slang
{

IRInst* lookUp(IRCloneEnv* env, IRInst* oldVal)
{
    for (auto ee = env; ee; ee = ee->parent)
    {
        IRInst* newVal = nullptr;
        if (ee->mapOldValToNew.tryGetValue(oldVal, newVal))
            return newVal;
    }
    return nullptr;
}

IRInst* findCloneForOperand(IRCloneEnv* env, IRInst* oldOperand)
{
    if (!oldOperand)
        return nullptr;

    // If there is a registered replacement for
    // the existing operand, then use it.
    //
    if (IRInst* newVal = lookUp(env, oldOperand))
        return newVal;

    // Otherwise, we assume that the caller wants
    // to default to using existing values wherever
    // an explicit replacement hasn't been registered.
    //
    // This is, notably, the right default whenever
    // `oldOperand` is a global value or constant
    // and our cloned code will sit in the same
    // module as the original.
    //
    // TODO: We could make this a customization point
    // down the road, if we ever had a case where
    // we want to clone things with a different policy.
    //
    return oldOperand;
}

IRInst* cloneInstAndOperands(IRCloneEnv* env, IRBuilder* builder, IRInst* oldInst)
{
    SLANG_ASSERT(env);
    SLANG_ASSERT(builder);
    SLANG_ASSERT(oldInst);

    // We start by mapping the type of the orignal instruction
    // to its replacement value, if any.
    //
    auto oldType = oldInst->getFullType();
    auto newType = (IRType*)findCloneForOperand(env, oldType);

    // Pointer literals need to be handled separately, as they carry other data
    // than just the operands.
    if (oldInst->getOp() == kIROp_PtrLit)
    {
        auto oldPtr = as<IRPtrLit>(oldInst);
        return builder->getPtrValue(newType, oldPtr->value.ptrVal);
    }

    // This logic will not handle any instructions
    // with special-case data attached, but that only
    // applies to `IRConstant`s at this point, and those
    // should only appear at the global scope rather than
    // in function bodies.
    //
    // TODO: It would be easy enough to extend this logic
    // to handle constants gracefully, if it ever comes up.
    //
    SLANG_ASSERT(!as<IRConstant>(oldInst));

    // Next we will iterate over the operands of `oldInst`
    // to find their replacements and install them as
    // the operands of `newInst`.
    //
    UInt operandCount = oldInst->getOperandCount();

    ShortList<IRInst*> newOperands;
    newOperands.setCount(operandCount);
    for (UInt ii = 0; ii < operandCount; ++ii)
    {
        auto oldOperand = oldInst->getOperand(ii);
        auto newOperand = findCloneForOperand(env, oldOperand);

        newOperands[ii] = newOperand;
    }

    // Finally we create the inst with the updated operands.
    auto newInst = builder->emitIntrinsicInst(
        newType,
        oldInst->getOp(),
        operandCount,
        newOperands.getArrayView().getBuffer());

    newInst->sourceLoc = oldInst->sourceLoc;

    return newInst;
}

// Copy the linkage decoration of oldInst (if present) and specialize it for target.
static void specializeLinkageDecoration(IRInst* target, IRSpecialize* oldInst, IRBuilder* builder)
{
    auto gen = as<IRGeneric>(oldInst->getBase());
    if (gen)
    {
        auto genLinkage = gen->findDecoration<IRLinkageDecoration>();
        if (genLinkage)
        {
            bool isExport = as<IRExportDecoration>(genLinkage);
            StringBuilder sb;
            sb.append(genLinkage->getMangledName());
            sb.append("G");
            IRSpecialize* specializationProvider = oldInst;
            if (auto targetAsSpec = as<IRSpecialize>(target))
            {
                specializationProvider = targetAsSpec;
            }
            for (UInt i = 0; i < specializationProvider->getArgCount(); ++i)
            {
                auto arg = specializationProvider->getArg(i);
                sb.append(i);
                if (auto typeLinkage = arg->findDecoration<IRLinkageDecoration>())
                {
                    sb.append(typeLinkage->getMangledName());
                }
                else
                {
                    // getTypeNameHint may produce a name with characters that can't
                    // be part of an identifier, so we need to filter it afterward.
                    StringBuilder tmp;
                    getTypeNameHint(tmp, arg);
                    emitNameForLinkage(sb, tmp.getUnownedSlice());
                }
            }
            if (auto previousLinkage = target->findDecoration<IRLinkageDecoration>())
            {
                // Overwrite the previous linkage decoration, since it was not specialized
                previousLinkage->setOperand(0, builder->getStringValue(sb.getUnownedSlice()));
            }
            else
            {
                if (isExport)
                {
                    builder->addExportDecoration(target, sb.getUnownedSlice());
                }
                else
                {
                    builder->addImportDecoration(target, sb.getUnownedSlice());
                }
            }
        }
    }
}

// The complexity of the second phase of cloning (the
// one that deals with decorations and children) comes
// from the fact that it needs to sequence the two phases
// of cloning for any child instructions. We will do this
// by performing the first phase of cloning, and building
// up a list of children that require the second phase of processing.
// Each entry in that list will be a pair of an old instruction
// and its new clone.
//
struct IRCloningOldNewPair
{
    IRInst* oldInst;
    IRInst* newInst;
};

// We will use an internal variant of `cloneInstDecorationsAndChildren`
// that modifies the provided `env` as it goes as the main
// workhorse, since we need to make sure that instructions in
// earlier blocks are visible to those in other, later, blocks
// when cloning a function, so that strict scoping along the
// lines of the nesting of instructions isn't sufficient.
//
static void _cloneInstDecorationsAndChildren(
    IRCloneEnv* env,
    IRModule* module,
    IRInst* oldInst,
    IRInst* newInst)
{
    SLANG_ASSERT(env);
    SLANG_ASSERT(oldInst);
    SLANG_ASSERT(newInst);

    // We will set up an IR builder that inserts
    // into the new parent instruction.
    //
    IRBuilder builderStorage(module);
    auto builder = &builderStorage;
    builder->setInsertInto(newInst);

    // If `newInst` already has non-decoration children, we want to
    // insert the new children between the existing decoration and non-decoration children
    // so that we maintain the invariant that all decorations are defined before non-decorations.
    if (auto firstChild = newInst->getFirstChild())
    {
        builder->setInsertBefore(firstChild);
    }

    // When applying the first phase of cloning to
    // children, we will keep track of those that
    // require the second phase.
    //
    List<IRCloningOldNewPair> pairs;
    ShortList<IRCloningOldNewPair> paramPairs;

    for (auto oldChild : oldInst->getDecorationsAndChildren())
    {
        // As a very subtle special case, if one of the children
        // of our `oldInst` already has a registered replacement,
        // then we don't want to clone it (not least because
        // the `Dictionary::Add` method would give us an error
        // when we try to insert a new value for the same key).
        //
        // This arises for entries in `mapOldValToNew` that were
        // seeded before cloning begain (e.g., function
        // parameters that are to be replaced).
        //
        if (lookUp(env, oldChild))
            continue;

        // Now we can perform the first phase of cloning
        // on the child, and register it in our map from
        // old to new values.
        //
        IRInst* newChild = nullptr;
        if (oldChild->getOp() == kIROp_Param)
        {
            // For parameters, don't clone its type just yet, since
            // the type might be a forward reference to things defined
            // later in the block that we haven't cloned and registered yet.
            newChild = builder->emitParam(nullptr);
            paramPairs.add({oldChild, newChild});
        }
        else
        {
            newChild = cloneInstAndOperands(env, builder, oldChild);
        }
        env->mapOldValToNew.add(oldChild, newChild);

        // If and only if the old child had decorations
        // or children, we will register it into our
        // list for processing in the second phase.
        //
        if (oldChild->getFirstDecorationOrChild())
        {
            pairs.add({oldChild, newChild});
        }
    }

    // Once we have done first-phase processing for
    // all child instructions, we scan through those
    // in the list that required second-phase processing,
    // and clone their decorations and/or children recursively.
    //
    for (auto pair : pairs)
    {
        auto oldChild = pair.oldInst;
        auto newChild = pair.newInst;

        _cloneInstDecorationsAndChildren(env, module, oldChild, newChild);
    }

    // For params, we can now clone their types since we have done cloning the entire block.
    for (auto pair : paramPairs)
    {
        auto oldParam = pair.oldInst;
        auto newParam = pair.newInst;

        auto oldType = oldParam->getFullType();
        auto newType = (IRType*)findCloneForOperand(env, oldType);
        newParam->setFullType(newType);
        newParam->sourceLoc = oldParam->sourceLoc;
    }

    if (auto oldAsSpec = as<IRSpecialize>(oldInst))
    {
        specializeLinkageDecoration(newInst, oldAsSpec, builder);
    }
}

// The public version of `cloneInstDecorationsAndChildren` is then
// just a wrapper over the internal one that sets up a temporary
// environment to use for the cloning process when `env->squashChildrenMapping` is false (default),
// so that we do not leave any lasting changes in the user-provided `env` unless the caller
// explicitly asks for it.
//
void cloneInstDecorationsAndChildren(
    IRCloneEnv* env,
    IRModule* module,
    IRInst* oldInst,
    IRInst* newInst)
{
    SLANG_ASSERT(module);
    SLANG_ASSERT(oldInst);
    SLANG_ASSERT(newInst);

    IRCloneEnv* subEnv = nullptr;
    IRCloneEnv subEnvStorage;
    if (env->squashChildrenMapping)
    {
        subEnv = env;
    }
    else
    {
        subEnv = &subEnvStorage;
        subEnv->parent = env;
    }
    _cloneInstDecorationsAndChildren(subEnv, module, oldInst, newInst);
}

// The convenience function `cloneInst` just sequences the
// operations that have already been defined.
//
IRInst* cloneInst(IRCloneEnv* env, IRBuilder* builder, IRInst* oldInst)
{
    SLANG_ASSERT(env);
    SLANG_ASSERT(builder);
    SLANG_ASSERT(oldInst);

    IRInst* newInst = nullptr;
    if (env->mapOldValToNew.tryGetValue(oldInst, newInst))
    {
        // In this case, somebody is trying to clone an
        // instruction that already had been cloned
        // (e.g., trying to clone a `param` in a function
        // body that had already been mapped to a specialization)
        // so we will make the operation safer and more
        // convenient by just returning the registered value.
        //
        // TODO: There might be cases where the client doesn't
        // want this convenience feature (because it could
        // accidentally mask a bug), so we should consider
        // having two versions of `cloneInst()` with one
        // explicitly not including this feature.
        //
        return newInst;
    }

    newInst = cloneInstAndOperands(env, builder, oldInst);

    env->mapOldValToNew.add(oldInst, newInst);

    // For hoistable insts, its possible that the cloned inst is the same
    // as the original inst.
    // Skip the decoration/children cloning in that case (which will end up
    // in an infinite loop)
    //
    if (newInst == oldInst)
        return newInst;

    cloneInstDecorationsAndChildren(env, builder->getModule(), oldInst, newInst);

    return newInst;
}

void cloneDecoration(
    IRCloneEnv* cloneEnv,
    IRDecoration* oldDecoration,
    IRInst* newParent,
    IRModule* module)
{
    IRBuilder builder(module);

    if (auto first = newParent->getFirstDecorationOrChild())
        builder.setInsertBefore(first);
    else
        builder.setInsertInto(newParent);

    IRCloneEnv env;
    env.parent = cloneEnv;
    cloneInst(&env, &builder, oldDecoration);
}

void cloneDecoration(IRDecoration* oldDecoration, IRInst* newParent)
{
    cloneDecoration(nullptr, oldDecoration, newParent, newParent->getModule());
}

bool IRSimpleSpecializationKey::operator==(IRSimpleSpecializationKey const& other) const
{
    auto valCount = vals.getCount();
    if (valCount != other.vals.getCount())
        return false;
    for (Index ii = 0; ii < valCount; ++ii)
    {
        if (vals[ii] != other.vals[ii])
            return false;
    }
    return true;
}

HashCode IRSimpleSpecializationKey::getHashCode() const
{
    auto valCount = vals.getCount();
    HashCode hash = Slang::getHashCode(valCount);
    for (Index ii = 0; ii < valCount; ++ii)
    {
        hash = combineHash(hash, Slang::getHashCode(vals[ii]));
    }
    return hash;
}


} // namespace Slang
