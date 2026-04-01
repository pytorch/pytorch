#include "slang-ir-legalize-global-values.h"

#include "slang-ir-clone.h"
#include "slang-ir-util.h"

namespace Slang
{

void GlobalInstInliningContextGeneric::inlineGlobalValuesAndRemoveIfUnused(IRModule* module)
{
    List<IRUse*> globalInstUsesToInline;

    for (auto globalInst : module->getGlobalInsts())
    {
        if (isInlinableGlobalInst(globalInst))
        {
            for (auto use = globalInst->firstUse; use; use = use->nextUse)
            {
                if (getParentFunc(use->getUser()) != nullptr)
                    globalInstUsesToInline.add(use);
            }
        }
    }

    HashSet<IRInst*> globalInstsToConsiderDeleting;
    for (auto use : globalInstUsesToInline)
    {
        auto user = use->getUser();
        IRBuilder builder(user);
        builder.setInsertBefore(getOutsideASM(user));
        IRCloneEnv cloneEnv;
        auto val = maybeInlineGlobalValue(builder, use->getUser(), use->get(), cloneEnv);
        if (val != use->get())
        {
            // Since certain globals that appear in the IR are considered illegal for all targets,
            // e.g. calls to functions, we delete the globals we've inlined.
            // Note that the inlining is done such that none of the descendants of the global will
            // have any uses either.
            globalInstsToConsiderDeleting.add(use->usedValue);

            builder.replaceOperand(use, val);
        }
    }

    for (auto globalInst : globalInstsToConsiderDeleting)
    {
        if (!globalInst->hasUses())
            globalInst->removeAndDeallocate();
    }
}

bool GlobalInstInliningContextGeneric::isLegalGlobalInst(IRInst* inst)
{
    switch (inst->getOp())
    {
    case kIROp_MakeStruct:
    case kIROp_MakeArray:
    case kIROp_MakeArrayFromElement:
    case kIROp_MakeVector:
    case kIROp_MakeMatrix:
    case kIROp_MakeMatrixFromScalar:
    case kIROp_MakeVectorFromScalar:
        return true;
    default:
        if (as<IRConstant>(inst))
            return true;
        if (isLegalGlobalInstForTarget(inst))
            return true;
        return false;
    }
}

bool GlobalInstInliningContextGeneric::isInlinableGlobalInst(IRInst* inst)
{
    switch (inst->getOp())
    {
    case kIROp_Add:
    case kIROp_Sub:
    case kIROp_Mul:
    case kIROp_FRem:
    case kIROp_IRem:
    case kIROp_Lsh:
    case kIROp_Rsh:
    case kIROp_And:
    case kIROp_Or:
    case kIROp_Not:
    case kIROp_Neg:
    case kIROp_Div:
    case kIROp_FieldExtract:
    case kIROp_FieldAddress:
    case kIROp_GetElement:
    case kIROp_GetElementPtr:
    case kIROp_GetOffsetPtr:
    case kIROp_UpdateElement:
    case kIROp_MakeTuple:
    case kIROp_GetTupleElement:
    case kIROp_MakeStruct:
    case kIROp_MakeArray:
    case kIROp_MakeArrayFromElement:
    case kIROp_MakeVector:
    case kIROp_MakeMatrix:
    case kIROp_MakeMatrixFromScalar:
    case kIROp_MakeVectorFromScalar:
    case kIROp_swizzle:
    case kIROp_swizzleSet:
    case kIROp_MatrixReshape:
    case kIROp_MakeString:
    case kIROp_MakeResultError:
    case kIROp_MakeResultValue:
    case kIROp_GetResultError:
    case kIROp_GetResultValue:
    case kIROp_CastFloatToInt:
    case kIROp_CastIntToFloat:
    case kIROp_CastIntToPtr:
    case kIROp_PtrCast:
    case kIROp_CastPtrToBool:
    case kIROp_CastPtrToInt:
    case kIROp_BitAnd:
    case kIROp_BitNot:
    case kIROp_BitOr:
    case kIROp_BitXor:
    case kIROp_BitCast:
    case kIROp_IntCast:
    case kIROp_FloatCast:
    case kIROp_Greater:
    case kIROp_Less:
    case kIROp_Geq:
    case kIROp_Leq:
    case kIROp_Neq:
    case kIROp_Eql:
    case kIROp_Call:
        return true;
    default:
        if (isInlinableGlobalInstForTarget(inst))
            return true;
        return false;
    }
}

bool GlobalInstInliningContextGeneric::shouldInlineInstImpl(IRInst* inst)
{
    // If 'inst' has an ancestor that is currently being inlined, then we
    // better inline it since we'll be removing the ancestor.
    bool ancestorShouldBeInlined = false;
    for (IRInst* ancestor = inst->parent; ancestor != nullptr; ancestor = ancestor->parent)
        if (m_mapGlobalInstToShouldInline.tryGetValue(inst, ancestorShouldBeInlined) &&
            ancestorShouldBeInlined)
            return true;

    if (!isInlinableGlobalInst(inst))
        return false;
    if (isLegalGlobalInst(inst))
    {
        for (UInt i = 0; i < inst->getOperandCount(); i++)
            if (shouldInlineInst(inst->getOperand(i)))
                return true;
        return false;
    }
    return true;
}

bool GlobalInstInliningContextGeneric::shouldInlineInst(IRInst* inst)
{
    bool result = false;
    if (m_mapGlobalInstToShouldInline.tryGetValue(inst, result))
        return result;
    result = shouldInlineInstImpl(inst);
    m_mapGlobalInstToShouldInline[inst] = result;
    return result;
}

IRInst* GlobalInstInliningContextGeneric::inlineInst(
    IRBuilder& builder,
    IRCloneEnv& cloneEnv,
    IRInst* inst)
{
    // We rely on this dictionary in order to force inlining of any nodes with that should be
    // inlined
    SLANG_ASSERT(m_mapGlobalInstToShouldInline[inst]);

    IRInst* result;
    if (cloneEnv.mapOldValToNew.tryGetValue(inst, result))
        return result;

    for (UInt i = 0; i < inst->getOperandCount(); i++)
    {
        auto operand = inst->getOperand(i);
        IRBuilder operandBuilder(builder);
        operandBuilder.setInsertBefore(getOutsideASM(builder.getInsertLoc().getInst()));
        maybeInlineGlobalValue(operandBuilder, inst, operand, cloneEnv);
    }
    result = cloneInstAndOperands(&cloneEnv, &builder, inst);
    cloneEnv.mapOldValToNew[inst] = result;
    IRBuilder subBuilder(builder);
    subBuilder.setInsertInto(result);
    for (auto child : inst->getDecorations())
    {
        cloneInst(&cloneEnv, &subBuilder, child);
    }
    for (auto child : inst->getChildren())
    {
        m_mapGlobalInstToShouldInline[child] = true;
        inlineInst(subBuilder, cloneEnv, child);
    }
    return result;
}

IRInst* GlobalInstInliningContextGeneric::maybeInlineGlobalValue(
    IRBuilder& builder,
    IRInst* user,
    IRInst* inst,
    IRCloneEnv& cloneEnv)
{
    if (!shouldInlineInst(inst))
    {
        switch (inst->getOp())
        {
        case kIROp_Func:
        case kIROp_Specialize:
        case kIROp_Generic:
        case kIROp_LookupWitness:
            return inst;
        }
        if (as<IRType>(inst))
            return inst;
        if (!wrapReferences)
            return inst;

        // If we encounter a global value that shouldn't be inlined, e.g. a const literal,
        // we should insert a GlobalValueRef() inst to wrap around it, so all the dependent
        // uses can be pinned to the function body.
        auto result = inst;
        bool shouldWrapGlobalRef = true;
        if (!isLegalGlobalInst(user) && !getIROpInfo(user->getOp()).isHoistable())
            shouldWrapGlobalRef = false;
        else if (shouldBeInlinedForTarget(user))
            shouldWrapGlobalRef = false;
        if (shouldWrapGlobalRef)
            result = builder.emitGlobalValueRef(inst);
        cloneEnv.mapOldValToNew[inst] = result;
        return result;
    }

    // If the global value is inlinable, we make all its operands avaialble locally, and
    // then copy it to the local scope.
    return inlineInst(builder, cloneEnv, inst);
}

struct GlobalInstLegalizationInliningContext : public GlobalInstInliningContextGeneric
{
    static bool isSimpleConstantType(IRType* type)
    {
        for (;;)
        {
            if (!type)
                return true;
            if (as<IRBasicType>(type))
                return true;
            if (as<IRVectorType>(type))
                return true;
            if (as<IRMatrixType>(type))
                return true;
            if (auto arrayType = as<IRArrayTypeBase>(type))
            {
                type = arrayType->getElementType();
                continue;
            }
            return false;
        }
    }
    bool isLegalGlobalInstForTarget(IRInst* inst) override
    {
        auto type = inst->getDataType();
        return isSimpleConstantType(type);
    }

    bool isInlinableGlobalInstForTarget(IRInst* /* inst */) override { return false; }

    bool shouldBeInlinedForTarget(IRInst* /* user */) override { return false; }

    IRInst* getOutsideASM(IRInst* beforeInst) override { return beforeInst; }
};

void inlineGlobalConstantsForLegalization(IRModule* module)
{
    GlobalInstLegalizationInliningContext context;

    context.wrapReferences = false;
    context.inlineGlobalValuesAndRemoveIfUnused(module);
}

} // namespace Slang
