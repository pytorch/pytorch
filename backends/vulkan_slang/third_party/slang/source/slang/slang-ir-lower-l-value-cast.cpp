#include "slang-ir-lower-l-value-cast.h"

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

struct LValueCastLoweringContext
{
    void _addToWorkList(IRInst* inst)
    {
        if (!findOuterGeneric(inst) && !m_workList.contains(inst))
        {
            m_workList.add(inst);
        }
    }

    void _processInst(IRInst* inst)
    {
        switch (inst->getOp())
        {
        case kIROp_InOutImplicitCast:
        case kIROp_OutImplicitCast:
            _processLValueCast(inst);
            break;
        default:
            break;
        }
    }

    void processModule()
    {
        _addToWorkList(m_module->getModuleInst());

        while (m_workList.getCount() != 0)
        {
            IRInst* inst = m_workList.getLast();
            m_workList.removeLast();

            _processInst(inst);

            for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
            {
                _addToWorkList(child);
            }
        }
    }

    /// True if the conversion from a to b, can be achieved
    /// via a reinterpret cast/bitcast
    /// Only some targets can allow such conversions
    bool _canReinterpretCast(IRType* a, IRType* b)
    {
        auto ptrA = as<IRPtrTypeBase>(a);
        auto ptrB = as<IRPtrTypeBase>(b);

        // They must both be pointers...
        SLANG_ASSERT(ptrA && ptrB);

        a = ptrA->getValueType();
        b = ptrB->getValueType();

        if (a->m_op == b->m_op)
        {
            if (auto matA = as<IRMatrixType>(a))
            {
                auto matB = static_cast<IRMatrixType*>(b);

                if (getIntVal(matA->getColumnCount()) != getIntVal(matB->getColumnCount()))
                {
                    return false;
                }

                a = matA->getElementType();
                b = matB->getElementType();
            }
            else if (auto vecA = as<IRVectorType>(a))
            {
                auto vecB = static_cast<IRVectorType*>(b);

                if (getIntVal(vecA->getElementCount()) != getIntVal(vecB->getElementCount()))
                {
                    return false;
                }

                a = vecA->getElementType();
                b = vecB->getElementType();
            }
        }

        auto basicA = as<IRBasicType>(a);
        auto basicB = as<IRBasicType>(b);

        if (basicA && basicB)
        {
            auto baseA = basicA->getBaseType();
            auto baseB = basicB->getBaseType();

            const auto& infoA = BaseTypeInfo::getInfo(baseA);
            const auto& infoB = BaseTypeInfo::getInfo(baseB);

            // We allow reinterpret case for int type conversions of the same bit size for now
            if (infoA.sizeInBytes == infoB.sizeInBytes &&
                (infoA.flags & infoB.flags & BaseTypeInfo::Flag::Integer))
            {
                return true;
            }
        }

        return false;
    }

    /// True if for HLSL the cast can be removed entirely
    bool _canRemoveCastForHLSL(IRType* a, IRType* b)
    {
        // Currently _canReinterpret is exactly the same class of types that we can just ignore the
        // cast totally for HLSL If _canReinterpretCast changes, this will need to be updated
        return _canReinterpretCast(a, b);
    }

    void _processLValueCast(IRInst* castInst)
    {
        auto castOperand = castInst->getOperand(0);
        auto fromType = castOperand->getDataType();
        auto toType = castInst->getDataType();

        switch (m_intermediateSourceLanguage)
        {
        case SourceLanguage::HLSL:
            {
                // If the conversion can just be ignored for HLSL, just remove it
                if (_canRemoveCastForHLSL(fromType, toType))
                {
                    castInst->replaceUsesWith(castOperand);
                    castInst->removeAndDeallocate();
                    return;
                }
                break;
            }
        case SourceLanguage::C:
        case SourceLanguage::CPP:
        case SourceLanguage::CUDA:
            {
                // For languages with pointers, out parameter differences can *sometimes* just be
                // sidestepped with a reinterpret cast.
                if (_canReinterpretCast(fromType, toType))
                {
                    return;
                }
                break;
            }
        default:
            break;
        }

        // If we can't use the other mechanisms we are going to do a conversion
        // via a cast into a temporary of the approprite time before the useSite,
        // then immediately after converting back into the original location.
        //
        // With a special case for uses which are just out - where we don't need to
        // convert in.

        // Okay we are going to replace the implicit casts with temporaries around call sites/uses.
        List<IRUse*> useSites;
        for (auto use = castInst->firstUse; use; use = use->nextUse)
        {
            useSites.add(use);
        }

        // If there is a name hint on the source, we'll copy it over to the temporaries
        auto nameHintDecoration = castOperand->findDecoration<IRNameHintDecoration>();

        IRBuilder builder(m_module);

        IRType* toValueType = as<IRPtrType>(toType)->getValueType();
        IRType* fromValueType = as<IRPtrType>(fromType)->getValueType();

        for (auto useSite : useSites)
        {
            auto user = useSite->getUser();
            builder.setInsertBefore(user);
            auto tmpVar = builder.emitVar(toValueType);

            if (nameHintDecoration)
            {
                cloneDecoration(nameHintDecoration, tmpVar);
            }

            // If it's inout we convert via cast whats in the castOperand
            if (castInst->getOp() == kIROp_InOutImplicitCast && user->getOp() != kIROp_Store)
            {
                builder.emitStore(
                    tmpVar,
                    builder.emitCast(toValueType, builder.emitLoad(castOperand)));
            }

            // Convert the temporary back to the original location
            builder.setInsertAfter(user);
            builder.emitStore(
                castOperand,
                builder.emitCast(fromValueType, builder.emitLoad(tmpVar)));

            // Go through all of the operands of the use inst relacing, with the temporary
            builder.replaceOperand(useSite, tmpVar);
        }

        // When we are done we can destroy the inst
        castInst->removeAndDeallocate();
    }

    LValueCastLoweringContext(TargetProgram* target, IRModule* module)
        : m_targetProgram(target), m_module(module)
    {
        m_intermediateSourceLanguage = getIntermediateSourceLanguageForTarget(target);
    }

    // The intermediate source language used to produce code for the target.
    // If no intermediate source language is used will be SourceLanguage::Unknown.
    SourceLanguage m_intermediateSourceLanguage = SourceLanguage::Unknown;
    TargetProgram* m_targetProgram;
    IRModule* m_module;
    OrderedHashSet<IRInst*> m_workList;
};

void lowerLValueCast(TargetProgram* target, IRModule* module)
{
    LValueCastLoweringContext context(target, module);
    context.processModule();
}

} // namespace Slang
