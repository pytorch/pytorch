#include "slang-ir-autodiff-pairs.h"

namespace Slang
{

struct DiffPairLoweringPass : InstPassBase
{
    DiffPairLoweringPass(AutoDiffSharedContext* context)
        : InstPassBase(context->moduleInst->getModule()), pairBuilderStorage(context)
    {
        pairBuilder = &pairBuilderStorage;
    }

    IRInst* lowerPairType(IRBuilder* builder, IRType* pairType)
    {
        auto loweredPairType = pairBuilder->lowerDiffPairType(builder, pairType);
        return loweredPairType;
    }

    IRInst* lowerMakePair(IRBuilder* builder, IRInst* inst)
    {
        if (auto makePairInst = as<IRMakeDifferentialPairBase>(inst))
        {
            auto pairType = as<IRDifferentialPairTypeBase>(makePairInst->getDataType());
            builder->setInsertBefore(makePairInst);
            if (auto loweredPairType = (IRType*)lowerPairType(builder, pairType))
            {
                if (isRuntimeType(pairType->getValueType()))
                {
                    auto result = pairBuilder->emitExistentialMakePair(
                        builder,
                        loweredPairType,
                        makePairInst->getPrimalValue(),
                        makePairInst->getDifferentialValue());

                    makePairInst->replaceUsesWith(result);
                    makePairInst->removeAndDeallocate();
                    return result;
                }
                else if (auto typePack = as<IRTypePack>(pairType->getValueType()))
                {
                    // TODO: Do we need to flatten the packs here?

                    // If the type is a type pack, then the value must be in
                    // MakePair(MakeValuePack(p_0, p_1, ...), MakeValuePack(d_0, d_1, ...)) form
                    // Convert it to MakeValuePack(MakePair(p_0, d_0), MakePair(p_1, d_1), ...)
                    // and lower each MakePair.
                    //

                    // Primal pack
                    auto primalValue = as<IRMakeValuePack>(makePairInst->getPrimalValue());
                    SLANG_ASSERT(primalValue);

                    // Differential pack
                    auto diffValue = as<IRMakeValuePack>(makePairInst->getDifferentialValue());
                    SLANG_ASSERT(diffValue);

                    // Expect the lowered pair type to be a type pack of pair types.
                    SLANG_ASSERT(as<IRTypePack>(loweredPairType));

                    List<IRInst*> newValues;
                    for (UInt i = 0; i < typePack->getOperandCount(); i++)
                    {
                        auto primalElement = primalValue->getOperand(i);
                        auto diffElement = diffValue->getOperand(i);

                        auto loweredElementPairType = (IRType*)loweredPairType->getOperand(i);

                        IRInst* operands[] = {primalElement, diffElement};

                        auto loweredMakePair =
                            builder->emitMakeStruct((IRType*)loweredElementPairType, 2, operands);

                        newValues.add(loweredMakePair);
                    }

                    auto newPack = builder->emitMakeValuePack(
                        loweredPairType,
                        newValues.getCount(),
                        newValues.getBuffer());

                    makePairInst->replaceUsesWith(newPack);
                    makePairInst->removeAndDeallocate();
                    return newPack;
                }
                else
                {
                    IRInst* result = nullptr;

                    IRInst* operands[2] = {
                        makePairInst->getPrimalValue(),
                        makePairInst->getDifferentialValue()};
                    result = builder->emitMakeStruct((IRType*)(loweredPairType), 2, operands);

                    makePairInst->replaceUsesWith(result);
                    makePairInst->removeAndDeallocate();
                    return result;
                }
            }
        }

        return nullptr;
    }

    IRInst* lowerPairAccess(IRBuilder* builder, IRInst* inst)
    {
        if (auto getDiffInst = as<IRDifferentialPairGetDifferentialBase>(inst))
        {
            auto pairType = getDiffInst->getBase()->getDataType();
            if (auto pairPtrType = as<IRPtrTypeBase>(pairType))
            {
                pairType = pairPtrType->getValueType();
            }

            builder->setInsertBefore(getDiffInst);
            if (auto loweredType = lowerPairType(builder, pairType))
            {
                IRInst* diffFieldExtract = nullptr;
                diffFieldExtract = pairBuilder->emitDiffFieldAccess(
                    builder,
                    (IRType*)loweredType,
                    getDiffInst->getBase());
                getDiffInst->replaceUsesWith(diffFieldExtract);
                getDiffInst->removeAndDeallocate();
                return diffFieldExtract;
            }
        }
        else if (auto getPrimalInst = as<IRDifferentialPairGetPrimalBase>(inst))
        {
            auto pairType = getPrimalInst->getBase()->getDataType();
            if (auto pairPtrType = as<IRPtrTypeBase>(pairType))
            {
                pairType = pairPtrType->getValueType();
            }

            builder->setInsertBefore(getPrimalInst);
            if (auto loweredType = lowerPairType(builder, pairType))
            {
                IRInst* primalFieldExtract = nullptr;
                primalFieldExtract = pairBuilder->emitPrimalFieldAccess(
                    builder,
                    (IRType*)loweredType,
                    getPrimalInst->getBase());
                getPrimalInst->replaceUsesWith(primalFieldExtract);
                getPrimalInst->removeAndDeallocate();
                return primalFieldExtract;
            }
        }

        return nullptr;
    }

    bool processInstWithChildren(IRBuilder* builder, IRInst* instWithChildren)
    {
        bool modified = false;

        processAllInsts(
            [&](IRInst* inst)
            {
                // Make sure the builder is at the right level.
                builder->setInsertInto(instWithChildren);

                switch (inst->getOp())
                {
                case kIROp_DifferentialPairGetDifferential:
                case kIROp_DifferentialPairGetPrimal:
                case kIROp_DifferentialPairGetDifferentialUserCode:
                case kIROp_DifferentialPairGetPrimalUserCode:
                case kIROp_DifferentialPtrPairGetDifferential:
                case kIROp_DifferentialPtrPairGetPrimal:
                    lowerPairAccess(builder, inst);
                    break;

                case kIROp_MakeDifferentialPairUserCode:
                case kIROp_MakeDifferentialPtrPair:
                    lowerMakePair(builder, inst);
                    break;

                default:
                    break;
                }
            });

        OrderedDictionary<IRInst*, IRInst*> pendingReplacements;
        processAllInsts(
            [&](IRInst* inst)
            {
                if (auto pairType = as<IRDifferentialPairTypeBase>(inst))
                {
                    if (auto loweredType = lowerPairType(builder, pairType))
                    {
                        pendingReplacements.add(pairType, loweredType);
                        modified = true;
                    }
                }
            });
        for (auto replacement : pendingReplacements)
        {
            replacement.key->replaceUsesWith(replacement.value);
            replacement.key->removeAndDeallocate();
        }

        return modified;
    }

    bool processModule()
    {
        IRBuilder builder(module);
        return processInstWithChildren(&builder, module->getModuleInst());
    }

private:
    DifferentialPairTypeBuilder* pairBuilder;

    DifferentialPairTypeBuilder pairBuilderStorage;
};

bool processPairTypes(AutoDiffSharedContext* context)
{
    DiffPairLoweringPass pairLoweringPass(context);
    return pairLoweringPass.processModule();
}

struct DifferentialPairUserCodeTranscribePass : public InstPassBase
{
    DifferentialPairUserCodeTranscribePass(IRModule* module)
        : InstPassBase(module)
    {
    }

    IRInst* rewritePairType(IRBuilder* builder, IRType* pairType)
    {
        builder->setInsertBefore(pairType);
        auto originalPairType = as<IRDifferentialPairType>(pairType);
        return builder->getDifferentialPairUserCodeType(
            originalPairType->getValueType(),
            originalPairType->getWitness());
    }

    IRInst* rewriteMakePair(IRBuilder* builder, IRMakeDifferentialPair* inst)
    {
        auto pairType = as<IRDifferentialPairType>(inst->getFullType());
        builder->setInsertBefore(inst);
        auto newInst = builder->emitMakeDifferentialPairUserCode(
            (IRType*)pairType,
            inst->getPrimalValue(),
            inst->getDifferentialValue());
        inst->replaceUsesWith(newInst);
        inst->removeAndDeallocate();
        return newInst;
    }

    IRInst* rewritePairAccess(IRBuilder* builder, IRInst* inst)
    {
        if (auto getDiffInst = as<IRDifferentialPairGetDifferential>(inst))
        {
            builder->setInsertBefore(inst);

            auto newInst = builder->emitDifferentialPairGetDifferentialUserCode(
                (IRType*)inst->getFullType(),
                getDiffInst->getBase());
            inst->replaceUsesWith(newInst);
            inst->removeAndDeallocate();
        }
        else if (auto getPrimalInst = as<IRDifferentialPairGetPrimal>(inst))
        {
            builder->setInsertBefore(inst);
            auto newInst = builder->emitDifferentialPairGetPrimalUserCode(getPrimalInst->getBase());
            inst->replaceUsesWith(newInst);
            inst->removeAndDeallocate();
        }
        return inst;
    }

    bool processInstWithChildren(IRBuilder* builder, IRInst* instWithChildren)
    {
        SLANG_UNUSED(instWithChildren);

        bool modified = false;

        processAllInsts(
            [&](IRInst* inst)
            {
                switch (inst->getOp())
                {
                case kIROp_DifferentialPairGetDifferential:
                case kIROp_DifferentialPairGetPrimal:
                    rewritePairAccess(builder, inst);
                    break;

                case kIROp_MakeDifferentialPair:
                    rewriteMakePair(builder, as<IRMakeDifferentialPair>(inst));
                    break;

                default:
                    break;
                }
            });

        OrderedDictionary<IRInst*, IRInst*> pendingReplacements;
        processInstsOfType<IRDifferentialPairType>(
            kIROp_DifferentialPairType,
            [&](IRDifferentialPairType* inst)
            {
                if (auto loweredType = rewritePairType(builder, inst))
                {
                    pendingReplacements.add(inst, loweredType);
                    modified = true;
                }
            });
        for (auto replacement : pendingReplacements)
        {
            replacement.key->replaceUsesWith(replacement.value);
            replacement.key->removeAndDeallocate();
        }

        return modified;
    }

    bool processModule()
    {
        IRBuilder builder(module);
        return processInstWithChildren(&builder, module->getModuleInst());
    }
};

void rewriteDifferentialPairToUserCode(IRModule* module)
{
    DifferentialPairUserCodeTranscribePass pairRewritePass(module);
    pairRewritePass.processModule();
}

} // namespace Slang
