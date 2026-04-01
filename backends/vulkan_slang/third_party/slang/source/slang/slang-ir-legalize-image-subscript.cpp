#include "slang-ir-legalize-image-subscript.h"

#include "slang-ir-clone.h"
#include "slang-ir-insts.h"
#include "slang-ir-legalize-varying-params.h"
#include "slang-ir-specialize-address-space.h"
#include "slang-ir-util.h"
#include "slang-ir.h"
#include "slang-parameter-binding.h"

namespace Slang
{
void legalizeStore(
    TargetRequest* target,
    IRBuilder& builder,
    IRInst* storeInst,
    DiagnosticSink* sink)
{
    SLANG_ASSERT(storeInst);

    builder.setInsertBefore(storeInst);
    auto getElementPtr = as<IRGetElementPtr>(storeInst->getOperand(0));
    IRImageSubscript* imageSubscript = as<IRImageSubscript>(getRootAddr(storeInst->getOperand(0)));
    SLANG_ASSERT(imageSubscript);
    SLANG_ASSERT(imageSubscript->getImage());
    IRTextureType* textureType = as<IRTextureType>(imageSubscript->getImage()->getFullType());
    SLANG_ASSERT(textureType);
    auto imageElementType = cast<IRPtrTypeBase>(imageSubscript->getDataType())->getValueType();
    auto vectorBaseType = getIRVectorBaseType(imageElementType);
    IRType* vector4Type = builder.getVectorType(vectorBaseType, 4);
    IRType* coordType = imageSubscript->getCoord()->getDataType();
    int coordVectorSize = getIRVectorElementSize(coordType);

    bool seperateArrayCoord =
        (isMetalTarget(target) && textureType->isArray());     // seperate array param
    bool seperateSampleCoord = (textureType->isMultisample()); // seperate sample param

    if (seperateSampleCoord && isMetalTarget(target))
        sink->diagnose(
            imageSubscript->getImage(),
            Diagnostics::multiSampledTextureDoesNotAllowWrites,
            target->getTarget());

    IRType* indexingType = builder.getIntType();
    if (isMetalTarget(target))
        indexingType = builder.getUIntType();

    if (coordVectorSize != 1)
    {
        coordType = builder.getVectorType(
            indexingType,
            builder.getIntValue(builder.getIntType(), coordVectorSize));
    }
    else
    {
        coordType = indexingType;
    }

    auto legalizedCoord = imageSubscript->getCoord();
    if (coordType != imageSubscript->getCoord()->getDataType())
    {
        legalizedCoord = builder.emitCast(coordType, legalizedCoord);
    }

    const Index kCoordParamIndex = 1;
    const Index kValueParamIndex = 2;

    ShortList<IRInst*> loadParams;
    loadParams.reserveOverflowBuffer(4);
    loadParams.add(imageSubscript->getImage()); // image
    loadParams.add(legalizedCoord);             // coord

    ShortList<IRInst*> storeParams;
    storeParams.reserveOverflowBuffer(5);
    storeParams.add(imageSubscript->getImage()); // image
    storeParams.add(legalizedCoord);             // coord
    storeParams.add(nullptr);                    // value

    if (seperateArrayCoord)
    {

        UInt paramIndexToFetch = coordVectorSize - 1;

        auto seperatedParam =
            builder.emitSwizzle(indexingType, legalizedCoord, 1, &paramIndexToFetch);
        loadParams.add(seperatedParam);
        storeParams.add(seperatedParam);

        coordVectorSize -= 1;
        ShortList<UInt> paramToFetch;
        paramToFetch.reserveOverflowBuffer(coordVectorSize);
        for (int i = 0; i < coordVectorSize; i++)
        {
            paramToFetch.add(i);
        }
        auto newCoord = builder.emitSwizzle(
            builder.getVectorType(
                indexingType,
                builder.getIntValue(builder.getIntType(), coordVectorSize)),
            legalizedCoord,
            coordVectorSize,
            paramToFetch.getArrayView().getBuffer());
        storeParams[kCoordParamIndex] = newCoord;
        loadParams[kCoordParamIndex] = newCoord;
    }
    if (seperateSampleCoord)
    {
        loadParams.add(imageSubscript->getSampleCoord());
        storeParams.add(imageSubscript->getSampleCoord());
    }

    IRInst* legalizedStore = storeInst->getOperand(1);
    switch (storeInst->getOp())
    {
    case kIROp_Store:
        {
            IRInst* newValue = nullptr;
            if (getElementPtr)
            {
                auto originalValue = builder.emitImageLoad(vector4Type, loadParams);
                auto index = getElementPtr->getIndex();
                newValue =
                    builder.emitSwizzleSet(vector4Type, originalValue, legalizedStore, 1, &index);
            }
            else
            {
                newValue = legalizedStore;
                if (getIRVectorElementSize(imageElementType) != 4)
                {
                    newValue = builder.emitVectorReshape(
                        builder.getVectorType(
                            vectorBaseType,
                            builder.getIntValue(builder.getIntType(), 4)),
                        newValue);
                }
            }

            storeParams[kValueParamIndex] = newValue;
            auto imageStore = builder.emitImageStore(builder.getVoidType(), storeParams);
            storeInst->replaceUsesWith(imageStore);
            storeInst->removeAndDeallocate();
            if (!imageSubscript->hasUses())
            {
                imageSubscript->removeAndDeallocate();
            }
        }
        break;
    case kIROp_SwizzledStore:
        {
            auto swizzledStore = cast<IRSwizzledStore>(storeInst);
            // Here we assume the imageElementType is already lowered into float4/uint4 types from
            // any user-defined type.
            assert(imageElementType->getOp() == kIROp_VectorType);
            auto originalValue = builder.emitImageLoad(vector4Type, loadParams);
            Array<IRInst*, 4> indices;
            for (UInt i = 0; i < swizzledStore->getElementCount(); i++)
            {
                indices.add(swizzledStore->getElementIndex(i));
            }
            auto newValue = builder.emitSwizzleSet(
                vector4Type,
                originalValue,
                swizzledStore->getSource(),
                swizzledStore->getElementCount(),
                indices.getBuffer());
            storeParams[kValueParamIndex] = newValue;
            auto imageStore = builder.emitImageStore(builder.getVoidType(), storeParams);
            storeInst->replaceUsesWith(imageStore);
            storeInst->removeAndDeallocate();
            if (!imageSubscript->hasUses())
            {
                imageSubscript->removeAndDeallocate();
            }
        }
        break;
    default:
        break;
    }
}
void legalizeImageSubscript(TargetRequest* target, IRModule* module, DiagnosticSink* sink)
{
    IRBuilder builder(module);
    for (auto globalInst : module->getModuleInst()->getChildren())
    {
        auto func = as<IRFunc>(globalInst);
        if (!func)
            continue;
        for (auto block : func->getBlocks())
        {
            auto inst = block->getFirstInst();
            IRInst* next;
            for (; inst; inst = next)
            {
                next = inst->getNextInst();
                switch (inst->getOp())
                {
                case kIROp_Store:
                case kIROp_SwizzledStore:
                    if (as<IRImageSubscript>(getRootAddr(inst->getOperand(0))))
                        legalizeStore(target, builder, inst, sink);
                    continue;
                }
            }
        }
    }
}
} // namespace Slang
