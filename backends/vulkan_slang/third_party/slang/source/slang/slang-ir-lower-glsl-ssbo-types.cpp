#include "slang-ir-lower-glsl-ssbo-types.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
template<typename F>
static void overAllInsts(IRModule* module, F f)
{
    InstWorkList workList{module};
    InstHashSet workListSet{module};

    workList.add(module->getModuleInst());
    workListSet.add(module->getModuleInst());
    while (workList.getCount())
    {
        const auto inst = workList.getLast();
        workList.removeLast();
        workListSet.remove(inst);

        f(inst);

        for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
        {
            if (workListSet.contains(child))
                continue;
            workList.add(child);
            workListSet.add(child);
        }
    }
}

struct ArrayLikeSSBOInfo
{
    IRType* backingStruct;
    IRType* elementType;
    IRStructKey* elementsKey;
};

struct StructLikeSSBOInfo
{
    IRType* backingStruct;
};

struct SSBOIndexInfo
{
    IRInst* ssboVar;
    IRInst* index;
};

static void lowerArrayLikeSSBOs(IRModule* module)
{
    // Here we will:
    // - Identify SSBO types which comprise a single array field with
    //   element type T
    // - Create a (RW)StructuredBuffer<T> type
    // - Replace all uses of the SSBO with the StructuredBuffer type

    Dictionary<IRGLSLShaderStorageBufferType*, ArrayLikeSSBOInfo> arrayLikeSSBOTypes;
    overAllInsts(
        module,
        [&](IRInst* inst)
        {
            if (const auto ssbo = as<IRGLSLShaderStorageBufferType>(inst))
            {
                if (const auto backingStruct = as<IRStructType>(ssbo->getElementType()))
                {
                    auto members = backingStruct->getFields();
                    if (members.getFirst() != members.getLast())
                        return;
                    const auto onlyMember = members.getFirst();
                    if (!onlyMember)
                        return;
                    const auto onlyArrayMember = as<IRUnsizedArrayType>(onlyMember->getFieldType());
                    if (!onlyArrayMember)
                        return;
                    const auto elemType = onlyArrayMember->getElementType();
                    arrayLikeSSBOTypes.add(ssbo, {backingStruct, elemType, onlyMember->getKey()});
                }
            }
        });

    IRBuilder builder{module};
    for (const auto& [ssbo, info] : arrayLikeSSBOTypes)
    {
        builder.setInsertAfter(ssbo);
        IRInst* operands[2] = {info.elementType, ssbo->getDataLayout()};
        const auto sb = builder.getType(kIROp_HLSLRWStructuredBufferType, 2, operands);
        ssbo->replaceUsesWith(sb);
        ssbo->removeAndDeallocate();

        Dictionary<IRInst*, SSBOIndexInfo> indexes;
        traverseUses(
            info.elementsKey,
            [&](IRUse* use)
            {
                if (const auto fieldAddress = as<IRFieldAddress>(use->getUser()))
                {
                    traverseUses(
                        use->user,
                        [&](IRUse* arrayUse)
                        {
                            if (const auto gep = as<IRGetElementPtr>(arrayUse->user))
                            {
                                SLANG_ASSERT(gep->getBase() == use->user);
                                indexes.add(gep, {fieldAddress->getBase(), gep->getIndex()});
                            }
                            else
                            {
                                SLANG_UNIMPLEMENTED_X("Unhandled use of array-like SSBO");
                            }
                        });
                }
                else if (as<IRStructField>(use->getUser()))
                {
                    // We expect and can ignore the use when declaring a struct field
                }
                else
                {
                    SLANG_UNIMPLEMENTED_X("Unhandled use of array-like SSBO");
                }
            });
        for (const auto& [elemPtr, index] : indexes)
        {
            builder.setInsertBefore(elemPtr);
            const auto sbp =
                builder.emitRWStructuredBufferGetElementPtr(index.ssboVar, index.index);
            elemPtr->replaceUsesWith(sbp);
            elemPtr->removeAndDeallocate();
        }
    }
}

static void lowerStructLikeSSBOs(IRModule* module)
{
    // Here we will:
    // - Identify SSBO types without an unsized array member in the backing
    //   struct, T
    // - Create a (RW)StructuredBuffer<T> type
    // - Replace all uses of the SSBO with the StructuredBuffer type

    Dictionary<IRGLSLShaderStorageBufferType*, StructLikeSSBOInfo> structLikeSSBOTypes;
    overAllInsts(
        module,
        [&](IRInst* inst)
        {
            if (const auto ssbo = as<IRGLSLShaderStorageBufferType>(inst))
            {
                // Storage buffers are always backed by a struct
                if (const auto backingStruct = as<IRStructType>(ssbo->getElementType()))
                {
                    auto members = backingStruct->getFields();
                    const auto lastMember = members.getLast();
                    if (lastMember && as<IRUnsizedArrayType>(lastMember->getFieldType()))
                        return;
                    structLikeSSBOTypes.add(ssbo, {backingStruct});
                }
            }
        });

    // Collect the uses of these ssbos
    InstHashSet ssboUses(module);
    overAllInsts(
        module,
        [&](IRInst* inst)
        {
            StructLikeSSBOInfo slsi;
            if (const auto ssboType = as<IRGLSLShaderStorageBufferType>(inst->getDataType()))
                if (structLikeSSBOTypes.tryGetValue(ssboType, slsi))
                    ssboUses.add(inst);
        });

    IRBuilder builder{module};
    for (const auto& [ssbo, info] : structLikeSSBOTypes)
    {
        builder.setInsertAfter(ssbo);
        IRInst* operands = info.backingStruct;
        const auto sb = builder.getType(kIROp_HLSLRWStructuredBufferType, 1, &operands);

        ssbo->replaceUsesWith(sb);
        ssbo->removeAndDeallocate();
    }

    for (const auto& var : *ssboUses.set)
    {
        traverseUses(
            var,
            [&](IRUse* use)
            {
                // We only want to insert this access into blocks, anything
                // else we assume is just using the identity of the ssbo (for
                // instance IRStructFieldLayoutAttr)
                if (!as<IRBlock>(use->getUser()->getParent()))
                    return;
                builder.setInsertBefore(use->getUser());
                const auto sbp = builder.emitRWStructuredBufferGetElementPtr(
                    var,
                    builder.getIntValue(builder.getIntType(), 0));
                use->set(sbp);
            });
    }
}

static void lowerSSBOsToPointers(IRModule* module)
{
    IRBuilder builder{module};
    overAllInsts(
        module,
        [&](IRInst* inst)
        {
            if (const auto ssbo = as<IRGLSLShaderStorageBufferType>(inst))
            {
                builder.setInsertAfter(ssbo);
                const auto ptr = builder.getPtrType(ssbo->getElementType());
                ssbo->replaceUsesWith(ptr);
                ssbo->removeAndDeallocate();
            }
        });
}

static void diagnoseRemainingSSBOs(IRModule* module, DiagnosticSink* sink)
{
    overAllInsts(
        module,
        [&](IRInst* inst)
        {
            if (const auto ssbo = as<IRGLSLShaderStorageBufferType>(inst))
            {
                sink->diagnose(ssbo, Diagnostics::unhandledGLSLSSBOType);
            }
        });
}

void lowerGLSLShaderStorageBufferObjectsToStructuredBuffers(IRModule* module, DiagnosticSink* sink)
{
    lowerArrayLikeSSBOs(module);
    lowerStructLikeSSBOs(module);
    diagnoseRemainingSSBOs(module, sink);
}

void lowerGLSLShaderStorageBufferObjectsToPointers(IRModule* module, DiagnosticSink* sink)
{
    lowerSSBOsToPointers(module);
    diagnoseRemainingSSBOs(module, sink);
}
} // namespace Slang
