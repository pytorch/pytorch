// slang-ir-spirv-legalize.cpp
#include "slang-ir-spirv-legalize.h"

#include "slang-emit-base.h"
#include "slang-ir-call-graph.h"
#include "slang-ir-clone.h"
#include "slang-ir-composite-reg-to-mem.h"
#include "slang-ir-dce.h"
#include "slang-ir-dominators.h"
#include "slang-ir-float-non-uniform-resource-index.h"
#include "slang-ir-glsl-legalize.h"
#include "slang-ir-inline.h"
#include "slang-ir-insts.h"
#include "slang-ir-layout.h"
#include "slang-ir-legalize-global-values.h"
#include "slang-ir-legalize-mesh-outputs.h"
#include "slang-ir-loop-unroll.h"
#include "slang-ir-lower-buffer-element-type.h"
#include "slang-ir-peephole.h"
#include "slang-ir-redundancy-removal.h"
#include "slang-ir-sccp.h"
#include "slang-ir-simplify-cfg.h"
#include "slang-ir-specialize-address-space.h"
#include "slang-ir-util.h"
#include "slang-ir-validate.h"
#include "slang-ir.h"
#include "slang-legalize-types.h"

namespace Slang
{

//
// Legalization of IR for direct SPIRV emit.
//

struct SPIRVLegalizationContext : public SourceEmitterBase
{
    SPIRVEmitSharedContext* m_sharedContext;

    IRModule* m_module;

    DiagnosticSink* m_sink;

    struct LoweredStructuredBufferTypeInfo
    {
        IRType* structType;
        IRStructKey* arrayKey;
        IRArrayTypeBase* runtimeArrayType;
    };
    Dictionary<IRType*, LoweredStructuredBufferTypeInfo> m_loweredStructuredBufferTypes;

    IRInst* lowerTextureFootprintType(IRInst* footprintType)
    {
        // Lowers `IRTextureFootprintType` to a struct with the following definition:
        /*
            ```
            struct __SpvTextureFootprintData<let ND : int>
            {
                bool isSingleLevel;
                vector<uint, ND> anchor;
                vector<uint, ND> offset;
                uint2 mask;
                uint lod;
                uint granularity;
            }
            ```
        */

        auto ND = footprintType->getOperand(0);

        IRBuilder builder(footprintType);
        builder.setInsertBefore(footprintType);
        auto structType = builder.createStructType();
        auto isSingleLevel = builder.createStructKey();
        builder.addNameHintDecoration(isSingleLevel, UnownedStringSlice("isSingleLevel"));
        builder.createStructField(structType, isSingleLevel, builder.getBoolType());

        auto anchor = builder.createStructKey();
        builder.addNameHintDecoration(anchor, UnownedStringSlice("anchor"));
        builder.createStructField(
            structType,
            anchor,
            builder.getVectorType(builder.getUIntType(), ND));

        auto offset = builder.createStructKey();
        builder.addNameHintDecoration(offset, UnownedStringSlice("offset"));
        builder.createStructField(
            structType,
            offset,
            builder.getVectorType(builder.getUIntType(), ND));

        auto mask = builder.createStructKey();
        builder.addNameHintDecoration(mask, UnownedStringSlice("mask"));
        builder.createStructField(
            structType,
            mask,
            builder.getVectorType(
                builder.getUIntType(),
                builder.getIntValue(builder.getIntType(), 2)));

        auto lod = builder.createStructKey();
        builder.addNameHintDecoration(lod, UnownedStringSlice("lod"));
        builder.createStructField(structType, lod, builder.getUIntType());

        auto granularity = builder.createStructKey();
        builder.addNameHintDecoration(granularity, UnownedStringSlice("granularity"));
        builder.createStructField(structType, granularity, builder.getUIntType());

        return structType;
    }

    LoweredStructuredBufferTypeInfo lowerStructuredBufferType(IRHLSLStructuredBufferTypeBase* inst)
    {
        LoweredStructuredBufferTypeInfo result;
        if (m_loweredStructuredBufferTypes.tryGetValue(inst, result))
            return result;

        auto layoutRules = getTypeLayoutRuleForBuffer(m_sharedContext->m_targetProgram, inst);

        IRBuilder builder(m_sharedContext->m_irModule);

        builder.setInsertBefore(inst);
        auto elementType = inst->getElementType();
        IRSizeAndAlignment elementSize;
        getSizeAndAlignment(
            m_sharedContext->m_targetProgram->getOptionSet(),
            layoutRules,
            elementType,
            &elementSize);
        elementSize = layoutRules->alignCompositeElement(elementSize);

        const auto arrayType = builder.getUnsizedArrayType(
            inst->getElementType(),
            builder.getIntValue(builder.getIntType(), elementSize.getStride()));
        const auto structType = builder.createStructType();
        builder.addPhysicalTypeDecoration(structType);
        const auto arrayKey = builder.createStructKey();
        builder.createStructField(structType, arrayKey, arrayType);
        IRSizeAndAlignment structSize;
        getSizeAndAlignment(
            m_sharedContext->m_targetProgram->getOptionSet(),
            layoutRules,
            structType,
            &structSize);

        StringBuilder nameSb;
        switch (inst->getOp())
        {
        case kIROp_HLSLRWStructuredBufferType:
            nameSb << "RWStructuredBuffer";
            break;
        case kIROp_HLSLAppendStructuredBufferType:
            nameSb << "AppendStructuredBuffer";
            break;
        case kIROp_HLSLConsumeStructuredBufferType:
            nameSb << "ConsumeStructuredBuffer";
            break;
        case kIROp_HLSLRasterizerOrderedStructuredBufferType:
            nameSb << "RasterizerOrderedStructuredBuffer";
            break;
        default:
            nameSb << "StructuredBuffer";
            break;
        }
        builder.addNameHintDecoration(structType, nameSb.getUnownedSlice());
        if (m_sharedContext->isSpirv14OrLater())
            builder.addDecorationIfNotExist(structType, kIROp_SPIRVBlockDecoration);
        else
            builder.addDecorationIfNotExist(structType, kIROp_SPIRVBufferBlockDecoration);

        result.structType = structType;
        result.arrayKey = arrayKey;
        result.runtimeArrayType = arrayType;
        m_loweredStructuredBufferTypes[inst] = result;
        return result;
    }

    // We will use a single work list of instructions that need
    // to be considered for specialization or simplification,
    // whether generic, existential, etc.
    //
    OrderedHashSet<IRInst*> workList;

    void addToWorkList(IRInst* inst)
    {
        if (workList.add(inst))
        {
            addUsersToWorkList(inst);
        }
    }

    void addUsersToWorkList(IRInst* inst)
    {
        for (auto use = inst->firstUse; use; use = use->nextUse)
        {
            auto user = use->getUser();

            addToWorkList(user);
        }
    }

    SPIRVLegalizationContext(
        SPIRVEmitSharedContext* sharedContext,
        IRModule* module,
        DiagnosticSink* sink)
        : m_sharedContext(sharedContext), m_module(module), m_sink(sink)
    {
    }

    // Wraps the element type of a constant buffer or parameter block in a struct if it is not
    // already a struct, returns the newly created struct type.
    IRType* wrapConstantBufferElement(IRInst* cbParamInst)
    {
        auto innerType = as<IRParameterGroupType>(cbParamInst->getDataType())->getElementType();
        IRBuilder builder(cbParamInst);
        builder.setInsertBefore(cbParamInst);
        auto structType = builder.createStructType();
        builder.addPhysicalTypeDecoration(structType);
        addToWorkList(structType);
        StringBuilder sb;
        sb << "cbuffer_";
        getTypeNameHint(sb, innerType);
        sb << "_t";
        builder.addNameHintDecoration(structType, sb.produceString().getUnownedSlice());
        auto key = builder.createStructKey();
        builder.createStructField(structType, key, innerType);
        builder.setInsertBefore(cbParamInst);
        auto newCbType = builder.getType(cbParamInst->getDataType()->getOp(), structType);
        cbParamInst->setFullType(newCbType);
        auto rules = getTypeLayoutRuleForBuffer(
            m_sharedContext->m_targetProgram,
            cbParamInst->getDataType());
        IRSizeAndAlignment sizeAlignment;
        getSizeAndAlignment(
            m_sharedContext->m_targetProgram->getOptionSet(),
            rules,
            structType,
            &sizeAlignment);
        traverseUses(
            cbParamInst,
            [&](IRUse* use)
            {
                builder.setInsertBefore(use->getUser());
                auto addr = builder.emitFieldAddress(
                    builder.getPtrType(kIROp_PtrType, innerType, AddressSpace::Uniform),
                    cbParamInst,
                    key);
                use->set(addr);
            });
        return structType;
    }

    static void insertLoadAtLatestLocation(
        IRInst* addrInst,
        IRUse* inUse,
        AddressSpace addressSpace)
    {
        struct WorkItem
        {
            IRInst* addr;
            IRUse* use;
        };
        List<WorkItem> workList;
        List<IRInst*> instsToRemove;
        workList.add(WorkItem{addrInst, inUse});
        for (Index i = 0; i < workList.getCount(); i++)
        {
            auto use = workList[i].use;
            auto addr = workList[i].addr;
            auto user = use->getUser();
            IRBuilder builder(user);
            builder.setInsertBefore(user);

            if (user->getOp() == kIROp_GetLegalizedSPIRVGlobalParamAddr)
            {
                user->replaceUsesWith(addr);
                user->removeAndDeallocate();
            }
            else if (
                (as<IRGetElement>(user) || as<IRFieldExtract>(user)) && use == user->getOperands())
            {
                // If the use is the address operand of a getElement or FieldExtract,
                // replace the inst with the updated address and continue to follow the use chain.
                auto basePtrType = as<IRPtrTypeBase>(addr->getDataType());
                IRType* ptrType = nullptr;
                if (basePtrType->hasAddressSpace())
                    ptrType = builder.getPtrType(
                        kIROp_PtrType,
                        user->getDataType(),
                        basePtrType->getAddressSpace());
                else
                    ptrType = builder.getPtrType(kIROp_PtrType, user->getDataType());
                IRInst* subAddr = nullptr;
                if (user->getOp() == kIROp_GetElement)
                    subAddr = builder.emitElementAddress(
                        ptrType,
                        addr,
                        as<IRGetElement>(user)->getIndex());
                else
                    subAddr = builder.emitFieldAddress(
                        ptrType,
                        addr,
                        as<IRFieldExtract>(user)->getField());

                for (auto u = user->firstUse; u; u = u->nextUse)
                {
                    workList.add(WorkItem{subAddr, u});
                }
                instsToRemove.add(user);
            }
            else if (const auto spirvAsmOperand = as<IRSPIRVAsmOperandInst>(user))
            {
                // Skip load's for referenced `Input` variables since a ref implies
                // passing as is, which needs to be a pointer (pass as is).
                if (user->getDataType() && user->getDataType()->getOp() == kIROp_RefType &&
                    (addressSpace == AddressSpace::Input ||
                     addressSpace == AddressSpace::BuiltinInput))
                {
                    builder.replaceOperand(use, addr);
                    continue;
                }
                // If this is being used in an asm block, insert the load to
                // just prior to the block.
                const auto asmBlock = spirvAsmOperand->getAsmBlock();
                builder.setInsertBefore(asmBlock);
                auto loadedValue = builder.emitLoad(addr);
                builder.setInsertBefore(spirvAsmOperand);
                auto loadedValueOperand = builder.emitSPIRVAsmOperandInst(loadedValue);
                spirvAsmOperand->replaceUsesWith(loadedValueOperand);
                spirvAsmOperand->removeAndDeallocate();
            }
            else
            {
                if (!as<IRDecoration>(use->getUser()))
                {
                    auto val = builder.emitLoad(addr);
                    builder.replaceOperand(use, val);
                }
            }
        }

        for (auto i : instsToRemove)
        {
            i->removeAndDeallocate();
        }
    }

    // Returns true if the given type that should be decorated as in `UniformConstant` address
    // space. These are typically opaque resource handles that can't be marked as `Uniform`.
    bool isSpirvUniformConstantType(IRType* type)
    {
        if (as<IRTextureTypeBase>(type))
            return true;
        if (as<IRSubpassInputType>(type))
            return true;
        if (as<IRSamplerStateTypeBase>(type))
            return true;
        if (const auto arr = as<IRArrayTypeBase>(type))
            return isSpirvUniformConstantType(arr->getElementType());
        switch (type->getOp())
        {
        case kIROp_RaytracingAccelerationStructureType:
        case kIROp_GLSLAtomicUintType:
        case kIROp_RayQueryType:
            return true;
        default:
            return false;
        }
    }

    Stage getReferencingEntryPointStage(IRInst* inst)
    {
        for (auto use = inst->firstUse; use; use = use->nextUse)
        {
            if (auto f = getParentFunc(use->getUser()))
            {
                if (auto d = f->findDecoration<IREntryPointDecoration>())
                    return d->getProfile().getStage();
            }
        }
        return Stage::Unknown;
    }

    bool translatePerVertexInputType(IRInst* param)
    {
        if (auto interpolationModeDecor = param->findDecoration<IRInterpolationModeDecoration>())
        {
            if (interpolationModeDecor->getMode() == IRInterpolationMode::PerVertex)
            {
                if (getReferencingEntryPointStage(param) == Stage::Fragment)
                {
                    auto originalType = param->getFullType();
                    IRBuilder builder(param);
                    builder.setInsertBefore(param);
                    auto arrayType = builder.getArrayType(
                        originalType,
                        builder.getIntValue(builder.getIntType(), 3));
                    param->setFullType(arrayType);
                    return true;
                }
            }
        }
        return false;
    }

    void processGlobalParam(IRGlobalParam* inst)
    {
        // If the global param is not a pointer type, make it so and insert explicit load insts.
        auto ptrType = as<IRPtrTypeBase>(inst->getDataType());
        if (!ptrType)
        {
            bool needLoad = true;

            if (translatePerVertexInputType(inst))
                needLoad = false;

            auto innerType = inst->getFullType();

            auto arrayType = as<IRArrayTypeBase>(inst->getDataType());
            IRInst* arraySize = nullptr;
            if (arrayType)
            {
                arraySize = arrayType->getElementCount();
                innerType = arrayType->getElementType();
            }

            AddressSpace addressSpace = AddressSpace::ThreadLocal;
            // Figure out storage class based on builtin info or var layout.
            auto cls = getGlobalParamAddressSpace(inst);
            if (cls != AddressSpace::Generic)
                addressSpace = cls;

            // Don't do any processing for specialization constants.
            if (addressSpace == AddressSpace::SpecializationConstant)
            {
                return;
            }

            // Opaque resource handles can't be in Uniform for Vulkan, if they are
            // placed here then put them in UniformConstant instead
            if (isSpirvUniformConstantType(inst->getDataType()))
            {
                addressSpace = AddressSpace::UniformConstant;
            }

            // Strip any HLSL wrappers
            IRBuilder builder(m_sharedContext->m_irModule);
            auto cbufferType = as<IRConstantBufferType>(innerType);
            auto paramBlockType = as<IRParameterBlockType>(innerType);
            if (cbufferType || paramBlockType)
            {
                innerType = as<IRUniformParameterGroupType>(innerType)->getElementType();
                if (addressSpace == AddressSpace::ThreadLocal)
                    addressSpace = AddressSpace::Uniform;
                // Constant buffer is already treated like a pointer type, and
                // we are not adding another layer of indirection when replacing it
                // with a pointer type. Therefore we don't need to insert a load at
                // use sites.
                needLoad = false;
                // If inner element type is not a struct type, we need to wrap it with
                // a struct.
                if (!as<IRStructType>(innerType))
                {
                    innerType = wrapConstantBufferElement(inst);
                }
                builder.addDecorationIfNotExist(innerType, kIROp_SPIRVBlockDecoration);

                auto varLayoutInst = inst->findDecoration<IRLayoutDecoration>();
                if (paramBlockType)
                {
                    // A parameter block typed global parameter will have a VarLayout
                    // that contains an OffsetAttr(RegisterSpace, spaceId).
                    // We need to turn this VarLayout into a standard cbuffer VarLayout
                    // in the form of OffsetAttr(ConstantBuffer, 0, spaceId).
                    builder.setInsertBefore(inst);
                    IRVarLayout* varLayout = nullptr;
                    if (varLayoutInst)
                        varLayout = as<IRVarLayout>(varLayoutInst->getLayout());
                    if (varLayout)
                    {
                        auto registerSpaceOffsetAttr =
                            varLayout->findOffsetAttr(LayoutResourceKind::SubElementRegisterSpace);
                        if (registerSpaceOffsetAttr)
                        {
                            List<IRInst*> operands;
                            for (UInt i = 0; i < varLayout->getOperandCount(); i++)
                                operands.add(varLayout->getOperand(i));
                            operands.add(builder.getVarOffsetAttr(
                                LayoutResourceKind::ConstantBuffer,
                                0,
                                registerSpaceOffsetAttr->getOffset()));
                            auto newLayout = builder.getVarLayout(operands);
                            varLayoutInst->setOperand(0, newLayout);
                            varLayout->removeAndDeallocate();
                        }
                    }
                }
                else if (addressSpace == AddressSpace::PushConstant)
                {
                    // Push constant params does not need a VarLayout.
                    varLayoutInst->removeAndDeallocate();
                }
            }
            else if (auto structuredBufferType = as<IRHLSLStructuredBufferTypeBase>(innerType))
            {
                innerType = lowerStructuredBufferType(structuredBufferType).structType;
                addressSpace = getStorageBufferAddressSpace();
                needLoad = false;

                auto memoryFlags = MemoryQualifierSetModifier::Flags::kNone;

                // structured buffers in GLSL should be annotated as ReadOnly
                if (as<IRHLSLStructuredBufferType>(structuredBufferType))
                    memoryFlags = MemoryQualifierSetModifier::Flags::kReadOnly;
                if (as<IRHLSLRasterizerOrderedStructuredBufferType>(structuredBufferType))
                    memoryFlags = MemoryQualifierSetModifier::Flags::kRasterizerOrdered;

                if (memoryFlags != MemoryQualifierSetModifier::Flags::kNone)
                    builder.addMemoryQualifierSetDecoration(inst, memoryFlags);
            }
            else if (
                auto glslShaderStorageBufferType = as<IRGLSLShaderStorageBufferType>(innerType))
            {
                innerType = glslShaderStorageBufferType->getElementType();
                if (m_sharedContext->isSpirv14OrLater())
                {
                    builder.addDecorationIfNotExist(innerType, kIROp_SPIRVBlockDecoration);
                    addressSpace = AddressSpace::StorageBuffer;
                }
                else
                {
                    builder.addDecorationIfNotExist(innerType, kIROp_SPIRVBufferBlockDecoration);
                    addressSpace = AddressSpace::Uniform;
                }
                needLoad = false;
            }

            auto innerElementType = innerType;
            if (arrayType)
            {
                innerType =
                    (IRType*)builder.getArrayTypeBase(arrayType->getOp(), innerType, arraySize);
                if (!arraySize)
                {
                    builder.addRequireSPIRVDescriptorIndexingExtensionDecoration(inst);
                }
            }

            // Make a pointer type of storageClass.
            builder.setInsertBefore(inst);
            ptrType = builder.getPtrType(kIROp_PtrType, innerType, addressSpace);
            inst->setFullType(ptrType);
            if (needLoad)
            {
                // Insert an explicit load at each use site.
                traverseUses(
                    inst,
                    [&](IRUse* use) { insertLoadAtLatestLocation(inst, use, addressSpace); });
            }
            else if (arrayType)
            {
                traverseUses(
                    inst,
                    [&](IRUse* use)
                    {
                        auto user = use->getUser();
                        if (auto getElement = as<IRGetElement>(user))
                        {
                            // For array resources, getElement(r, index) ==> getElementPtr(r,
                            // index).
                            IRBuilder builder(getElement);
                            builder.setInsertBefore(user);
                            auto newAddr = builder.emitElementAddress(
                                builder.getPtrType(kIROp_PtrType, innerElementType, addressSpace),
                                inst,
                                getElement->getIndex());
                            user->replaceUsesWith(newAddr);
                            user->removeAndDeallocate();
                            return;
                        }
                    });
            }
        }
        processGlobalVar(inst);
    }

    AddressSpace getAddressSpaceFromGlobalParamResourceKind(LayoutResourceKind kind)
    {
        AddressSpace addressSpace = AddressSpace::Generic;
        switch (kind)
        {
        case LayoutResourceKind::Uniform:
        case LayoutResourceKind::DescriptorTableSlot:
        case LayoutResourceKind::ConstantBuffer:
            addressSpace = AddressSpace::Uniform;
            break;
        case LayoutResourceKind::VaryingInput:
            addressSpace = AddressSpace::Input;
            break;
        case LayoutResourceKind::VaryingOutput:
            addressSpace = AddressSpace::Output;
            break;
        case LayoutResourceKind::ShaderResource:
        case LayoutResourceKind::UnorderedAccess:
            addressSpace = getStorageBufferAddressSpace();
            break;
        case LayoutResourceKind::PushConstantBuffer:
            addressSpace = AddressSpace::PushConstant;
            break;
        case LayoutResourceKind::SpecializationConstant:
            addressSpace = AddressSpace::SpecializationConstant;
            break;
        case LayoutResourceKind::RayPayload:
            addressSpace = AddressSpace::IncomingRayPayload;
            break;
        case LayoutResourceKind::CallablePayload:
            addressSpace = AddressSpace::IncomingCallableData;
            break;
        case LayoutResourceKind::HitAttributes:
            addressSpace = AddressSpace::HitAttribute;
            break;
        case LayoutResourceKind::ShaderRecord:
            addressSpace = AddressSpace::ShaderRecordBuffer;
            break;
        default:
            break;
        }
        return addressSpace;
    }

    AddressSpace getGlobalParamAddressSpace(IRInst* varInst)
    {
        if (auto builtinDecor = varInst->findDecoration<IRTargetBuiltinVarDecoration>())
        {
            switch (builtinDecor->getBuiltinVarName())
            {
            case IRTargetBuiltinVarName::SpvInstanceIndex:
            case IRTargetBuiltinVarName::SpvBaseInstance:
                return AddressSpace::BuiltinInput;
            }
        }

        auto varLayout = getVarLayout(varInst);
        if (!varLayout)
            return AddressSpace::Generic;

        auto typeLayout = varLayout->getTypeLayout()->unwrapArray();
        if (auto parameterGroupTypeLayout = as<IRParameterGroupTypeLayout>(typeLayout))
        {
            varLayout = parameterGroupTypeLayout->getContainerVarLayout();
        }

        auto result = AddressSpace::Generic;
        for (auto rr : varLayout->getOffsetAttrs())
        {
            auto addressSpace = getAddressSpaceFromGlobalParamResourceKind(rr->getResourceKind());
            // If we haven't inferred a storage class yet, use the one we just found.
            if (result == AddressSpace::Generic)
                result = addressSpace;
            else if (result != addressSpace)
            {
                // If we have inferred an address space, and it is different from the one we just
                // found, then we have conflicting uses of the resource, and we cannot infer an
                // address space. An exception is that a uniform storage class can be further
                // specialized by PushConstants.
                if (result == AddressSpace::Uniform)
                    result = addressSpace;
                else
                    SLANG_UNEXPECTED("Var layout contains conflicting resource uses, cannot "
                                     "resolve a storage class address space.");
            }
        }
        auto systemValueAttr = varLayout->findSystemValueSemanticAttr();

        if (systemValueAttr)
        {
            // TODO: is this needed?
            String semanticName = systemValueAttr->getName();
            semanticName = semanticName.toLower();
            if (semanticName == "sv_pointsize")
                result = AddressSpace::BuiltinInput;
        }

        switch (result)
        {
        case AddressSpace::Input:
            if (systemValueAttr)
                result = AddressSpace::BuiltinInput;
            break;
        case AddressSpace::Output:
            if (systemValueAttr)
                result = AddressSpace::BuiltinOutput;
            break;
        }
        return result;
    }

    void processVar(IRInst* inst)
    {
        auto oldPtrType = as<IRPtrType>(inst->getDataType());
        if (!oldPtrType->hasAddressSpace())
        {
            IRBuilder builder(inst);
            builder.setInsertBefore(inst);
            auto newPtrType = builder.getPtrType(
                oldPtrType->getOp(),
                oldPtrType->getValueType(),
                AddressSpace::Function);
            inst->setFullType(newPtrType);
            addUsersToWorkList(inst);
        }
    }

    void processParam(IRInst* inst)
    {
        auto block = getBlock(inst);
        auto func = getParentFunc(block);
        if (!block || !func)
            return;
        auto oldPtrType = as<IRPtrType>(inst->getDataType());
        if (!oldPtrType)
            return;
        if (!oldPtrType->hasAddressSpace())
        {
            AddressSpace addressSpace = AddressSpace::Generic;

            if (block == func->getFirstBlock())
            {
                // A pointer typed function parameter should always be in the storage buffer address
                // space.
                addressSpace = AddressSpace::UserPointer;
            }
            else
            {
                // The address space of a phi inst should always be the same as arguments.
                auto args = getPhiArgs(inst);
                for (auto arg : args)
                {
                    auto argPtrType = as<IRPtrType>(arg->getDataType());
                    if (argPtrType->hasAddressSpace())
                    {
                        if (addressSpace == AddressSpace::Generic)
                            addressSpace = argPtrType->getAddressSpace();
                        else if (addressSpace != argPtrType->getAddressSpace())
                            m_sharedContext->m_sink->diagnose(
                                inst,
                                Diagnostics::inconsistentPointerAddressSpace,
                                inst);
                    }
                }
            }
            if (addressSpace != AddressSpace::Generic)
            {
                IRBuilder builder(inst);
                builder.setInsertBefore(inst);
                auto newPtrType = builder.getPtrType(
                    oldPtrType->getOp(),
                    oldPtrType->getValueType(),
                    AddressSpace::UserPointer);
                inst->setFullType(newPtrType);
                addUsersToWorkList(inst);
            }
        }
    }

    void processGlobalVar(IRInst* inst)
    {
        auto oldPtrType = as<IRPtrTypeBase>(inst->getDataType());
        if (!oldPtrType)
            return;

        // Update the pointer value type with storage-buffer-address-space-decorated types.
        auto newPtrValueType = oldPtrType->getValueType();
        if (newPtrValueType != oldPtrType->getValueType())
        {
            IRBuilder builder(inst);
            builder.setInsertBefore(inst);
            IRType* newPtrType = oldPtrType->hasAddressSpace()
                                     ? builder.getPtrType(
                                           oldPtrType->getOp(),
                                           newPtrValueType,
                                           oldPtrType->getAddressSpace())
                                     : builder.getPtrType(oldPtrType->getOp(), newPtrValueType);
            inst->setFullType(newPtrType);
        }

        // If the pointer type is already qualified with address spaces (such as
        // lowered pointer type from a `HLSLStructuredBufferType`), make no
        // further modifications.
        if (oldPtrType->hasAddressSpace())
        {
            addUsersToWorkList(inst);
            return;
        }

        auto addressSpace = AddressSpace::ThreadLocal;
        if (as<IRGroupSharedRate>(inst->getRate()))
        {
            addressSpace = AddressSpace::GroupShared;
        }
        else
        {
            auto cls = getGlobalParamAddressSpace(inst);
            if (cls != AddressSpace::Generic)
                addressSpace = cls;
        }
        for (auto decor : inst->getDecorations())
        {
            switch (decor->getOp())
            {
            case kIROp_VulkanRayPayloadDecoration:
                addressSpace = AddressSpace::RayPayloadKHR;
                break;
            case kIROp_VulkanRayPayloadInDecoration:
                addressSpace = AddressSpace::IncomingRayPayload;
                break;
            case kIROp_VulkanCallablePayloadDecoration:
                addressSpace = AddressSpace::CallableDataKHR;
                break;
            case kIROp_VulkanCallablePayloadInDecoration:
                addressSpace = AddressSpace::IncomingCallableData;
                break;
            case kIROp_VulkanHitObjectAttributesDecoration:
                addressSpace = AddressSpace::HitObjectAttribute;
                break;
            case kIROp_VulkanHitAttributesDecoration:
                addressSpace = AddressSpace::HitAttribute;
                break;
            }
        }
        IRBuilder builder(m_sharedContext->m_irModule);
        builder.setInsertBefore(inst);
        auto newPtrType =
            builder.getPtrType(oldPtrType->getOp(), oldPtrType->getValueType(), addressSpace);
        inst->setFullType(newPtrType);
        addUsersToWorkList(inst);
        return;
    }

    void processCall(IRCall* inst)
    {
        auto funcValue = inst->getOperand(0);
        if (auto targetIntrinsic = Slang::findBestTargetIntrinsicDecoration(
                funcValue,
                m_sharedContext->m_targetRequest->getTargetCaps()))
        {
            SpvSnippet* snippet = m_sharedContext->getParsedSpvSnippet(targetIntrinsic);
            if (!snippet)
                return;
            if (snippet->resultStorageClass != SpvStorageClassMax)
                SLANG_UNIMPLEMENTED_X(
                    "Specifying storage classes in spirv __target_intrinsic snippets");
            return;
        }

        // According to SPIRV spec, the if the operands of a call has pointer
        // type, then it can only be a memory-object. This means that if the
        // pointer is a result of `getElementPtr`, we cannot use it as an
        // argument. In this case, we have to allocate a temp var to pass the
        // value, and write them back to the original pointer after the call.
        //
        // > SPIRV Spec section 2.16.1:
        // >   - Any pointer operand to an OpFunctionCall must be a memory object
        // >     declaration, or
        // >     - a pointer to an element in an array that is a memory object
        // >       declaration, where the element type is OpTypeSampler or OpTypeImage.
        //
        List<IRInst*> newArgs;
        struct WriteBackPair
        {
            IRInst* originalAddrArg;
            IRInst* tempVar;
        };
        List<WriteBackPair> writeBacks;
        IRBuilder builder(inst);
        builder.setInsertBefore(inst);
        auto funcType = as<IRFuncType>(funcValue->getFullType());
        for (UInt i = 0; i < inst->getArgCount(); i++)
        {
            auto arg = inst->getArg(i);
            auto paramType = funcType->getParamType(i);
            if (as<IRPtrType>(paramType))
            {
                // If the parameter has an explicit pointer type,
                // then we know the user is using the variable pointer
                // capability to pass a true pointer.
                // In this case we should not rewrite the call.
                newArgs.add(arg);
                continue;
            }
            auto ptrType = as<IRPtrTypeBase>(arg->getDataType());
            if (!as<IRPtrTypeBase>(arg->getDataType()))
            {
                newArgs.add(arg);
                continue;
            }
            // Is the arg already a memory-object by SPIRV definition?
            // If so we don't need to allocate a temp var.
            switch (arg->getOp())
            {
            case kIROp_Var:
            case kIROp_GlobalVar:
                newArgs.add(arg);
                continue;
            case kIROp_Param:
                if (arg->getParent() == getParentFunc(arg)->getFirstBlock())
                {
                    newArgs.add(arg);
                    continue;
                }
                break;
            default:
                break;
            }
            auto root = getRootAddr(arg);
            if (root)
            {
                switch (root->getOp())
                {
                case kIROp_RWStructuredBufferGetElementPtr:
                    if (funcType)
                    {
                        if (funcType->getParamCount() > i &&
                            as<IRRefType>(funcType->getParamType(i)))
                        {
                            // If we are passing an address from a structured buffer as a
                            // ref argument, pass the original pointer as is.
                            // This is to support core module atomic functions.
                            newArgs.add(arg);
                            continue;
                        }
                    }
                }
            }

            // If we reach here, we need to allocate a temp var.
            auto tempVar = builder.emitVar(ptrType->getValueType());
            auto load = builder.emitLoad(arg);
            builder.emitStore(tempVar, load);
            newArgs.add(tempVar);
            writeBacks.add(WriteBackPair{arg, tempVar});
        }
        SLANG_ASSERT((UInt)newArgs.getCount() == inst->getArgCount());
        if (writeBacks.getCount())
        {
            auto newCall = builder.emitCallInst(inst->getFullType(), inst->getCallee(), newArgs);
            for (auto wb : writeBacks)
            {
                auto newVal = builder.emitLoad(wb.tempVar);
                builder.emitStore(wb.originalAddrArg, newVal);
            }
            inst->replaceUsesWith(newCall);
            inst->removeAndDeallocate();
            addUsersToWorkList(newCall);
        }
    }

    Dictionary<IRInst*, IRInst*> m_mapArrayValueToVar;

    // Replace getElement(x, i) with, y = store(x); p = getElementPtr(y, i); load(p),
    // when i is not a constant. SPIR-V has no support for dynamic indexing into values like we do.
    // It may be advantageous however to do this further up the pipeline
    void processGetElement(IRGetElement* inst)
    {
        IRInst* x = nullptr;
        List<IRInst*> indices;
        IRGetElement* c = inst;
        do
        {
            if (as<IRIntLit>(c->getIndex()))
                break;
            x = c->getBase();
            indices.add(c->getIndex());
        } while (c = as<IRGetElement>(c->getBase()), c);

        if (!x)
            return;

        IRBuilder builder(m_sharedContext->m_irModule);
        IRInst* y = nullptr;
        builder.setInsertBefore(inst);
        if (!m_mapArrayValueToVar.tryGetValue(x, y))
        {
            if (x->getParent()->getOp() == kIROp_Module)
                builder.setInsertBefore(inst);
            else
                setInsertAfterOrdinaryInst(&builder, x);
            y = builder.emitVar(x->getDataType(), AddressSpace::Function);
            builder.emitStore(y, x);
            if (x->getParent()->getOp() != kIROp_Module)
                m_mapArrayValueToVar.set(x, y);
        }
        builder.setInsertBefore(inst);
        for (Index i = indices.getCount() - 1; i >= 0; --i)
            y = builder.emitElementAddress(y, indices[i]);
        const auto newInst = builder.emitLoad(y);
        inst->replaceUsesWith(newInst);
        inst->removeAndDeallocate();
        addUsersToWorkList(newInst);
    }

    void processGetElementPtrImpl(IRInst* gepInst, IRInst* base, IRInst* index)
    {
        if (auto ptrType = as<IRPtrTypeBase>(base->getDataType()))
        {
            if (!ptrType->hasAddressSpace())
                return;
            auto oldResultType = as<IRPtrTypeBase>(gepInst->getDataType());
            if (oldResultType->getAddressSpace() != ptrType->getAddressSpace())
            {
                IRBuilder builder(m_sharedContext->m_irModule);
                builder.setInsertBefore(gepInst);
                auto newPtrType = builder.getPtrType(
                    oldResultType->getOp(),
                    oldResultType->getValueType(),
                    ptrType->getAddressSpace());
                IRInst* args[2] = {base, index};
                auto newInst = builder.emitIntrinsicInst(newPtrType, gepInst->getOp(), 2, args);
                gepInst->replaceUsesWith(newInst);
                gepInst->removeAndDeallocate();
                addUsersToWorkList(newInst);
            }
        }
    }

    void processGetElementPtr(IRGetElementPtr* gepInst)
    {
        processGetElementPtrImpl(gepInst, gepInst->getBase(), gepInst->getIndex());
    }

    void processRWStructuredBufferGetElementPtr(IRRWStructuredBufferGetElementPtr* gepInst)
    {
        processGetElementPtrImpl(gepInst, gepInst->getBase(), gepInst->getIndex());
    }

    void processMeshOutputGetElementPtr(IRMeshOutputRef* gepInst)
    {
        processGetElementPtrImpl(gepInst, gepInst->getBase(), gepInst->getIndex());
    }

    void processMeshOutputSet(IRMeshOutputSet* setInst)
    {
        IRBuilder builder(m_sharedContext->m_irModule);
        builder.setInsertBefore(setInst);
        const auto p = builder.emitElementAddress(setInst->getBase(), setInst->getIndex());
        const auto s = builder.emitStore(p, setInst->getElementValue());
        setInst->removeAndDeallocate();
        addToWorkList(p);
        addToWorkList(s);
    }

    void processGetOffsetPtr(IRInst* offsetPtrInst)
    {
        auto ptrOperandType = as<IRPtrType>(offsetPtrInst->getOperand(0)->getDataType());
        if (!ptrOperandType)
            return;
        if (!ptrOperandType->hasAddressSpace())
            return;
        auto resultPtrType = as<IRPtrType>(offsetPtrInst->getDataType());
        if (!resultPtrType)
            return;
        if (resultPtrType->getAddressSpace() != ptrOperandType->getAddressSpace())
        {
            IRBuilder builder(offsetPtrInst);
            builder.setInsertBefore(offsetPtrInst);
            auto newResultType = builder.getPtrType(
                resultPtrType->getOp(),
                resultPtrType->getValueType(),
                ptrOperandType->getAddressSpace());
            auto newInst = builder.replaceOperand(&offsetPtrInst->typeUse, newResultType);
            addUsersToWorkList(newInst);
        }
    }

    AddressSpace getStorageBufferAddressSpace()
    {
        return m_sharedContext->isSpirv14OrLater() ? AddressSpace::StorageBuffer
                                                   : AddressSpace::Uniform;
    }

    void processStructuredBufferLoad(IRInst* loadInst)
    {
        auto sb = loadInst->getOperand(0);
        auto index = loadInst->getOperand(1);
        IRBuilder builder(sb);
        builder.setInsertBefore(loadInst);
        IRInst* args[] = {sb, index};
        auto addrInst = builder.emitIntrinsicInst(
            builder
                .getPtrType(kIROp_PtrType, loadInst->getFullType(), getStorageBufferAddressSpace()),
            kIROp_RWStructuredBufferGetElementPtr,
            2,
            args);
        auto value = builder.emitLoad(addrInst);
        loadInst->replaceUsesWith(value);
        loadInst->removeAndDeallocate();
        addUsersToWorkList(value);
    }

    void processRWStructuredBufferStore(IRInst* storeInst)
    {
        auto sb = storeInst->getOperand(0);
        auto index = storeInst->getOperand(1);
        auto value = storeInst->getOperand(2);
        IRBuilder builder(sb);
        builder.setInsertBefore(storeInst);
        IRInst* args[] = {sb, index};
        auto addrInst = builder.emitIntrinsicInst(
            builder.getPtrType(kIROp_PtrType, value->getFullType(), getStorageBufferAddressSpace()),
            kIROp_RWStructuredBufferGetElementPtr,
            2,
            args);
        auto newStore = builder.emitStore(addrInst, value);
        storeInst->replaceUsesWith(newStore);
        storeInst->removeAndDeallocate();
        addUsersToWorkList(newStore);
    }

    void processImageSubscript(IRImageSubscript* subscript)
    {
        if (auto ptrType = as<IRPtrTypeBase>(subscript->getDataType()))
        {
            if (ptrType->hasAddressSpace())
                return;
            IRBuilder builder(m_sharedContext->m_irModule);
            builder.setInsertBefore(subscript);
            auto newPtrType =
                builder.getPtrType(ptrType->getOp(), ptrType->getValueType(), AddressSpace::Image);
            subscript->setFullType(newPtrType);

            // HACK: assumes the image operand is a load and replace it with
            // the pointer to satisfy SPIRV requirements.
            // We should consider changing the front-end to pass `this` by ref
            // for the __ref accessor so we will be guaranteed to have a pointer
            // image operand here.
            auto image = subscript->getImage();
            if (auto load = as<IRLoad>(image))
                subscript->setOperand(0, load->getPtr());

            addUsersToWorkList(subscript);
        }
    }

    void processFieldAddress(IRFieldAddress* inst)
    {
        if (auto ptrType = as<IRPtrTypeBase>(inst->getBase()->getDataType()))
        {
            if (!ptrType->hasAddressSpace())
                return;
            auto oldResultType = as<IRPtrTypeBase>(inst->getDataType());
            auto oldValueType = oldResultType->getValueType();
            auto newValueType = oldValueType;

            if (oldValueType != newValueType ||
                oldResultType->getAddressSpace() != ptrType->getAddressSpace())
            {
                IRBuilder builder(m_sharedContext->m_irModule);
                builder.setInsertBefore(inst);
                auto newPtrType = builder.getPtrType(
                    oldResultType->getOp(),
                    newValueType,
                    ptrType->getAddressSpace());
                auto newInst =
                    builder.emitFieldAddress(newPtrType, inst->getBase(), inst->getField());
                inst->replaceUsesWith(newInst);
                inst->removeAndDeallocate();
                addUsersToWorkList(newInst);
            }
        }
    }

    void processFieldExtract(IRFieldExtract* inst)
    {
        auto ptrType = as<IRPtrType>(inst->getDataType());
        if (!ptrType)
            return;
        auto newPtrType = ptrType;
        if (newPtrType == ptrType)
            return;
        IRBuilder builder(inst);
        auto newInst = builder.replaceOperand(&inst->typeUse, newPtrType);
        addUsersToWorkList(newInst);
    }

    void duplicateMergeBlockIfNeeded(IRUse* breakBlockUse)
    {
        auto breakBlock = as<IRBlock>(breakBlockUse->get());
        if (breakBlock->getFirstInst()->getOp() != kIROp_Unreachable)
        {
            return;
        }
        bool hasMoreThanOneUser = false;
        for (auto use = breakBlock->firstUse; use; use = use->nextUse)
        {
            if (use->getUser() != breakBlockUse->getUser())
            {
                hasMoreThanOneUser = true;
                break;
            }
        }
        if (!hasMoreThanOneUser)
            return;

        // Create a duplicate block for this use.
        IRBuilder builder(breakBlock);
        builder.setInsertBefore(breakBlock);
        auto block = builder.emitBlock();
        builder.emitUnreachable();
        breakBlockUse->set(block);
    }

    void processLoop(IRLoop* loop)
    {
        // 2.11.1. Rules for Structured Control-flow Declarations
        // Structured control flow declarations must satisfy the following
        // rules:
        //   - the merge block declared by a header block must not be a merge
        //     block declared by any other header block
        //   - each header block must strictly structurally dominate its merge
        //     block
        //   - all back edges must branch to a loop header, with each loop
        //     header having exactly one back edge branching to it
        //   - for a given loop header, its merge block, OpLoopMerge Continue
        //     Target, and corresponding back-edge block:
        //       - the Continue Target and merge block must be different blocks
        //       - the loop header must structurally dominate the Continue
        //         Target
        //       - the Continue Target must structurally dominate the back-edge
        //         block
        //       - the back-edge block must structurally post dominate the
        //         Continue Target

        // By this point, we should have already eliminated all continue jumps and
        // turned them into a break jump. So all loop insts should satisfy
        // continueBlock == targetBlock.
        const auto t = loop->getTargetBlock();
        auto c = loop->getContinueBlock();

        // Our IR allows multiple back-edges to a loop header if this is also
        // the loop continue block. SPIR-V does not so replace them with a
        // single intermediate block
        if (c == t)
        {
            // Subtract one predecessor for the loop entry
            const auto numBackEdges = c->getPredecessors().getCount() - 1;

            // If we have multiple back-edges, make a new block at the end of
            // the loop to be the new continue block which jumps straight to
            // the loop header.
            //
            // If we have a single back-edge, we still may need to perform this
            // transformation to make sure that the back-edge block
            // structurally post-dominates the continue target. For example
            // consider the loop:
            //
            // int i = 0;
            // while(true)
            //     if(foo()) break;
            //
            // If we translate this to
            // loop target=t break=b, continue=t
            // t: if foo goto x else goto y
            // x: goto b -- break
            // y: goto t
            // b: ...
            //
            // The back edge block, y, does not post-dominate the continue target, t.
            //
            // So we transform this to:
            //
            // loop target=t break=b, continue=c
            // t: if foo goto x else goto y
            // x: goto b -- break
            // y: goto c
            // c: goto t
            // b: ...
            //
            // Now the back edge block and the continue target are one and the
            // same, so the condition trivially holds.
            //
            // TODO: We don't need to always perform this, we could replace the
            // below condition with `numBackEdges > 1 ||
            //     !postDominates(backJumpingBlock, c)`
            if (numBackEdges > 0)
            {
                IRBuilder builder(m_sharedContext->m_irModule);
                builder.setInsertInto(loop->getParent());
                IRCloneEnv cloneEnv;
                cloneEnv.squashChildrenMapping = true;

                // Insert a new continue block at the end of the loop
                const auto newContinueBlock = builder.emitBlock();
                addToWorkList(newContinueBlock);

                newContinueBlock->insertBefore(loop->getBreakBlock());

                // This block simply branches to the loop header, forwarding
                // any params
                List<IRInst*> ps;
                for (const auto p : c->getParams())
                {
                    const auto q = cast<IRParam>(cloneInst(&cloneEnv, &builder, p));
                    newContinueBlock->addParam(q);
                    ps.add(q);
                }
                // Replace all jumps to our loop header/old continue block
                c->replaceUsesWith(newContinueBlock);

                // Restore the target block
                loop->block.set(t);

                // Branch to the target in our new continue block
                auto branch = builder.emitBranch(t, ps.getCount(), ps.getBuffer());
                addToWorkList(branch);
            }
        }
        duplicateMergeBlockIfNeeded(&loop->breakBlock);
        addToWorkList(loop->getTargetBlock());
    }

    void processIfElse(IRIfElse* inst)
    {
        duplicateMergeBlockIfNeeded(&inst->afterBlock);

        // SPIRV does not allow using merge block directly as true/false block,
        // so we need to create an intermediate block if this is the case.
        IRBuilder builder(inst);
        if (inst->getTrueBlock() == inst->getAfterBlock())
        {
            builder.setInsertBefore(inst->getAfterBlock());
            auto newBlock = builder.emitBlock();
            builder.emitBranch(inst->getAfterBlock());
            inst->trueBlock.set(newBlock);
            addToWorkList(newBlock);
        }
        if (inst->getFalseBlock() == inst->getAfterBlock())
        {
            builder.setInsertBefore(inst->getAfterBlock());
            auto newBlock = builder.emitBlock();
            builder.emitBranch(inst->getAfterBlock());
            inst->falseBlock.set(newBlock);
            addToWorkList(newBlock);
        }
    }

    void processSwitch(IRSwitch* inst)
    {
        duplicateMergeBlockIfNeeded(&inst->breakLabel);

        // SPIRV does not allow using merge block directly as case block,
        // so we need to create an intermediate block if this is the case.
        IRBuilder builder(inst);
        if (inst->getDefaultLabel() == inst->getBreakLabel())
        {
            builder.setInsertBefore(inst->getBreakLabel());
            auto newBlock = builder.emitBlock();
            builder.emitBranch(inst->getBreakLabel());
            inst->defaultLabel.set(newBlock);
            addToWorkList(newBlock);
        }
        for (UInt i = 0; i < inst->getCaseCount(); i++)
        {
            if (inst->getCaseLabel(i) == inst->getBreakLabel())
            {
                builder.setInsertBefore(inst->getBreakLabel());
                auto newBlock = builder.emitBlock();
                builder.emitBranch(inst->getBreakLabel());
                inst->getCaseLabelUse(i)->set(newBlock);
                addToWorkList(newBlock);
            }
        }
    }

    void maybeHoistConstructInstToGlobalScope(IRInst* inst)
    {
        // If all of the operands to this instruction are global, and are not global
        // variables, we can hoist this constructor to be a global too.
        // This is important to make sure
        // that vectors made of constant components end up being emitted as
        // constant vectors (using OpConstantComposite).
        UIndex opIndex = 0;
        for (auto operand = inst->getOperands(); opIndex < inst->getOperandCount();
             operand++, opIndex++)
        {
            if (operand->get()->getParent() != m_module->getModuleInst())
                return;

            if (as<IRGlobalParam>(operand->get()))
                return;
        }
        inst->insertAtEnd(m_module->getModuleInst());
    }

    void processConstructor(IRInst* inst)
    {
        maybeHoistConstructInstToGlobalScope(inst);

        if (inst->getOp() == kIROp_MakeVector && inst->getParent()->getOp() == kIROp_Module &&
            inst->getOperandCount() !=
                (UInt)getIntVal(as<IRVectorType>(inst->getDataType())->getElementCount()))
        {
            // SPIRV's OpConstantComposite inst requires the number of operands to
            // exactly match the number of elements of the composite, so the general
            // form of vector construction will not work, and we need to convert it.
            //
            List<IRInst*> args;
            IRBuilder builder(inst);
            builder.setInsertBefore(inst);
            for (UInt i = 0; i < inst->getOperandCount(); i++)
            {
                auto operand = inst->getOperand(i);
                if (auto operandVecType = as<IRVectorType>(operand->getDataType()))
                {
                    auto operandVecSize = getIntVal(operandVecType->getElementCount());
                    for (IRIntegerValue j = 0; j < operandVecSize; j++)
                    {
                        args.add(builder.emitElementExtract(operand, j));
                    }
                }
                else
                {
                    args.add(operand);
                }
            }
            auto newMakeVector = builder.emitMakeVector(inst->getDataType(), args);
            inst->replaceUsesWith(newMakeVector);
        }
    }

    static bool isAsmInst(IRInst* inst)
    {
        return (as<IRSPIRVAsmInst>(inst) || as<IRSPIRVAsmOperand>(inst));
    }

    void processConvertTexel(IRInst* asmBlockInst, IRInst* inst)
    {
        // If we see `__convertTexel(x)`, we need to return a vector<__sampledElementType(x), 4>.
        IRInst* operand = inst->getOperand(0);
        auto elementType = getSPIRVSampledElementType(operand->getDataType());
        auto valueElementType = getVectorElementType(operand->getDataType());
        IRBuilder builder(inst);
        builder.setInsertBefore(asmBlockInst);
        if (elementType != valueElementType)
        {
            auto floatCastType = replaceVectorElementType(operand->getDataType(), elementType);
            operand = builder.emitCast(floatCastType, operand);
        }
        auto vecType = builder.getVectorType(elementType, 4);
        if (vecType != operand->getDataType())
        {
            if (!as<IRVectorType>(operand->getDataType()))
                operand = builder.emitMakeVectorFromScalar(vecType, operand);
            else
                operand = builder.emitVectorReshape(vecType, operand);
        }
        builder.setInsertBefore(inst);
        auto spvAsmOperand = builder.emitSPIRVAsmOperandInst(operand);
        inst->replaceUsesWith(spvAsmOperand);
        inst->removeAndDeallocate();
    }

    void processSPIRVAsm(IRSPIRVAsm* inst)
    {
        // Move anything that is not an spirv instruction to the outer parent.
        for (auto child : inst->getModifiableChildren())
        {
            if (!isAsmInst(child))
                child->insertBefore(inst);
            else if (child->getOp() == kIROp_SPIRVAsmOperandConvertTexel)
                processConvertTexel(inst, child);
        }
    }

    void legalizeSPIRVEntryPoint(IRFunc* func, IREntryPointDecoration* entryPointDecor)
    {
        auto stage = entryPointDecor->getProfile().getStage();
        switch (stage)
        {
        case Stage::Geometry:
            if (!func->findDecoration<IRInstanceDecoration>())
            {
                IRBuilder builder(func);
                builder.addDecoration(
                    func,
                    kIROp_InstanceDecoration,
                    builder.getIntValue(builder.getUIntType(), 1));
            }
            break;
        case Stage::Compute:
            if (!func->findDecoration<IRNumThreadsDecoration>())
            {
                IRBuilder builder(func);
                auto one = builder.getIntValue(builder.getUIntType(), 1);
                IRInst* args[3] = {one, one, one};
                builder.addDecoration(func, kIROp_NumThreadsDecoration, args, 3);
            }
            break;
        }
    }

    struct GlobalInstInliningContext : public GlobalInstInliningContextGeneric
    {
        bool isLegalGlobalInstForTarget(IRInst* inst) override
        {
            return as<IRSPIRVAsmOperand>(inst);
        }

        bool isInlinableGlobalInstForTarget(IRInst* inst) override
        {
            switch (inst->getOp())
            {
            case kIROp_SPIRVAsm:
                return true;
            default:
                break;
            }

            if (as<IRSPIRVAsmInst>(inst))
                return true;
            if (as<IRSPIRVAsmOperand>(inst))
                return true;
            return false;
        }

        bool shouldBeInlinedForTarget(IRInst* user) override
        {
            if (as<IRSPIRVAsmOperand>(user) && as<IRSPIRVAsmOperandInst>(user))
                return true;
            else if (as<IRSPIRVAsmInst>(user))
                return true;
            return false;
        }

        IRInst* getOutsideASM(IRInst* beforeInst) override
        {
            auto parent = beforeInst->getParent();
            while (parent)
            {
                if (as<IRSPIRVAsm>(parent))
                {
                    return parent;
                }
                parent = parent->getParent();
            }
            return beforeInst;
        }
    };

    void processBranch(IRInst* branch) { addToWorkList(branch->getOperand(0)); }

    void processPtrLit(IRInst* inst)
    {
        IRBuilder builder(inst);
        builder.setInsertBefore(inst);
        auto newPtrType = as<IRPtrType>(inst->getFullType());
        auto newInst =
            builder.emitCastIntToPtr(newPtrType, builder.getIntValue(builder.getUInt64Type(), 0));
        inst->replaceUsesWith(newInst);
        addUsersToWorkList(newInst);
    }

    void processStructField(IRStructField* field)
    {
        auto newFieldType = field->getFieldType();
        if (newFieldType != field->getFieldType())
            field->setFieldType(newFieldType);
    }

    void processComparison(IRInst* inst)
    {
        auto operand0 = inst->getOperand(0);
        if (as<IRPtrType>(operand0->getDataType()))
        {
            // If we are doing pointer comparison, convert the operands into uints first.
            IRBuilder builder(inst);
            builder.setInsertBefore(inst);
            auto castToUInt = [&](IRInst* operand)
            {
                if (as<IRPtrLit>(operand))
                    return builder.getIntValue(builder.getUInt64Type(), 0);
                else
                    return builder.emitCastPtrToInt(operand);
            };
            auto newOperand0 = castToUInt(operand0);
            SLANG_ASSERT(as<IRPtrType>(inst->getOperand(1)->getDataType()));
            auto newOperand1 = castToUInt(inst->getOperand(1));
            inst = builder.replaceOperand(inst->getOperands(), newOperand0);
            inst = builder.replaceOperand(inst->getOperands() + 1, newOperand1);
        }
    }

    List<IRInst*> m_instsToRemove;
    void processWorkList()
    {
        while (workList.getCount() != 0)
        {
            IRInst* inst = workList.getLast();
            workList.removeLast();

            // Skip if inst has already been removed.
            if (!inst->parent)
                continue;

            switch (inst->getOp())
            {
            case kIROp_StructField:
                processStructField(as<IRStructField>(inst));
                break;
            case kIROp_GlobalParam:
                processGlobalParam(as<IRGlobalParam>(inst));
                break;
            case kIROp_GlobalVar:
                processGlobalVar(as<IRGlobalVar>(inst));
                break;
            case kIROp_Var:
                processVar(as<IRVar>(inst));
                break;
            case kIROp_Param:
                processParam(as<IRParam>(inst));
                break;
            case kIROp_Call:
                processCall(as<IRCall>(inst));
                break;
            case kIROp_GetElement:
                processGetElement(as<IRGetElement>(inst));
                break;
            case kIROp_GetElementPtr:
                processGetElementPtr(as<IRGetElementPtr>(inst));
                break;
            case kIROp_GetOffsetPtr:
                processGetOffsetPtr(inst);
                break;
            case kIROp_FieldAddress:
                processFieldAddress(as<IRFieldAddress>(inst));
                break;
            case kIROp_FieldExtract:
                processFieldExtract(as<IRFieldExtract>(inst));
                break;
            case kIROp_ImageSubscript:
                processImageSubscript(as<IRImageSubscript>(inst));
                break;
            case kIROp_RWStructuredBufferGetElementPtr:
                processRWStructuredBufferGetElementPtr(
                    cast<IRRWStructuredBufferGetElementPtr>(inst));
                break;
            case kIROp_MeshOutputRef:
                processMeshOutputGetElementPtr(cast<IRMeshOutputRef>(inst));
                break;
            case kIROp_MeshOutputSet:
                processMeshOutputSet(cast<IRMeshOutputSet>(inst));
                break;
            case kIROp_RWStructuredBufferLoad:
            case kIROp_StructuredBufferLoad:
            case kIROp_RWStructuredBufferLoadStatus:
            case kIROp_StructuredBufferLoadStatus:
                processStructuredBufferLoad(inst);
                break;
            case kIROp_RWStructuredBufferStore:
                processRWStructuredBufferStore(inst);
                break;
            case kIROp_NonUniformResourceIndex:
                processNonUniformResourceIndex(inst, NonUniformResourceIndexFloatMode::SPIRV);
                break;
            case kIROp_loop:
                processLoop(as<IRLoop>(inst));
                break;
            case kIROp_ifElse:
                processIfElse(as<IRIfElse>(inst));
                break;
            case kIROp_Switch:
                processSwitch(as<IRSwitch>(inst));
                break;
            case kIROp_Less:
            case kIROp_Leq:
            case kIROp_Eql:
            case kIROp_Geq:
            case kIROp_Greater:
            case kIROp_Neq:
                processComparison(inst);
                break;
            case kIROp_MakeVectorFromScalar:
            case kIROp_MakeUInt64:
            case kIROp_MakeVector:
            case kIROp_MakeMatrix:
            case kIROp_MakeMatrixFromScalar:
            case kIROp_MatrixReshape:
            case kIROp_MakeArray:
            case kIROp_MakeArrayFromElement:
            case kIROp_MakeStruct:
            case kIROp_MakeTuple:
            case kIROp_MakeTargetTuple:
            case kIROp_MakeResultValue:
            case kIROp_MakeResultError:
            case kIROp_MakeOptionalValue:
            case kIROp_MakeOptionalNone:
                processConstructor(inst);
                break;
            case kIROp_PtrLit:
                processPtrLit(inst);
                break;
            case kIROp_unconditionalBranch:
                processBranch(inst);
                break;
            case kIROp_SPIRVAsm:
                processSPIRVAsm(as<IRSPIRVAsm>(inst));
                break;
            case kIROp_DebugValue:
                if (!isSimpleDataType(as<IRDebugValue>(inst)->getDebugVar()->getDataType()))
                    inst->removeAndDeallocate();
                break;
            case kIROp_DebugVar:
                if (!isSimpleDataType(as<IRDebugVar>(inst)->getDataType()))
                {
                    inst->removeFromParent();
                    m_instsToRemove.add(inst);
                }
                break;
            case kIROp_Func:
                eliminateContinueBlocksInFunc(m_module, as<IRFunc>(inst));
                [[fallthrough]];
            default:
                for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
                {
                    addToWorkList(child);
                }
                break;
            }
        }
    }

    void determineSpirvVersion()
    {
        // Determine minimum spirv version from target request.
        auto targetCaps = m_sharedContext->m_targetProgram->getTargetReq()->getTargetCaps();
        for (auto targetAtomSet : targetCaps.getAtomSets())
        {
            for (auto atom : targetAtomSet)
            {
                auto spirvAtom = ((CapabilityName)atom);
                switch (spirvAtom)
                {
                case CapabilityName::_spirv_1_0:
                    m_sharedContext->requireSpirvVersion(0x10000);
                    break;
                case CapabilityName::_spirv_1_1:
                    m_sharedContext->requireSpirvVersion(0x10100);
                    break;
                case CapabilityName::_spirv_1_2:
                    m_sharedContext->requireSpirvVersion(0x10200);
                    break;
                case CapabilityName::_spirv_1_3:
                    m_sharedContext->requireSpirvVersion(0x10300);
                    break;
                case CapabilityName::_spirv_1_4:
                    m_sharedContext->requireSpirvVersion(0x10400);
                    break;
                case CapabilityName::_spirv_1_5:
                    m_sharedContext->requireSpirvVersion(0x10500);
                    break;
                case CapabilityName::_spirv_1_6:
                    m_sharedContext->requireSpirvVersion(0x10600);
                    break;
                case CapabilityName::SPV_EXT_demote_to_helper_invocation:
                    m_sharedContext->m_useDemoteToHelperInvocationExtension = true;
                    break;
                default:
                    break;
                }
            }
        }
        // Scan through the entry points and find the max version required.
        auto processInst = [&](IRInst* globalInst)
        {
            for (auto decor : globalInst->getDecorations())
            {
                switch (decor->getOp())
                {
                case kIROp_RequireCapabilityAtomDecoration:
                    {
                        auto atomDecor = as<IRRequireCapabilityAtomDecoration>(decor);
                        switch (atomDecor->getAtom())
                        {
                        case CapabilityName::spirv_1_3:
                            m_sharedContext->requireSpirvVersion(0X10300);
                            break;
                        case CapabilityName::spirv_1_4:
                            m_sharedContext->requireSpirvVersion(0X10400);
                            break;
                        case CapabilityName::spirv_1_5:
                            m_sharedContext->requireSpirvVersion(0X10500);
                            break;
                        case CapabilityName::spirv_1_6:
                            m_sharedContext->requireSpirvVersion(0X10600);
                            break;
                        }
                        break;
                    }
                }
            }
        };

        processInst(m_module->getModuleInst());
        for (auto globalInst : m_module->getGlobalInsts())
        {
            processInst(globalInst);
        }

        if (m_sharedContext->m_spvVersion < 0x10300)
        {
            // Direct SPIRV backend does not support generating SPIRV before 1.3,
            // we will issue an error message here.
            m_sharedContext->m_sink->diagnose(SourceLoc(), Diagnostics::spirvVersionNotSupported);
        }
    }

    struct SpirvAddressSpaceAssigner : InitialAddressSpaceAssigner
    {
        virtual bool tryAssignAddressSpace(IRInst* inst, AddressSpace& outAddressSpace) override
        {
            SLANG_UNUSED(inst);
            // Don't assign address space to additional insts, since we should have
            // already assigned address space to them in earlier stages of legalization.
            outAddressSpace = AddressSpace::Generic;
            return false;
        }

        virtual AddressSpace getAddressSpaceFromVarType(IRInst* type) override
        {
            if (auto ptrType = as<IRPtrTypeBase>(type))
            {
                if (ptrType->hasAddressSpace())
                    return ptrType->getAddressSpace();
            }
            return AddressSpace::Generic;
        }

        virtual AddressSpace getLeafInstAddressSpace(IRInst* inst) override
        {
            // Don't assign address space to additional insts, since we should have
            // already assigned address space to them in earlier stages of legalization.
            auto type = inst->getDataType();
            for (;;)
            {
                auto newType = (IRType*)unwrapAttributedType(type);
                newType = unwrapArray(newType);
                if (newType == type)
                    break;
                type = newType;
            }
            if (!type)
                return AddressSpace::Generic;
            return getAddressSpaceFromVarType(type);
        }
    };

    void propagateAddressAlignment()
    {
        // Work list of load/store insts to add Aligned attribute to.
        List<IRInst*> loadStoreInsts;

        for (auto globalInst : m_module->getGlobalInsts())
        {
            auto func = as<IRFunc>(globalInst);
            if (!func)
                continue;
            for (auto block : func->getBlocks())
            {
                for (auto inst : block->getChildren())
                {
                    switch (inst->getOp())
                    {
                    case kIROp_GetElementPtr:
                    case kIROp_FieldAddress:
                        {
                            auto base = inst->getOperand(0);
                            auto ptrType = as<IRPtrTypeBase>(base->getDataType());
                            if (!ptrType)
                                break;
                            // Propagate address alignment if possible.
                            auto alignDecor = base->findDecoration<IRAlignedAddressDecoration>();
                            if (!alignDecor)
                                break;
                            auto valueType = ptrType->getValueType();
                            auto layout = valueType->findDecoration<IRSizeAndAlignmentDecoration>();
                            if (!layout)
                                break;
                            auto alignment = getIntVal(alignDecor->getAlignment());
                            if (inst->getOp() == kIROp_GetElementPtr)
                            {
                                if (alignment >= layout->getAlignment())
                                {
                                    IRBuilder builder(inst);
                                    builder.addAlignedAddressDecoration(
                                        inst,
                                        alignDecor->getAlignment());
                                }
                            }
                            else
                            {
                                IRTypeLayoutRuleName layoutRuleName = layout->getLayoutName();
                                auto field = findStructField(
                                    valueType,
                                    (IRStructKey*)as<IRFieldAddress>(inst)->getField());
                                if (!field)
                                    break;
                                IRIntegerValue offset = 0;
                                if (getOffset(
                                        m_sharedContext->m_targetProgram->getOptionSet(),
                                        IRTypeLayoutRules::get(layoutRuleName),
                                        field,
                                        &offset) != SLANG_OK)
                                    break;
                                if (offset % alignment == 0)
                                {
                                    IRBuilder builder(inst);
                                    builder.addAlignedAddressDecoration(
                                        inst,
                                        alignDecor->getAlignment());
                                }
                            }
                        }
                        break;
                    case kIROp_Load:
                    case kIROp_Store:
                        {
                            if (inst->findAttr<IRAlignedAttr>())
                                break;
                            loadStoreInsts.add(inst);
                        }
                        break;
                    }
                }
            }
        }

        // Process the work list.
        // If load/store doesn't have Aligned attribute, and the ptr has
        // a IRAlignedAddress decoration, we should create a load/store
        // with a Aligned attribute.
        for (auto inst : loadStoreInsts)
        {

            if (auto load = as<IRLoad>(inst))
            {
                auto ptr = load->getPtr();
                if (auto decor = ptr->findDecoration<IRAlignedAddressDecoration>())
                {
                    IRBuilder builder(inst);
                    builder.setInsertBefore(inst);
                    auto newLoad =
                        builder.emitLoad(load->getFullType(), ptr, decor->getAlignment());
                    load->replaceUsesWith(newLoad);
                    load->removeAndDeallocate();
                }
            }
            else if (auto store = as<IRStore>(inst))
            {
                auto ptr = store->getPtr();
                if (auto decor = ptr->findDecoration<IRAlignedAddressDecoration>())
                {
                    IRBuilder builder(inst);
                    builder.setInsertBefore(inst);
                    builder.emitStore(ptr, store->getVal(), decor->getAlignment());
                    store->removeAndDeallocate();
                }
            }
        }
    }

    void processModule()
    {
        determineSpirvVersion();

        // Process global params before anything else, so we don't generate inefficient
        // array marhalling code for array-typed global params.
        for (auto globalInst : m_module->getGlobalInsts())
        {
            if (auto globalParam = as<IRGlobalParam>(globalInst))
            {
                processGlobalParam(globalParam);
            }
            else
            {
                addToWorkList(globalInst);
            }
        }
        processWorkList();

        for (auto inst : m_instsToRemove)
            inst->removeAndDeallocate();

        // Translate types.
        List<IRHLSLStructuredBufferTypeBase*> instsToProcess;
        List<IRInst*> textureFootprintTypes;

        for (auto globalInst : m_module->getGlobalInsts())
        {
            if (auto t = as<IRHLSLStructuredBufferTypeBase>(globalInst))
            {
                instsToProcess.add(t);
            }
            else if (globalInst->getOp() == kIROp_TextureFootprintType)
            {
                textureFootprintTypes.add(globalInst);
            }
        }
        for (auto t : instsToProcess)
        {
            auto lowered = lowerStructuredBufferType(t);
            IRBuilder builder(t);
            builder.setInsertBefore(t);
            t->replaceUsesWith(builder.getPtrType(
                kIROp_PtrType,
                lowered.structType,
                getStorageBufferAddressSpace()));
        }
        for (auto t : textureFootprintTypes)
        {
            auto lowered = lowerTextureFootprintType(t);
            IRBuilder builder(t);
            builder.setInsertBefore(t);
            t->replaceUsesWith(lowered);
        }

        for (auto globalInst : m_module->getGlobalInsts())
        {
            if (auto func = as<IRFunc>(globalInst))
            {
                if (auto entryPointDecor = func->findDecoration<IREntryPointDecoration>())
                {
                    legalizeSPIRVEntryPoint(func, entryPointDecor);
                }
                // SPIRV requires a dominator block to appear before dominated blocks.
                // After legalizing the control flow, we need to sort our blocks to ensure this
                // is true.
                sortBlocksInFunc(func);
            }
        }

        GlobalInstInliningContext().inlineGlobalValuesAndRemoveIfUnused(m_module);

        // Some legalization processing may change the function parameter types,
        // so we need to update the function types to match that.
        updateFunctionTypes();

        // Lower all loads/stores from buffer pointers to use correct storage types.
        // We didn't do the lowering for buffer pointers because we don't know which pointer
        // types are actual storage buffer pointers until we propagated the address space of
        // pointers in this pass. In the future we should consider separate out IRAddress as
        // the type for IRVar, and use IRPtrType to dedicate pointers in user code, so we can
        // safely lower the pointer load stores early together with other buffer types.
        BufferElementTypeLoweringOptions bufferElementTypeLoweringOptions;
        bufferElementTypeLoweringOptions.lowerBufferPointer = true;
        lowerBufferElementTypeToStorageType(
            m_sharedContext->m_targetProgram,
            m_module,
            bufferElementTypeLoweringOptions);

        // Inline all pack/unpack storage type functions generated during buffer element
        // lowering pass.
        performForceInlining(m_module);

        // The above step may produce empty struct types, so we need to lower them out of
        // existence.
        legalizeEmptyTypes(m_sharedContext->m_targetProgram, m_module, m_sink);

        // Propagate alignment hints on address instructions.
        propagateAddressAlignment();

        // Specalize address space for all pointers.
        SpirvAddressSpaceAssigner addressSpaceAssigner;
        specializeAddressSpace(m_module, &addressSpaceAssigner);

        // For SPIR-V, we don't skip this validation, because we might then be generating
        // invalid SPIR-V.
        bool skipFuncParamValidation = false;
        validateAtomicOperations(skipFuncParamValidation, m_sink, m_module->getModuleInst());
    }

    void updateFunctionTypes()
    {
        IRBuilder builder(m_module);
        for (auto globalInst : m_module->getGlobalInsts())
        {
            auto func = as<IRFunc>(globalInst);
            if (!func)
                continue;
            auto firstBlock = func->getFirstBlock();
            if (!firstBlock)
                continue;

            builder.setInsertBefore(func);
            auto type = func->getDataType();
            auto oldFuncType = as<IRFuncType>(type);
            auto resultType = oldFuncType->getResultType();
            List<IRType*> newOperands;
            for (auto block : func->getBlocks())
            {
                for (auto inst : block->getChildren())
                {
                    if (auto retInst = as<IRReturn>(inst))
                    {
                        resultType = retInst->getVal()->getFullType();
                        break;
                    }
                }
            }
            for (auto param : firstBlock->getParams())
            {
                newOperands.add(param->getDataType());
            }
            bool changed = resultType != oldFuncType->getResultType();
            if (!changed)
            {
                for (UInt i = 0; i < oldFuncType->getParamCount(); i++)
                {
                    if (oldFuncType->getParamType(i) != newOperands[i])
                    {
                        changed = true;
                        break;
                    }
                }
            }
            if (changed)
            {
                builder.setInsertBefore(func);
                auto newFuncType = builder.getFuncType(newOperands, resultType);
                func->setFullType(newFuncType);
            }
        }
    }
};

SpvSnippet* SPIRVEmitSharedContext::getParsedSpvSnippet(IRTargetIntrinsicDecoration* intrinsic)
{
    RefPtr<SpvSnippet> snippet;
    if (m_parsedSpvSnippets.tryGetValue(intrinsic, snippet))
    {
        return snippet.Ptr();
    }
    snippet = SpvSnippet::parse(*m_grammarInfo, intrinsic->getDefinition());
    if (!snippet)
    {
        m_sink->diagnose(intrinsic, Diagnostics::snippetParsingFailed, intrinsic->getDefinition());
        return nullptr;
    }
    m_parsedSpvSnippets[intrinsic] = snippet;
    return snippet;
}

void legalizeSPIRV(SPIRVEmitSharedContext* sharedContext, IRModule* module, DiagnosticSink* sink)
{
    SPIRVLegalizationContext context(sharedContext, module, sink);
    context.processModule();
}

void simplifyIRForSpirvLegalization(TargetProgram* target, DiagnosticSink* sink, IRModule* module)
{
    bool changed = true;
    const int kMaxIterations = 8;
    const int kMaxFuncIterations = 16;
    int iterationCounter = 0;

    while (changed && iterationCounter < kMaxIterations)
    {
        if (sink && sink->getErrorCount())
            break;

        changed = false;

        changed |= applySparseConditionalConstantPropagationForGlobalScope(module, sink);
        changed |= peepholeOptimizeGlobalScope(target, module);

        for (auto inst : module->getGlobalInsts())
        {
            auto func = as<IRGlobalValueWithCode>(inst);
            if (!func)
                continue;
            bool funcChanged = true;
            int funcIterationCount = 0;
            while (funcChanged && funcIterationCount < kMaxFuncIterations)
            {
                funcChanged = false;
                funcChanged |= applySparseConditionalConstantPropagation(func, sink);
                funcChanged |= peepholeOptimize(target, func);
                funcChanged |= removeRedundancyInFunc(func, false);
                CFGSimplificationOptions options;
                options.removeTrivialSingleIterationLoops = true;
                options.removeSideEffectFreeLoops = false;
                funcChanged |= simplifyCFG(func, options);
                eliminateDeadCode(func);
            }
        }
    }
}

static bool isRasterOrderedResource(IRInst* inst)
{
    if (auto memoryQualifierDecoration = inst->findDecoration<IRMemoryQualifierSetDecoration>())
    {
        if (memoryQualifierDecoration->getMemoryQualifierBit() &
            MemoryQualifierSetModifier::Flags::kRasterizerOrdered)
            return true;
    }
    auto type = inst->getDataType();
    for (;;)
    {
        if (auto ptrType = as<IRPtrTypeBase>(type))
        {
            type = ptrType->getValueType();
            continue;
        }
        if (auto arrayType = as<IRArrayTypeBase>(type))
        {
            type = arrayType->getElementType();
            continue;
        }
        break;
    }
    if (auto textureType = as<IRTextureTypeBase>(type))
    {
        if (textureType->getAccess() == SLANG_RESOURCE_ACCESS_RASTER_ORDERED)
            return true;
    }
    return false;
}

static bool hasExplicitInterlockInst(IRFunc* func)
{
    for (auto block : func->getBlocks())
    {
        for (auto inst : block->getChildren())
        {
            if (inst->getOp() == kIROp_BeginFragmentShaderInterlock)
                return true;
        }
    }
    return false;
}

void insertFragmentShaderInterlock(SPIRVEmitSharedContext* context, IRModule* module)
{
    HashSet<IRFunc*> fragmentShaders;
    for (auto& [inst, entryPoints] : context->m_referencingEntryPoints)
    {
        if (isRasterOrderedResource(inst))
        {
            for (auto entryPoint : entryPoints)
            {
                auto entryPointDecor = entryPoint->findDecoration<IREntryPointDecoration>();
                if (!entryPointDecor)
                    continue;

                if (entryPointDecor->getProfile().getStage() == Stage::Fragment)
                {
                    fragmentShaders.add(entryPoint);
                }
            }
        }
    }

    IRBuilder builder(module);
    for (auto entryPoint : fragmentShaders)
    {
        if (hasExplicitInterlockInst(entryPoint))
            continue;
        auto firstBlock = entryPoint->getFirstBlock();
        if (!firstBlock)
            continue;
        builder.setInsertBefore(firstBlock->getFirstOrdinaryInst());
        builder.emitBeginFragmentShaderInterlock();
        for (auto block : entryPoint->getBlocks())
        {
            if (auto inst = block->getTerminator())
            {
                if (inst->getOp() == kIROp_Return ||
                    !context->isSpirv16OrLater() && inst->getOp() == kIROp_discard)
                {
                    builder.setInsertBefore(inst);
                    builder.emitEndFragmentShaderInterlock();
                }
            }
        }
    }
}

void legalizeIRForSPIRV(
    SPIRVEmitSharedContext* context,
    IRModule* module,
    const List<IRFunc*>& entryPoints,
    CodeGenContext* codeGenContext)
{
    SLANG_UNUSED(entryPoints);
    legalizeSPIRV(context, module, codeGenContext->getSink());
    simplifyIRForSpirvLegalization(context->m_targetProgram, codeGenContext->getSink(), module);
    buildEntryPointReferenceGraph(context->m_referencingEntryPoints, module);
    insertFragmentShaderInterlock(context, module);
}

} // namespace Slang
