#include "slang-ir-lower-combined-texture-sampler.h"

#include "slang-ir-insts.h"
#include "slang-ir-layout.h"
#include "slang-ir-util.h"

namespace Slang
{
struct LoweredCombinedSamplerStructInfo
{
    IRStructKey* texture;
    IRStructKey* sampler;
    IRStructType* type;
    IRType* samplerType;
    IRType* textureType;
    IRTypeLayout* typeLayout;
};

struct LowerCombinedSamplerContext
{
    Dictionary<IRType*, LoweredCombinedSamplerStructInfo> mapTypeToLoweredInfo;
    Dictionary<IRType*, LoweredCombinedSamplerStructInfo> mapLoweredTypeToLoweredInfo;
    CodeGenTarget codeGenTarget;

    LoweredCombinedSamplerStructInfo lowerCombinedTextureSamplerType(IRTextureTypeBase* textureType)
    {
        if (auto loweredInfo = mapTypeToLoweredInfo.tryGetValue(textureType))
            return *loweredInfo;
        LoweredCombinedSamplerStructInfo info;
        IRBuilder builder(textureType);
        builder.setInsertBefore(textureType);
        auto structType = builder.createStructType();
        StringBuilder sb;
        getTypeNameHint(sb, textureType);
        builder.addNameHintDecoration(structType, sb.getUnownedSlice());
        info.sampler = builder.createStructKey();
        builder.addNameHintDecoration(info.sampler, toSlice("sampler"));
        info.texture = builder.createStructKey();
        builder.addNameHintDecoration(info.texture, toSlice("texture"));
        info.type = structType;

        bool isMutable =
            getIntVal(textureType->getAccessInst()) == kCoreModule_ResourceAccessReadOnly ? false
                                                                                          : true;

        info.textureType = builder.getTextureType(
            textureType->getElementType(),
            textureType->getShapeInst(),
            textureType->getIsArrayInst(),
            textureType->getIsMultisampleInst(),
            textureType->getSampleCountInst(),
            textureType->getAccessInst(),
            textureType->getIsShadowInst(),
            builder.getIntValue(builder.getIntType(), 0),
            textureType->getFormatInst());
        builder.createStructField(structType, info.texture, info.textureType);

        if (getIntVal(textureType->getIsShadowInst()) != 0)
            info.samplerType = builder.getType(kIROp_SamplerComparisonStateType);
        else
            info.samplerType = builder.getType(kIROp_SamplerStateType);
        builder.createStructField(structType, info.sampler, info.samplerType);

        // Type layout.

        bool isWGSLTarget = codeGenTarget == CodeGenTarget::WGSL;
        LayoutResourceKind textureResourceKind =
            isMutable ? LayoutResourceKind::UnorderedAccess : LayoutResourceKind::ShaderResource;
        LayoutResourceKind samplerResourceKind = LayoutResourceKind::SamplerState;
        if (isWGSLTarget)
        {
            textureResourceKind = LayoutResourceKind::DescriptorTableSlot;
            samplerResourceKind = LayoutResourceKind::DescriptorTableSlot;
        }

        IRTypeLayout::Builder textureTypeLayoutBuilder(&builder);
        textureTypeLayoutBuilder.addResourceUsage(textureResourceKind, LayoutSize(1));
        auto textureTypeLayout = textureTypeLayoutBuilder.build();

        IRTypeLayout::Builder samplerTypeLayoutBuilder(&builder);
        samplerTypeLayoutBuilder.addResourceUsage(samplerResourceKind, LayoutSize(1));
        auto samplerTypeLayout = samplerTypeLayoutBuilder.build();

        IRVarLayout::Builder textureVarLayoutBuilder(&builder, textureTypeLayout);
        textureVarLayoutBuilder.findOrAddResourceInfo(textureResourceKind)->offset = 0;
        auto textureVarLayout = textureVarLayoutBuilder.build();

        IRVarLayout::Builder samplerVarLayoutBuilder(&builder, samplerTypeLayout);
        samplerVarLayoutBuilder.findOrAddResourceInfo(samplerResourceKind)->offset =
            isWGSLTarget ? 1 : 0;
        auto samplerVarLayout = samplerVarLayoutBuilder.build();

        IRStructTypeLayout::Builder layoutBuilder(&builder);
        layoutBuilder.addField(info.texture, textureVarLayout);
        layoutBuilder.addField(info.sampler, samplerVarLayout);
        info.typeLayout = layoutBuilder.build();
        builder.addLayoutDecoration(structType, info.typeLayout);

        mapTypeToLoweredInfo.add(textureType, info);
        mapLoweredTypeToLoweredInfo.add(info.type, info);
        return info;
    }
};

void lowerCombinedTextureSamplers(
    CodeGenContext* codeGenContext,
    IRModule* module,
    DiagnosticSink* sink)
{
    SLANG_UNUSED(sink);

    LowerCombinedSamplerContext context;
    context.codeGenTarget = codeGenContext->getTargetFormat();

    // Lower combined texture sampler type into a struct type.
    for (auto globalInst : module->getGlobalInsts())
    {
        auto textureType = as<IRTextureTypeBase>(globalInst);
        if (!textureType || getIntVal(textureType->getIsCombinedInst()) == 0)
            continue;
        auto typeInfo = context.lowerCombinedTextureSamplerType(textureType);

        for (auto use = textureType->firstUse; use; use = use->nextUse)
        {
            auto typeUser = use->getUser();
            if (use != &typeUser->typeUse)
                continue;

            auto layoutDecor = typeUser->findDecoration<IRLayoutDecoration>();
            if (!layoutDecor)
                continue;
            // Replace the original VarLayout with the new StructTypeVarLayout.
            auto varLayout = as<IRVarLayout>(layoutDecor->getLayout());
            if (!varLayout)
                continue;
            IRBuilder subBuilder(typeUser);
            IRVarLayout::Builder newVarLayoutBuilder(&subBuilder, typeInfo.typeLayout);
            newVarLayoutBuilder.cloneEverythingButOffsetsFrom(varLayout);
            IRVarOffsetAttr* resOffsetAttr = nullptr;
            IRVarOffsetAttr* descriptorTableSlotOffsetAttr = nullptr;

            for (auto offsetAttr : varLayout->getOffsetAttrs())
            {
                LayoutResourceKind resKind = offsetAttr->getResourceKind();
                if (resKind == LayoutResourceKind::UnorderedAccess ||
                    resKind == LayoutResourceKind::ShaderResource)
                    resOffsetAttr = offsetAttr;
                else if (resKind == LayoutResourceKind::DescriptorTableSlot)
                    descriptorTableSlotOffsetAttr = offsetAttr;
                auto info = newVarLayoutBuilder.findOrAddResourceInfo(resKind);
                info->offset = offsetAttr->getOffset();
                info->space = offsetAttr->getSpace();
                info->kind = offsetAttr->getResourceKind();
            }
            // If the user provided an layout offset for the texture but not for descriptor table
            // slot, then we use the texture offset for the descriptor table slot offset.
            if (resOffsetAttr && !descriptorTableSlotOffsetAttr)
            {
                auto info = newVarLayoutBuilder.findOrAddResourceInfo(
                    LayoutResourceKind::DescriptorTableSlot);
                info->offset = resOffsetAttr->getOffset();
                info->space = resOffsetAttr->getSpace();
                info->kind = LayoutResourceKind::DescriptorTableSlot;
            }
            auto newVarLayout = newVarLayoutBuilder.build();
            subBuilder.addLayoutDecoration(typeUser, newVarLayout);
            varLayout->removeAndDeallocate();
        }
    }

    // If no combined texture sampler type exist in the IR module,
    // we can exit now.
    if (context.mapTypeToLoweredInfo.getCount() == 0)
        return;

    // We need to process all insts in the module, and replace
    // CombinedTextureSamplerGetTexture and CombinedTextureSamplerGetSampler into
    // FieldExtracts.
    IRBuilder builder(module);
    for (auto globalInst : module->getGlobalInsts())
    {
        auto func = as<IRFunc>(getGenericReturnVal(globalInst));
        if (!func)
            continue;
        for (auto block : func->getBlocks())
        {
            for (auto inst : block->getModifiableChildren())
            {
                switch (inst->getOp())
                {
                case kIROp_CombinedTextureSamplerGetTexture:
                case kIROp_CombinedTextureSamplerGetSampler:
                    {
                        auto combinedSamplerType = inst->getOperand(0)->getDataType();
                        auto loweredInfo =
                            context.mapTypeToLoweredInfo.tryGetValue(combinedSamplerType);
                        if (!loweredInfo)
                            loweredInfo = context.mapLoweredTypeToLoweredInfo.tryGetValue(
                                combinedSamplerType);
                        if (!loweredInfo)
                            continue;
                        builder.setInsertBefore(inst);
                        auto fieldExtract = builder.emitFieldExtract(
                            inst->getFullType(),
                            inst->getOperand(0),
                            inst->getOp() == kIROp_CombinedTextureSamplerGetSampler
                                ? loweredInfo->sampler
                                : loweredInfo->texture);
                        inst->replaceUsesWith(fieldExtract);
                        inst->removeAndDeallocate();
                    }
                    break;
                case kIROp_CastDescriptorHandleToResource:
                    {
                        auto handle = inst->getOperand(0);
                        if (as<IRDescriptorHandleType>(handle->getDataType()))
                        {
                            // If handle is still a DescriptorHandle, we are on a target that
                            // where native resource handles are already bindless, e.g. metal.
                            // On these platforms, the handle is a struct containing texture
                            // and sampler fields, so we just need to insert the extract operations.
                            auto combinedSamplerType = inst->getDataType();
                            auto loweredInfo =
                                context.mapTypeToLoweredInfo.tryGetValue(combinedSamplerType);
                            if (!loweredInfo)
                                continue;
                            builder.setInsertBefore(inst);
                            auto textureVal = builder.emitFieldExtract(
                                loweredInfo->textureType,
                                handle,
                                loweredInfo->texture);
                            auto samplerVal = builder.emitFieldExtract(
                                loweredInfo->samplerType,
                                handle,
                                loweredInfo->sampler);
                            IRInst* args[] = {textureVal, samplerVal};
                            auto combinedSampler =
                                builder.emitMakeStruct(loweredInfo->type, 2, args);
                            inst->replaceUsesWith(combinedSampler);
                            inst->removeAndDeallocate();
                        }
                    }
                    break;

                case kIROp_MakeCombinedTextureSamplerFromHandle:
                    {
                        auto combinedSamplerType = inst->getDataType();
                        auto loweredInfo =
                            context.mapTypeToLoweredInfo.tryGetValue(combinedSamplerType);
                        if (!loweredInfo)
                            continue;
                        auto handle = inst->getOperand(0);
                        builder.setInsertBefore(inst);
                        auto textureIndex = builder.emitElementExtract(handle, IRIntegerValue(0));
                        auto texture = builder.emitIntrinsicInst(
                            loweredInfo->textureType,
                            kIROp_LoadResourceDescriptorFromHeap,
                            1,
                            &textureIndex);
                        auto samplerIndex = builder.emitElementExtract(handle, IRIntegerValue(1));
                        auto sampler = builder.emitIntrinsicInst(
                            loweredInfo->samplerType,
                            kIROp_LoadSamplerDescriptorFromHeap,
                            1,
                            &samplerIndex);
                        IRInst* args[] = {texture, sampler};
                        auto combinedSampler = builder.emitMakeStruct(loweredInfo->type, 2, args);
                        inst->replaceUsesWith(combinedSampler);
                        inst->removeAndDeallocate();
                    }
                    break;
                }
            }
        }
    }

    // Replace all other type use with the lowered struct type.
    for (auto typeInfo : context.mapTypeToLoweredInfo)
    {
        auto loweredInfo = typeInfo.second;
        typeInfo.first->replaceUsesWith(loweredInfo.type);
        typeInfo.first->removeAndDeallocate();
    }
}

} // namespace Slang
