#include "slang-ir-lower-dynamic-resource-heap.h"

#include "compiler-core/slang-artifact-associated-impl.h"
#include "slang-ir-util.h"

namespace Slang
{
UInt findUnusedSpaceIndex(TargetProgram* targetProgram, IRModule* module)
{
    HashSet<int> usedSpaces;
    auto processVarLayout = [&](IRVarLayout* varLayout)
    {
        UInt spaceOffset = 0;
        if (auto spaceAttr = varLayout->findOffsetAttr(LayoutResourceKind::RegisterSpace))
        {
            spaceOffset = spaceAttr->getOffset();
        }
        for (auto sizeAttr : varLayout->getTypeLayout()->getSizeAttrs())
        {
            auto kind = sizeAttr->getResourceKind();
            if (!ShaderBindingRange::isUsageTracked(kind))
                continue;

            if (auto offsetAttr = varLayout->findOffsetAttr(kind))
            {
                // Get the binding information from this attribute and insert it into the list
                auto spaceIndex = spaceOffset + offsetAttr->getSpace();
                usedSpaces.add((int)spaceIndex);
            }
        }
    };

    for (auto inst : module->getGlobalInsts())
    {
        if (as<IRGlobalParam>(inst))
        {
            auto varLayout = findVarLayout(inst);
            if (!varLayout)
                continue;
            processVarLayout(varLayout);

            auto paramGroupTypeLayout = as<IRParameterGroupTypeLayout>(varLayout->getTypeLayout());
            if (!paramGroupTypeLayout)
                continue;
            auto containerVarLayout = paramGroupTypeLayout->getContainerVarLayout();
            if (!containerVarLayout)
                continue;
            processVarLayout(containerVarLayout);
        }
    }

    // Find next unused space index.
    int index = targetProgram->getOptionSet().getIntOption(CompilerOptionName::BindlessSpaceIndex);
    while (usedSpaces.contains(index))
    {
        index++;
    }
    return index;
}

IRVarLayout* createResourceHeapVarLayoutWithSpace(
    IRBuilder& builder,
    IRInst* param,
    UInt spaceIndex)
{
    SLANG_UNUSED(param);
    IRTypeLayout::Builder typeLayoutBuilder(&builder);
    typeLayoutBuilder.addResourceUsage(
        LayoutResourceKind::DescriptorTableSlot,
        LayoutSize::infinite());
    auto typeLayout = typeLayoutBuilder.build();
    IRVarLayout::Builder varLayoutBuilder(&builder, typeLayout);
    varLayoutBuilder.findOrAddResourceInfo(LayoutResourceKind::RegisterSpace)->offset = spaceIndex;
    varLayoutBuilder.findOrAddResourceInfo(LayoutResourceKind::DescriptorTableSlot)->offset = 0;
    return varLayoutBuilder.build();
}

void lowerDynamicResourceHeap(TargetProgram* targetProgram, IRModule* module, DiagnosticSink* sink)
{
    SLANG_UNUSED(sink);
    auto unusedSpaceIndex = findUnusedSpaceIndex(targetProgram, module);
    List<IRInst*> workList;
    for (auto globalInst : module->getGlobalInsts())
    {
        if (globalInst->getOp() == kIROp_GetDynamicResourceHeap)
        {
            workList.add(globalInst);
        }
    }
    for (auto inst : workList)
    {
        auto arrayType = as<IRArrayTypeBase>(inst->getDataType());
        IRBuilder builder(inst);
        builder.setInsertBefore(inst);

        auto param = builder.createGlobalParam(arrayType);
        auto varLayout = createResourceHeapVarLayoutWithSpace(builder, param, unusedSpaceIndex);
        builder.addLayoutDecoration(param, varLayout);
        builder.addNameHintDecoration(param, toSlice("__slang_resource_heap"));
        inst->replaceUsesWith(param);
    }
}

} // namespace Slang
