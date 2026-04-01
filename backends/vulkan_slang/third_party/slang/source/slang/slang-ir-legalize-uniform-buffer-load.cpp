#include "slang-ir-legalize-uniform-buffer-load.h"

namespace Slang
{
void legalizeUniformBufferLoad(IRModule* module)
{
    List<IRLoad*> workList;
    for (auto globalInst : module->getGlobalInsts())
    {
        if (auto func = as<IRGlobalValueWithCode>(globalInst))
        {
            for (auto block : func->getBlocks())
            {
                for (auto inst : block->getChildren())
                {
                    if (auto load = as<IRLoad>(inst))
                    {
                        auto uniformBufferType =
                            as<IRConstantBufferType>(load->getPtr()->getDataType());
                        if (!uniformBufferType)
                            continue;
                        workList.add(load);
                    }
                }
            }
        }
    }

    IRBuilder builder(module);
    for (auto load : workList)
    {
        auto uniformBufferType = as<IRConstantBufferType>(load->getPtr()->getDataType());
        SLANG_ASSERT(uniformBufferType);
        auto structType = as<IRStructType>(uniformBufferType->getElementType());
        if (!structType)
            continue;
        builder.setInsertBefore(load);
        List<IRInst*> fieldLoads;
        for (auto field : structType->getFields())
        {
            auto fieldAddr = builder.emitFieldAddress(
                builder.getPtrType(field->getFieldType()),
                load->getPtr(),
                field->getKey());
            auto fieldLoad = builder.emitLoad(fieldAddr);
            fieldLoads.add(fieldLoad);
        }
        auto makeStruct = builder.emitMakeStruct(structType, fieldLoads);
        load->replaceUsesWith(makeStruct);
    }
}

} // namespace Slang
