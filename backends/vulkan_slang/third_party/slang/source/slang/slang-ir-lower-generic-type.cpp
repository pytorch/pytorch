// slang-ir-lower-generic-type.cpp
#include "slang-ir-lower-generic-type.h"

#include "slang-ir-clone.h"
#include "slang-ir-generics-lowering-context.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{
// This is a subpass of generics lowering IR transformation.
// This pass lowers all generic/polymorphic types into IRAnyValueType.
struct GenericTypeLoweringContext
{
    SharedGenericsLoweringContext* sharedContext;

    IRInst* processInst(IRInst* inst)
    {
        // Ensure exported struct types has RTTI object defined.
        if (as<IRStructType>(inst))
        {
            if (inst->findDecoration<IRHLSLExportDecoration>())
            {
                sharedContext->maybeEmitRTTIObject(inst);
            }
        }

        // Don't modify type insts themselves.
        if (as<IRType>(inst))
            return inst;

        IRBuilder builderStorage(sharedContext->module);
        auto builder = &builderStorage;
        builder->setInsertBefore(inst);

        auto newType = sharedContext->lowerType(builder, inst->getFullType());
        if (newType != inst->getFullType())
            inst = builder->replaceOperand(&inst->typeUse, newType);

        switch (inst->getOp())
        {
        default:
            break;
        case kIROp_StructField:
            {
                // Translate the struct field type.
                auto structField = static_cast<IRStructField*>(inst);
                auto loweredFieldType =
                    sharedContext->lowerType(builder, structField->getFieldType());
                structField->setOperand(1, loweredFieldType);
            }
            break;
        }
        return inst;
    }

    void processModule()
    {
        sharedContext->addToWorkList(sharedContext->module->getModuleInst());

        while (sharedContext->workList.getCount() != 0)
        {
            IRInst* inst = sharedContext->workList.getLast();

            sharedContext->workList.removeLast();
            sharedContext->workListSet.remove(inst);

            inst = processInst(inst);

            for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
            {
                sharedContext->addToWorkList(child);
            }
        }
        sharedContext->mapInterfaceRequirementKeyValue.clear();
    }
};

void lowerGenericType(SharedGenericsLoweringContext* sharedContext)
{
    GenericTypeLoweringContext context;
    context.sharedContext = sharedContext;
    context.processModule();
}
} // namespace Slang
