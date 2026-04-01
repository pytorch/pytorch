#include "slang-ir-vk-invert-y.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

static IRInst* _invertYOfVector(IRBuilder& builder, IRInst* originalVector)
{
    auto vectorType = as<IRVectorType>(originalVector->getDataType());
    SLANG_ASSERT(vectorType);
    UInt elementIndexY = 1;
    auto originalY =
        builder.emitSwizzle(vectorType->getElementType(), originalVector, 1, &elementIndexY);
    auto negY = builder.emitNeg(originalY->getDataType(), originalY);
    auto newVal =
        builder
            .emitSwizzleSet(originalVector->getDataType(), originalVector, negY, 1, &elementIndexY);
    return newVal;
}
// Find outputs to SV_Position and invert the y coordinates of it right before the write.
void invertYOfPositionOutput(IRModule* module)
{
    for (auto globalInst : module->getGlobalInsts())
    {
        if (globalInst->findDecoration<IRGLPositionOutputDecoration>())
        {
            // Find all loads and stores to it.
            IRBuilder builder(module);
            List<IRUse*> useWorkList;
            auto processUse = [&](IRUse* use)
            {
                if (auto store = as<IRStore>(use->getUser()))
                {
                    if (getRootAddr(store->getPtr()) != globalInst)
                        return;

                    builder.setInsertBefore(store);
                    auto originalVal = store->getVal();
                    auto invertedVal = _invertYOfVector(builder, originalVal);
                    builder.replaceOperand(&store->val, invertedVal);
                }
                else if (auto load = as<IRLoad>(use->getUser()))
                {
                    // Since we negate the y coordinate before writing
                    // to gl_Position, we also need to negate the value after reading from it.
                    builder.setInsertAfter(load);
                    // Store existing uses of the load that we are going to replace with
                    // inverted val later.
                    List<IRUse*> oldUses;
                    for (auto loadUse = load->firstUse; loadUse; loadUse = loadUse->nextUse)
                        oldUses.add(loadUse);
                    // Get the inverted vector.
                    auto invertedVal = _invertYOfVector(builder, load);
                    // Replace original uses with the invertex vector.
                    for (auto loadUse : oldUses)
                        builder.replaceOperand(loadUse, invertedVal);
                }
                else if (auto getElementPtr = as<IRGetElementPtr>(use->getUser()))
                {
                    traverseUses(getElementPtr, [&](IRUse* use) { useWorkList.add(use); });
                }
            };
            traverseUses(globalInst, processUse);
            for (Index i = 0; i < useWorkList.getCount(); i++)
            {
                processUse(useWorkList[i]);
            }
        }
    }
}


static IRInst* _invertWOfVector(IRBuilder& builder, IRInst* originalVector)
{
    auto vectorType = as<IRVectorType>(originalVector->getDataType());
    SLANG_ASSERT(vectorType);
    UInt elementIndexW = 3;
    auto originalW =
        builder.emitSwizzle(vectorType->getElementType(), originalVector, 1, &elementIndexW);
    auto rcpW = builder.emitDiv(
        originalW->getDataType(),
        builder.getFloatValue(originalW->getDataType(), 1.0),
        originalW);
    auto newVal =
        builder
            .emitSwizzleSet(originalVector->getDataType(), originalVector, rcpW, 1, &elementIndexW);
    return newVal;
}

// Find inputs of SV_Position and rcp the w coordinates of it right after the read.
void rcpWOfPositionInput(IRModule* module)
{
    for (auto globalInst : module->getGlobalInsts())
    {
        if (globalInst->findDecoration<IRGLPositionInputDecoration>())
        {
            // Find all loads and replace them with reciprocals.
            IRBuilder builder(module);
            traverseUses(
                globalInst,
                [&](IRUse* use)
                {
                    // Get the inverted vector.
                    auto user = use->getUser();
                    if (user->getOp() == kIROp_Load)
                    {
                        builder.setInsertBefore(user);
                        auto val = builder.emitLoad(globalInst);
                        auto invertedVal = _invertWOfVector(builder, val);
                        user->replaceUsesWith(invertedVal);
                        user->removeAndDeallocate();
                    }
                });
        }
    }
}
} // namespace Slang
