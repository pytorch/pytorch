// slang-ir-single-return.cpp
#include "slang-ir-single-return.h"

#include "slang-ir-clone.h"
#include "slang-ir-eliminate-multilevel-break.h"
#include "slang-ir-inst-pass-base.h"
#include "slang-ir-insts.h"
#include "slang-ir-simplify-cfg.h"
#include "slang-ir.h"

namespace Slang
{

struct SingleReturnContext : public InstPassBase
{
    SingleReturnContext(IRModule* inModule)
        : InstPassBase(inModule)
    {
    }
    void processFunc(IRGlobalValueWithCode* func)
    {
        IRBuilder builder(module);
        simplifyCFG(func, CFGSimplificationOptions::getFast());

        // We make use of the `eliminate-multi-level-break` pass to implement the transformation.
        // To be able to do that, we need to prepare `func` so that the entire function body
        // is wrapped in a trivial loop and turn all `return`s into `break`s out of the outter most
        // loop.
        builder.setInsertInto(func);
        auto breakBlock = builder.emitBlock();
        auto returnBlock = builder.emitBlock();
        builder.setInsertInto(breakBlock);
        auto resultType = as<IRFuncType>(func->getDataType())->getResultType();

        IRInst* retValParam = nullptr;
        if (resultType->getOp() != kIROp_VoidType)
        {
            retValParam = builder.emitParam(resultType);
        }
        builder.emitBranch(returnBlock);

        auto originalStartBlock = func->getFirstBlock();
        auto loopHeaderBlock = builder.createBlock();
        loopHeaderBlock->insertBefore(originalStartBlock);
        builder.setInsertInto(loopHeaderBlock);

        // Move all params into `loopHeaderBlock`.
        List<IRParam*> params;
        for (auto param : originalStartBlock->getParams())
        {
            params.add(param);
        }
        for (auto param : params)
        {
            loopHeaderBlock->addParam(param);
        }

        builder.emitLoop(originalStartBlock, breakBlock, originalStartBlock);

        // Now replace all return insts as break insts.
        processChildInstsOfType<IRReturn>(
            kIROp_Return,
            func,
            [&](IRReturn* returnInst)
            {
                IRInst* retVal = nullptr;
                if (returnInst->getOperandCount() == 0)
                    retVal = builder.getVoidValue();
                else
                    retVal = returnInst->getVal();
                builder.setInsertBefore(returnInst);
                if (resultType->getOp() == kIROp_VoidType)
                {
                    builder.emitBranch(breakBlock);
                }
                else
                {
                    builder.emitBranch(breakBlock, 1, &retVal);
                }
                returnInst->removeAndDeallocate();
            });

        builder.setInsertInto(returnBlock);
        if (retValParam)
            builder.emitReturn(retValParam);
        else
            builder.emitReturn();
    }
};

void convertFuncToSingleReturnForm(IRModule* irModule, IRGlobalValueWithCode* func)
{
    SingleReturnContext context(irModule);
    context.processFunc(func);
}

int getReturnCount(IRGlobalValueWithCode* func)
{
    int returnCount = 0;
    for (auto block : func->getBlocks())
    {
        for (auto inst : block->getChildren())
        {
            if (inst->getOp() == kIROp_Return)
            {
                returnCount++;
            }
        }
    }
    return returnCount;
}

bool isSingleReturnFunc(IRGlobalValueWithCode* func)
{
    return getReturnCount(func) == 1;
}

} // namespace Slang
