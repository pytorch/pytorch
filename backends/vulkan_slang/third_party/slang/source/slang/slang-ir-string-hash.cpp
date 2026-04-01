// slang-ir-string-hash.cpp
#include "slang-ir-string-hash.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

static void _findGetStringHashRec(IRInst* inst, List<IRGetStringHash*>& outInsts)
{
    for (IRInst* child = inst->getFirstDecorationOrChild(); child; child = child->getNextInst())
    {
        if (IRGetStringHash* getInst = as<IRGetStringHash>(child))
        {
            outInsts.add(getInst);
        }
        _findGetStringHashRec(child, outInsts);
    }
}

void findGetStringHashInsts(IRModule* module, List<IRGetStringHash*>& outInsts)
{
    _findGetStringHashRec(module->getModuleInst(), outInsts);
}

void findGlobalHashedStringLiterals(IRModule* module, StringSlicePool& pool)
{
    IRModuleInst* moduleInst = module->getModuleInst();

    for (IRInst* child : moduleInst->getChildren())
    {
        if (IRGlobalHashedStringLiterals* hashedStringLits =
                as<IRGlobalHashedStringLiterals>(child))
        {
            const Index count = hashedStringLits->getOperandCount();
            for (Index i = 0; i < count; ++i)
            {
                IRStringLit* stringLit = as<IRStringLit>(hashedStringLits->getOperand(i));
                pool.add(stringLit->getStringSlice());
            }
        }
    }
}

void addGlobalHashedStringLiterals(const StringSlicePool& pool, IRModule* module)
{
    auto slices = pool.getAdded();
    if (slices.getCount() == 0)
    {
        return;
    }

    IRBuilder builder(module);

    // We need to add a global instruction that references all of these string literals
    builder.setInsertInto(module->getModuleInst());

    const Index slicesCount = slices.getCount();

    ShortList<IRInst*> operandInsts;
    for (Index i = 0; i < slicesCount; ++i)
    {
        IRStringLit* stringLit = builder.getStringValue(slices[i]);
        operandInsts.add(stringLit);
    }

    IRInst* globalHashedInst = builder.emitIntrinsicInst(
        nullptr,
        kIROp_GlobalHashedStringLiterals,
        UInt(slicesCount),
        operandInsts.getArrayView().getBuffer());

    // Mark to keep alive
    builder.addKeepAliveDecoration(globalHashedInst);
}

Result checkGetStringHashInsts(IRModule* module, DiagnosticSink* sink)
{
    // Check all getStringHash are all on string literals
    List<IRGetStringHash*> insts;
    findGetStringHashInsts(module, insts);

    for (auto inst : insts)
    {
        if (inst->getStringLit() == nullptr)
        {
            if (sink)
            {
                sink->diagnose(inst, Diagnostics::getStringHashMustBeOnStringLiteral);
            }

            // Doesn't access a string literal
            return SLANG_FAIL;
        }
    }

    return SLANG_OK;
}

} // namespace Slang
