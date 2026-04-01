// slang-ir-operator-shift-overflow.cpp
#include "slang-ir-operator-shift-overflow.h"

#include "slang-ir-insts.h"
#include "slang-ir-layout.h"
#include "slang-ir.h"
#include "slang.h"

namespace Slang
{

class DiagnosticSink;
struct IRModule;

void checkForOperatorShiftOverflowRecursive(
    IRInst* inst,
    CompilerOptionSet& optionSet,
    DiagnosticSink* sink)
{
    if (auto code = as<IRGlobalValueWithCode>(inst))
    {
        for (auto block : code->getBlocks())
        {
            for (auto opInst : block->getChildren())
            {
                switch (opInst->getOp())
                {
                case kIROp_Lsh:
                    {
                        SLANG_ASSERT(opInst->getOperandCount() == 2);

                        IRInst* rhs = opInst->getOperand(1);
                        auto rhsLit = as<IRIntLit>(rhs);
                        if (!rhsLit)
                            continue;

                        IRInst* lhs = opInst->getOperand(0);
                        IRType* lhsType = lhs->getDataType();

                        IRSizeAndAlignment sizeAlignment;
                        if (SLANG_FAILED(
                                getNaturalSizeAndAlignment(optionSet, lhsType, &sizeAlignment)))
                            continue;

                        IRIntegerValue shiftAmount = rhsLit->getValue();
                        if (sizeAlignment.size * 8 <= shiftAmount)
                        {
                            sink->diagnose(
                                opInst,
                                Diagnostics::operatorShiftLeftOverflow,
                                lhsType,
                                shiftAmount);
                        }
                        break;
                    }
                }
            }
        }
    }

    for (auto childInst : inst->getChildren())
    {
        checkForOperatorShiftOverflowRecursive(childInst, optionSet, sink);
    }
}

void checkForOperatorShiftOverflow(
    IRModule* module,
    CompilerOptionSet& optionSet,
    DiagnosticSink* sink)
{
    // Look for `operator<<` instructions
    checkForOperatorShiftOverflowRecursive(module->getModuleInst(), optionSet, sink);
}

} // namespace Slang
