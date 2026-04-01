// slang-ir-strip-legalization-insts.cpp
#include "slang-ir-strip-legalization-insts.h"

#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

void stripLegalizationOnlyInstructions(IRModule* module)
{
    for (auto inst : module->getGlobalInsts())
    {
        switch (inst->getOp())
        {
        // Remove cached dictionaries.
        case kIROp_GenericSpecializationDictionary:
        case kIROp_ExistentialFuncSpecializationDictionary:
        case kIROp_ExistentialTypeSpecializationDictionary:
            {
                inst->removeAndDeallocate();
                break;
            }

        // Remove global param entry point param decoration.
        case kIROp_GlobalParam:
            {
                if (const auto entryPointParamDecoration =
                        inst->findDecoration<IREntryPointParamDecoration>())
                    entryPointParamDecoration->removeAndDeallocate();
                break;
            }

        // Remove witness tables.
        // Our goal here is to empty out any witness tables in
        // the IR so that they don't keep other symbols alive
        // further into compilation. Luckily we expect all
        // witness tables to live directly at the global scope
        // (or inside of a generic, which we can ignore for
        // now because the emit logic also ignores generics),
        // and there is a single function we can call to
        // remove all of the content from the witness tables
        // (since the key-value associations are stored as
        // children of each table).
        case kIROp_WitnessTable:
            {
                auto witnessTable = as<IRWitnessTable>(inst);
                auto conformanceType = witnessTable->getConformanceType();
                if (!conformanceType ||
                    !conformanceType->findDecoration<IRComInterfaceDecoration>())
                {
                    witnessTable->removeAndDeallocateAllDecorationsAndChildren();
                }
                break;
            }

        default:
            break;
        }
    }
}

void unpinWitnessTables(IRModule* module)
{
    for (auto inst : module->getGlobalInsts())
    {
        auto witnessTable = as<IRWitnessTable>(inst);
        if (!witnessTable)
            continue;

        // If a witness table is not used for dynamic dispatch, unpin it.
        if (!witnessTable->findDecoration<IRDynamicDispatchWitnessDecoration>())
        {
            while (auto decor = witnessTable->findDecoration<IRKeepAliveDecoration>())
                decor->removeAndDeallocate();
        }
    }
}

} // namespace Slang
