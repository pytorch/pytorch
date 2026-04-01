// slang-ir-ssa-register-allocate.h
#pragma once

#include "slang-ir.h"

namespace Slang
{
struct IRDominatorTree;

struct RegisterInfo : RefObject
{
    IRType* type;
    List<IRInst*> insts;
};

struct RegisterAllocationResult
{
    OrderedDictionary<IRType*, List<RefPtr<RegisterInfo>>> mapTypeToRegisterList;
    Dictionary<IRInst*, RefPtr<RegisterInfo>> mapInstToRegister;
};

RegisterAllocationResult allocateRegistersForFunc(
    IRGlobalValueWithCode* func,
    RefPtr<IRDominatorTree>& inOutDom,
    bool allocateForCompositeTypesOnly);

} // namespace Slang
