#include "slang-ir-defunctionalization.h"

#include "slang-ir-insts.h"
#include "slang-ir-specialize-function-call.h"
#include "slang-ir-ssa-simplification.h"
#include "slang-ir.h"

namespace Slang
{

struct FunctionParameterSpecializationCondition : FunctionCallSpecializeCondition
{
    TargetRequest* targetRequest = nullptr;

    bool doesParamWantSpecialization(IRParam* param, IRInst* /*arg*/)
    {
        IRType* type = param->getDataType();
        return as<IRFuncType>(type);
    }
};

bool specializeHigherOrderParameters(CodeGenContext* codeGenContext, IRModule* module)
{
    bool result = false;
    FunctionParameterSpecializationCondition condition;
    condition.targetRequest = codeGenContext->getTargetReq();
    bool changed = true;
    while (changed)
    {
        changed = specializeFunctionCalls(codeGenContext, module, &condition);
        result |= changed;
    }
    return result;
}

} // namespace Slang
