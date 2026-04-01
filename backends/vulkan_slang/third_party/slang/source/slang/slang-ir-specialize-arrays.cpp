// slang-ir-specialize-arrays.cpp
#include "slang-ir-specialize-arrays.h"

#include "slang-ir-insts.h"
#include "slang-ir-specialize-function-call.h"
#include "slang-ir.h"

namespace Slang
{

struct ArrayParameterSpecializationCondition : FunctionCallSpecializeCondition
{
    // This pass is intended to specialize functions
    // with struct parameters that has array fields
    // to avoid performance problems for GLSL targets.
    // Returns true if `type` is an `IRStructType` with array-typed fields.
    // It will also specialize functions with unsized array parameters into
    // sized arrays, if the function is called with an argument that has a
    // sized array type.
    //
    bool isStructTypeWithArray(IRType* type)
    {
        if (auto structType = as<IRStructType>(type))
        {
            for (auto field : structType->getFields())
            {
                if (const auto arrayType = as<IRArrayType>(field->getFieldType()))
                {
                    return true;
                }
                if (auto subStructType = as<IRStructType>(field->getFieldType()))
                {
                    if (isStructTypeWithArray(subStructType))
                        return true;
                }
            }
        }
        return false;
    }

    bool doesParamWantSpecialization(IRParam* param, IRInst* arg)
    {
        SLANG_UNUSED(arg);
        if (isKhronosTarget(codeGenContext->getTargetReq()))
            return isStructTypeWithArray(param->getDataType());
        return false;
    }

    bool doesParamTypeWantSpecialization(IRParam* param, IRInst* arg)
    {
        auto paramType = param->getDataType();
        auto argType = arg->getDataType();
        if (auto outTypeBase = as<IROutTypeBase>(paramType))
        {
            paramType = outTypeBase->getValueType();
            SLANG_ASSERT(as<IRPtrTypeBase>(argType));
            argType = as<IRPtrTypeBase>(argType)->getValueType();
        }
        else if (auto refType = as<IRRefType>(paramType))
        {
            paramType = refType->getValueType();
            SLANG_ASSERT(as<IRPtrTypeBase>(argType));
            argType = as<IRPtrTypeBase>(argType)->getValueType();
        }
        else if (auto constRefType = as<IRConstRefType>(paramType))
        {
            paramType = constRefType->getValueType();
            SLANG_ASSERT(as<IRPtrTypeBase>(argType));
            argType = as<IRPtrTypeBase>(argType)->getValueType();
        }
        auto arrayType = as<IRUnsizedArrayType>(paramType);
        if (!arrayType)
            return false;
        auto argArrayType = as<IRArrayType>(argType);
        if (!argArrayType)
            return false;
        if (as<IRIntLit>(argArrayType->getElementCount()))
        {
            return true;
        }
        return false;
    }

    CodeGenContext* codeGenContext = nullptr;
};

void specializeArrayParameters(CodeGenContext* codeGenContext, IRModule* module)
{
    ArrayParameterSpecializationCondition condition;
    condition.codeGenContext = codeGenContext;
    specializeFunctionCalls(codeGenContext, module, &condition);
}

} // namespace Slang
