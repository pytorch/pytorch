#include "slang-ir-check-recursion.h"

#include "slang-ir-util.h"

namespace Slang
{
bool checkTypeRecursionImpl(
    HashSet<IRInst*>& checkedTypes,
    HashSet<IRInst*>& stack,
    IRInst* type,
    IRInst* field,
    DiagnosticSink* sink)
{
    auto visitElementType = [&](IRInst* elementType, IRInst* field) -> bool
    {
        if (!stack.add(elementType))
        {
            sink->diagnose(field ? field : type, Diagnostics::recursiveType, type);
            return false;
        }
        if (checkedTypes.add(elementType))
            checkTypeRecursionImpl(checkedTypes, stack, elementType, field, sink);
        stack.remove(elementType);
        return true;
    };
    if (auto arrayType = as<IRArrayTypeBase>(type))
    {
        return visitElementType(arrayType->getElementType(), field);
    }
    else if (auto structType = as<IRStructType>(type))
    {
        for (auto sfield : structType->getFields())
            if (!visitElementType(sfield->getFieldType(), sfield))
                return false;
    }
    return true;
}

void checkTypeRecursion(HashSet<IRInst*>& checkedTypes, IRInst* type, DiagnosticSink* sink)
{
    HashSet<IRInst*> stack;
    if (checkedTypes.add(type))
    {
        stack.add(type);
        checkTypeRecursionImpl(checkedTypes, stack, type, nullptr, sink);
    }
}

void checkForRecursiveTypes(IRModule* module, DiagnosticSink* sink)
{
    HashSet<IRInst*> checkedTypes;
    for (auto globalInst : module->getGlobalInsts())
    {
        switch (globalInst->getOp())
        {
        case kIROp_StructType:
            {
                checkTypeRecursion(checkedTypes, globalInst, sink);
            }
            break;
        default:
            break;
        }
    }
}

bool checkFunctionRecursionImpl(
    HashSet<IRFunc*>& checkedFuncs,
    HashSet<IRFunc*>& callStack,
    IRFunc* func,
    DiagnosticSink* sink)
{
    for (auto block : func->getBlocks())
    {
        for (auto inst : block->getChildren())
        {
            auto callInst = as<IRCall>(inst);
            if (!callInst)
                continue;
            auto callee = as<IRFunc>(callInst->getCallee());
            if (!callee)
                continue;
            if (!callStack.add(callee))
            {
                sink->diagnose(callInst, Diagnostics::unsupportedRecursion, callee);
                return false;
            }
            if (checkedFuncs.add(callee))
                checkFunctionRecursionImpl(checkedFuncs, callStack, callee, sink);
            callStack.remove(callee);
        }
    }
    return true;
}

void checkFunctionRecursion(HashSet<IRFunc*>& checkedFuncs, IRFunc* func, DiagnosticSink* sink)
{
    HashSet<IRFunc*> callStack;
    if (checkedFuncs.add(func))
    {
        callStack.add(func);
        checkFunctionRecursionImpl(checkedFuncs, callStack, func, sink);
    }
}

void checkForRecursiveFunctions(TargetRequest* target, IRModule* module, DiagnosticSink* sink)
{
    HashSet<IRFunc*> checkedFuncsForRecursionDetection;
    for (auto globalInst : module->getGlobalInsts())
    {
        switch (globalInst->getOp())
        {
        case kIROp_Func:
            if (!isCPUTarget(target))
                checkFunctionRecursion(
                    checkedFuncsForRecursionDetection,
                    as<IRFunc>(globalInst),
                    sink);
            break;
        default:
            break;
        }
    }
}

} // namespace Slang
