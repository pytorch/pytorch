#include "slang-ir-check-shader-parameter-type.h"

#include "slang-ir-util.h"

namespace Slang
{

template<typename P>
auto isOrContains(P predicate, IRType* type) -> decltype(predicate(type))
{
    HashSet<IRType*> visited;

    auto go = [&visited, &predicate](auto&& self, IRType* type) -> decltype(predicate(type))
    {
        // Prevent infinite recursion by tracking visited types
        if (!visited.add(type))
            return {};

        // Check if the current type matches the predicate
        if (auto result = predicate(type))
            return result;

        // Recursively check struct fields
        if (auto structType = as<IRStructType>(type))
        {
            for (auto field : structType->getFields())
            {
                auto fieldType = field->getFieldType();
                if (auto result = self(self, fieldType))
                    return result;
            }
        }

        return {};
    };

    return go(go, type);
}

void checkForInvalidShaderParameterTypeForMetal(IRModule* module, DiagnosticSink* sink)
{
    auto isConstantBufferWithResource = [](IRType* type) -> std::optional<IRType*>
    {
        if (type->getOp() == kIROp_ConstantBufferType)
        {
            // Get the type inside the constant buffer
            auto innerType = as<IRType>(type->getOperand(0));

            // Check if the inner type contains any resource types
            auto hasResource = [](IRType* t) -> std::optional<IRType*>
            {
                if (isResourceType(t))
                    return t;
                return {};
            };

            if (auto resourceType = isOrContains(hasResource, innerType))
                return type; // Return the constant buffer type if it contains a resource
        }
        return {};
    };

    for (auto inst : module->getGlobalInsts())
    {
        if (inst->getOp() != kIROp_ParameterBlockType)
            continue;

        auto type = as<IRType>(inst->getOperand(0));
        if (auto invalidCBType = isOrContains(isConstantBufferWithResource, type))
        {
            // Try to find a valid source location from uses
            bool foundUseSite = false;
            for (auto use = inst->firstUse; use; use = use->nextUse)
            {
                auto user = use->getUser();
                if (user->sourceLoc.isValid())
                {
                    sink->diagnose(
                        user,
                        Diagnostics::
                            resourceTypesInConstantBufferInParameterBlockNotAllowedOnMetal);
                    foundUseSite = true;
                    break;
                }
            }

            if (!foundUseSite)
                sink->diagnose(
                    inst,
                    Diagnostics::resourceTypesInConstantBufferInParameterBlockNotAllowedOnMetal);
        }
    }
}
void checkForInvalidShaderParameterType(
    TargetRequest* target,
    IRModule* module,
    DiagnosticSink* sink)
{
    if (isMetalTarget(target))
        checkForInvalidShaderParameterTypeForMetal(module, sink);
}
} // namespace Slang
