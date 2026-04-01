#include "slang-ir-any-value-inference.h"

#include "../core/slang-func-ptr.h"
#include "slang-ir-generics-lowering-context.h"
#include "slang-ir-insts.h"
#include "slang-ir-layout.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

void _findDependenciesOfTypeInSet(
    IRType* type,
    HashSet<IRInterfaceType*>& targetSet,
    List<IRInterfaceType*>& result)
{
    switch (type->getOp())
    {
    case kIROp_InterfaceType:
        {
            auto interfaceType = cast<IRInterfaceType>(type);
            if (targetSet.contains(interfaceType))
            {
                result.add(interfaceType);
                return;
            }
        }
        break;
    case kIROp_StructType:
        {
            auto structType = cast<IRStructType>(type);
            for (auto field : structType->getFields())
            {
                _findDependenciesOfTypeInSet(field->getFieldType(), targetSet, result);
            }
        }
        break;
    default:
        {
            for (UInt i = 0; i < type->getOperandCount(); i++)
            {
                if (auto operandType = as<IRType>(type->getOperand(i)))
                    _findDependenciesOfTypeInSet(operandType, targetSet, result);
            }
        }
        break;
    }
}

List<IRInterfaceType*> findDependenciesOfTypeInSet(
    IRType* type,
    HashSet<IRInterfaceType*> targetSet)
{
    List<IRInterfaceType*> result;
    _findDependenciesOfTypeInSet(type, targetSet, result);

    return result;
}

void _sortTopologically(
    IRInterfaceType* interfaceType,
    HashSet<IRInterfaceType*>& visited,
    List<IRInterfaceType*>& sortedInterfaceTypes,
    const Func<HashSet<IRInterfaceType*>, IRInterfaceType*>& getDependencies)
{
    if (visited.contains(interfaceType))
        return;

    visited.add(interfaceType);

    for (auto dependency : getDependencies(interfaceType))
    {
        _sortTopologically(dependency, visited, sortedInterfaceTypes, getDependencies);
    }

    sortedInterfaceTypes.add(interfaceType);
}

List<IRInterfaceType*> sortTopologically(
    HashSet<IRInterfaceType*> interfaceTypes,
    const Func<HashSet<IRInterfaceType*>, IRInterfaceType*>& getDependencies)
{
    List<IRInterfaceType*> sortedInterfaceTypes;
    HashSet<IRInterfaceType*> visited;
    for (auto interfaceType : interfaceTypes)
    {
        _sortTopologically(interfaceType, visited, sortedInterfaceTypes, getDependencies);
    }
    return sortedInterfaceTypes;
}

void inferAnyValueSizeWhereNecessary(TargetProgram* targetProgram, IRModule* module)
{
    // Go through the global insts and collect all interface types.
    // For each interface type, infer its any-value-size, by looking up
    // all witness tables whose conformance type matches the interface type.
    // then using _calcNaturalSizeAndAlignment to find the max size.
    //
    // Note: we only infer any-value-size for interface types that are used
    // as a generic type parameter, because we don't want to infer any-value-size
    // for interface types that are used as a witness table type.
    //

    HashSet<IRInst*> implementedInterfaces;
    // Add all interface type that are implemented by at least one type to a set.
    for (auto inst : module->getGlobalInsts())
    {
        if (inst->getOp() == kIROp_WitnessTable)
        {
            auto interfaceType =
                cast<IRWitnessTableType>(inst->getDataType())->getConformanceType();
            implementedInterfaces.add(interfaceType);
        }
    }

    // Collect all interface types that require inference.
    HashSet<IRInterfaceType*> interfaceTypes;
    for (auto inst : module->getGlobalInsts())
    {
        if (inst->getOp() == kIROp_InterfaceType)
        {
            auto interfaceType = cast<IRInterfaceType>(inst);

            // Do not infer anything for COM interfaces.
            if (isComInterfaceType((IRType*)interfaceType))
                continue;

            // Also skip builtin types.
            if (interfaceType->findDecoration<IRBuiltinDecoration>())
                continue;

            // If the interface already has an explicit any-value-size, don't infer anything.
            if (interfaceType->findDecoration<IRAnyValueSizeDecoration>())
                continue;

            // Skip interfaces that are not implemented by any type.
            if (!implementedInterfaces.contains(interfaceType))
                continue;

            interfaceTypes.add(interfaceType);
        }
    }

    Dictionary<IRInterfaceType*, List<IRInst*>> mapInterfaceToImplementations;

    // Collect all concrete types that conform to this interface type.
    for (auto interfaceType : interfaceTypes)
    {
        IRWitnessTableType* witnessTableType = nullptr;
        // Find witness table type corresponding to this interface.
        for (auto use = interfaceType->firstUse; use; use = use->nextUse)
        {
            if (auto _witnessTableType = as<IRWitnessTableType>(use->getUser()))
            {
                if (_witnessTableType->getConformanceType() == interfaceType &&
                    _witnessTableType->hasUses())
                {
                    witnessTableType = _witnessTableType;
                    break;
                }
            }
        }

        // If we hit this case, we have an interface without any conforming implementations.
        // This case should be handled before this point.
        //
        SLANG_ASSERT(witnessTableType);

        List<IRInst*> implList;

        // Walk through all the uses of this witness table type to find the witness tables.
        for (auto use = witnessTableType->firstUse; use; use = use->nextUse)
        {
            auto witnessTable = as<IRWitnessTable>(use->getUser());
            if (!witnessTable || witnessTable->getDataType() != witnessTableType)
                continue;

            auto concreteImpl = witnessTable->getConcreteType();

            // Only consider implementations at the top-level (ignore those nested
            // in generics)
            //
            if (concreteImpl->getParent() == module->getModuleInst())
                implList.add(concreteImpl);
        }

        mapInterfaceToImplementations.add(interfaceType, implList);
    }

    Dictionary<IRInterfaceType*, HashSet<IRInterfaceType*>> interfaceDependencyMap;

    // Collect dependencies for each interface.
    for (auto interfaceType : interfaceTypes)
    {
        HashSet<IRInterfaceType*> dependencySet;
        for (auto impl : mapInterfaceToImplementations[interfaceType])
        {
            auto dependencies = findDependenciesOfTypeInSet((IRType*)impl, interfaceTypes);
            for (auto dependency : dependencies)
                dependencySet.add(dependency);
        }
        interfaceDependencyMap.add(interfaceType, dependencySet);
    }

    // Sort the interface types in topological order.
    // This is necessary because we need to infer the any-value-size of an interface type
    // before we infer the any-value-size of an interface type that depends on it.
    //
    List<IRInterfaceType*> sortedInterfaceTypes = sortTopologically(
        interfaceTypes,
        [&](IRInterfaceType* interfaceType) { return interfaceDependencyMap[interfaceType]; });

    for (auto interfaceType : sortedInterfaceTypes)
    {
        IRIntegerValue maxAnyValueSize = -1;
        for (auto implType : mapInterfaceToImplementations[interfaceType])
        {
            IRSizeAndAlignment sizeAndAlignment;
            getNaturalSizeAndAlignment(
                targetProgram->getOptionSet(),
                (IRType*)implType,
                &sizeAndAlignment);

            maxAnyValueSize = Math::Max(maxAnyValueSize, sizeAndAlignment.size);
        }

        // Should not encounter interface types without any conforming implementations.
        SLANG_ASSERT(maxAnyValueSize >= 0);

        // If we found a max size, add an any-value-size decoration to the interface type.
        if (maxAnyValueSize >= 0)
        {
            IRBuilder builder(module);
            builder.addAnyValueSizeDecoration(interfaceType, maxAnyValueSize);
        }
    }
}
}; // namespace Slang
