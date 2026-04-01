// slang-ir-generics-lowering-context.cpp

#include "slang-ir-generics-lowering-context.h"

#include "slang-ir-layout.h"
#include "slang-ir-util.h"

namespace Slang
{
bool isPolymorphicType(IRInst* typeInst)
{
    if (as<IRParam>(typeInst) && as<IRTypeType>(typeInst->getDataType()))
        return true;
    switch (typeInst->getOp())
    {
    case kIROp_ThisType:
    case kIROp_AssociatedType:
    case kIROp_InterfaceType:
    case kIROp_LookupWitness:
        return true;
    case kIROp_Specialize:
        {
            for (UInt i = 0; i < typeInst->getOperandCount(); i++)
            {
                if (isPolymorphicType(typeInst->getOperand(i)))
                    return true;
            }
            return false;
        }
    default:
        break;
    }
    if (auto ptrType = as<IRPtrTypeBase>(typeInst))
    {
        return isPolymorphicType(ptrType->getValueType());
    }
    return false;
}

bool isTypeValue(IRInst* typeInst)
{
    if (typeInst)
    {
        switch (typeInst->getOp())
        {
        case kIROp_TypeType:
        case kIROp_TypeKind:
            return true;
        default:
            return false;
        }
    }
    return false;
}

IRInst* SharedGenericsLoweringContext::maybeEmitRTTIObject(IRInst* typeInst)
{
    IRInst* result = nullptr;
    if (mapTypeToRTTIObject.tryGetValue(typeInst, result))
        return result;
    IRBuilder builderStorage(module);
    auto builder = &builderStorage;
    builder->setInsertAfter(typeInst);

    result = builder->emitMakeRTTIObject(typeInst);

    // For now the only type info we encapsualte is type size.
    IRSizeAndAlignment sizeAndAlignment;
    getNaturalSizeAndAlignment(targetProgram->getOptionSet(), (IRType*)typeInst, &sizeAndAlignment);
    builder->addRTTITypeSizeDecoration(result, sizeAndAlignment.size);

    // Give a name to the rtti object.
    if (auto exportDecoration = typeInst->findDecoration<IRExportDecoration>())
    {
        String rttiObjName = exportDecoration->getMangledName();
        builder->addExportDecoration(result, rttiObjName.getUnownedSlice());
    }

    // Make sure the RTTI object for an exported struct type is marked as export if the type is.
    if (typeInst->findDecoration<IRHLSLExportDecoration>())
    {
        builder->addHLSLExportDecoration(result);
        builder->addKeepAliveDecoration(result);
    }
    mapTypeToRTTIObject[typeInst] = result;
    return result;
}

IRInst* SharedGenericsLoweringContext::findInterfaceRequirementVal(
    IRInterfaceType* interfaceType,
    IRInst* requirementKey)
{
    if (auto dict = mapInterfaceRequirementKeyValue.tryGetValue(interfaceType))
        return dict->getValue(requirementKey);
    _builldInterfaceRequirementMap(interfaceType);
    return findInterfaceRequirementVal(interfaceType, requirementKey);
}

void SharedGenericsLoweringContext::_builldInterfaceRequirementMap(IRInterfaceType* interfaceType)
{
    mapInterfaceRequirementKeyValue.add(interfaceType, Dictionary<IRInst*, IRInst*>());
    auto dict = mapInterfaceRequirementKeyValue.tryGetValue(interfaceType);
    for (UInt i = 0; i < interfaceType->getOperandCount(); i++)
    {
        auto entry = cast<IRInterfaceRequirementEntry>(interfaceType->getOperand(i));
        (*dict)[entry->getRequirementKey()] = entry->getRequirementVal();
    }
}

IRType* SharedGenericsLoweringContext::lowerAssociatedType(IRBuilder* builder, IRInst* type)
{
    if (type->getOp() != kIROp_AssociatedType)
        return (IRType*)type;
    IRIntegerValue anyValueSize = kInvalidAnyValueSize;
    for (UInt i = 0; i < type->getOperandCount(); i++)
    {
        anyValueSize =
            Math::Min(anyValueSize, getInterfaceAnyValueSize(type->getOperand(i), type->sourceLoc));
    }
    if (anyValueSize == kInvalidAnyValueSize)
    {
        // We could conceivably make it an error to have an associated type
        // without an `[anyValueSize(...)]` attribute, but then we risk
        // producing error messages even when doing 100% static specialization.
        //
        // It is simpler to use a reasonable default size and treat any
        // type without an explicit attribute as using that size.
        //
        anyValueSize = kDefaultAnyValueSize;
    }
    return builder->getAnyValueType(anyValueSize);
}

IRType* SharedGenericsLoweringContext::lowerType(
    IRBuilder* builder,
    IRInst* paramType,
    const Dictionary<IRInst*, IRInst*>& typeMapping,
    IRType* concreteType)
{
    if (!paramType)
        return nullptr;

    IRInst* resultType;
    if (typeMapping.tryGetValue(paramType, resultType))
        return (IRType*)resultType;

    if (isTypeValue(paramType))
    {
        return builder->getRTTIHandleType();
    }

    switch (paramType->getOp())
    {
    case kIROp_WitnessTableType:
    case kIROp_WitnessTableIDType:
    case kIROp_ExtractExistentialType:
        // Do not translate these types.
        return (IRType*)paramType;
    case kIROp_Param:
        {
            if (auto anyValueSizeDecor = paramType->findDecoration<IRTypeConstraintDecoration>())
            {
                if (isBuiltin(anyValueSizeDecor->getConstraintType()))
                    return (IRType*)paramType;
                auto anyValueSize = getInterfaceAnyValueSize(
                    anyValueSizeDecor->getConstraintType(),
                    paramType->sourceLoc);
                return builder->getAnyValueType(anyValueSize);
            }
            // We could conceivably make it an error to have a generic parameter
            // without an `[anyValueSize(...)]` attribute, but then we risk
            // producing error messages even when doing 100% static specialization.
            //
            // It is simpler to use a reasonable default size and treat any
            // type without an explicit attribute as using that size.
            //
            return builder->getAnyValueType(kDefaultAnyValueSize);
        }
    case kIROp_ThisType:
        {
            auto interfaceType = cast<IRThisType>(paramType)->getConstraintType();

            if (isBuiltin(interfaceType))
                return (IRType*)paramType;

            if (isComInterfaceType((IRType*)interfaceType))
                return (IRType*)interfaceType;

            auto anyValueSize = getInterfaceAnyValueSize(
                cast<IRThisType>(paramType)->getConstraintType(),
                paramType->sourceLoc);
            return builder->getAnyValueType(anyValueSize);
        }
    case kIROp_AssociatedType:
        {
            return lowerAssociatedType(builder, paramType);
        }
    case kIROp_InterfaceType:
        {
            if (isBuiltin(paramType))
                return (IRType*)paramType;

            if (isComInterfaceType((IRType*)paramType))
                return (IRType*)paramType;

            // In the dynamic-dispatch case, a value of interface type
            // is going to be packed into the "any value" part of a tuple.
            // The size of the "any value" part depends on the interface
            // type (e.g., it might have an `[anyValueSize(8)]` attribute
            // indicating that 8 bytes needs to be reserved).
            //
            auto anyValueSize = getInterfaceAnyValueSize(paramType, paramType->sourceLoc);

            // If there is a non-null `concreteType` parameter, then this
            // interface type is one that has been statically bound (via
            // specialization parameters) to hold a value of that concrete
            // type.
            //
            IRType* pendingType = nullptr;
            if (concreteType)
            {
                // Because static specialization is being used (at least in part),
                // we do *not* have a guarantee that the `concreteType` is one
                // that can fit into the `anyValueSize` of the interface.
                //
                // We will use the IR layout logic to see if we can compute
                // a size for the type, which can lead to a few different outcomes:
                //
                // * If a size is computed successfully, and it is smaller than or
                //   equal to `anyValueSize`, then the concrete value will fit into
                //   the reserved area, and the layout will match the dynamic case.
                //
                // * If a size is computed successfully, and it is larger than
                //   `anyValueSize`, then the concrete value cannot fit into the
                //   reserved area, and it needs to be stored out-of-line.
                //
                // * If size cannot be computed, then that implies that the type
                //   includes non-ordinary data (e.g., a `Texture2D` on a D3D11
                //   target), and cannot possible fit into the reserved area
                //   (which consists of only uniform bytes). In this case, the
                //   value must be stored out-of-line.
                //
                IRSizeAndAlignment sizeAndAlignment;
                Result result = getNaturalSizeAndAlignment(
                    targetProgram->getOptionSet(),
                    concreteType,
                    &sizeAndAlignment);
                if (SLANG_FAILED(result) || (sizeAndAlignment.size > anyValueSize))
                {
                    // If the value must be stored out-of-line, we construct
                    // a "pseudo pointer" to the concrete type, and the
                    // constructed tuple will contain such a pseudo pointer.
                    //
                    // Semantically, the pseudo pointer behaves a bit like
                    // a pointer to the concrete type, in that it can be
                    // (pseudo-)dereferenced to produce a value of the chosen
                    // type.
                    //
                    // In terms of layout, the pseudo pointer occupies no
                    // space in the parent tuple/type, and will be automatically
                    // moved out-of-line by a later type legalization pass.
                    //
                    pendingType = builder->getPseudoPtrType(concreteType);
                }
            }

            auto anyValueType = builder->getAnyValueType(anyValueSize);
            auto witnessTableType = builder->getWitnessTableIDType((IRType*)paramType);
            auto rttiType = builder->getRTTIHandleType();

            IRType* tupleType = nullptr;
            if (!pendingType)
            {
                // In the oridnary (dynamic) case, an existential type decomposes
                // into a tuple of:
                //
                //      (RTTI, witness table, any-value).
                //
                tupleType = builder->getTupleType(rttiType, witnessTableType, anyValueType);
            }
            else
            {
                // In the case where static specialization mandateds out-of-line storage,
                // an existential type decomposes into a tuple of:
                //
                //      (RTTI, witness table, pseudo pointer, any-value)
                //
                tupleType =
                    builder->getTupleType(rttiType, witnessTableType, pendingType, anyValueType);
                //
                // Note that in each of the cases, the third element of the tuple
                // is a representation of the value being stored in the existential.
                //
                // Also note that each of these representations has the same
                // size and alignment when only "ordinary" data is considered
                // (the pseudo-pointer will eventually be legalized away, leaving
                // behind a tuple with equivalent layout).
            }

            return tupleType;
        }
    case kIROp_LookupWitness:
        {
            auto lookupInterface = static_cast<IRLookupWitnessMethod*>(paramType);
            auto witnessTableType =
                as<IRWitnessTableType>(lookupInterface->getWitnessTable()->getDataType());
            if (!witnessTableType)
                return (IRType*)paramType;
            auto interfaceType = as<IRInterfaceType>(witnessTableType->getConformanceType());
            if (!interfaceType || isBuiltin(interfaceType))
                return (IRType*)paramType;
            // Make sure we are looking up inside the original interface type (prior to lowering).
            // Only in the original interface type will an associated type entry have an
            // IRAssociatedType value. We need to extract AnyValueSize from this IRAssociatedType.
            // In lowered interface type, that entry is lowered into an Ptr(RTTIType) and this info
            // is lost.
            mapLoweredInterfaceToOriginal.tryGetValue(interfaceType, interfaceType);
            auto reqVal =
                findInterfaceRequirementVal(interfaceType, lookupInterface->getRequirementKey());
            SLANG_ASSERT(reqVal && reqVal->getOp() == kIROp_AssociatedType);
            return lowerType(builder, reqVal, typeMapping, nullptr);
        }
    case kIROp_BoundInterfaceType:
        {
            // A bound interface type represents an existential together with
            // static knowledge that the value stored in the extistential has
            // a particular concrete type.
            //
            // We handle this case by lowering the underlying interface type,
            // but pass along the concrete type so that it can impact the
            // layout of the interface type.
            //
            auto boundInterfaceType = static_cast<IRBoundInterfaceType*>(paramType);
            return lowerType(
                builder,
                boundInterfaceType->getInterfaceType(),
                typeMapping,
                boundInterfaceType->getConcreteType());
        }
    default:
        {
            bool translated = false;
            List<IRInst*> loweredOperands;
            for (UInt i = 0; i < paramType->getOperandCount(); i++)
            {
                loweredOperands.add(
                    lowerType(builder, paramType->getOperand(i), typeMapping, nullptr));
                if (loweredOperands.getLast() != paramType->getOperand(i))
                    translated = true;
            }
            if (translated)
                return builder->getType(
                    paramType->getOp(),
                    loweredOperands.getCount(),
                    loweredOperands.getBuffer());
            return (IRType*)paramType;
        }
    }
}

List<IRWitnessTable*> getWitnessTablesFromInterfaceType(IRModule* module, IRInst* interfaceType)
{
    List<IRWitnessTable*> witnessTables;
    for (auto globalInst : module->getGlobalInsts())
    {
        if (globalInst->getOp() == kIROp_WitnessTable &&
            cast<IRWitnessTableType>(globalInst->getDataType())->getConformanceType() ==
                interfaceType)
        {
            witnessTables.add(cast<IRWitnessTable>(globalInst));
        }
    }
    return witnessTables;
}

List<IRWitnessTable*> SharedGenericsLoweringContext::getWitnessTablesFromInterfaceType(
    IRInst* interfaceType)
{
    return Slang::getWitnessTablesFromInterfaceType(module, interfaceType);
}

IRIntegerValue SharedGenericsLoweringContext::getInterfaceAnyValueSize(
    IRInst* type,
    SourceLoc usageLoc)
{
    SLANG_UNUSED(usageLoc);

    if (auto decor = type->findDecoration<IRAnyValueSizeDecoration>())
    {
        return decor->getSize();
    }

    // We could conceivably make it an error to have an interface
    // without an `[anyValueSize(...)]` attribute, but then we risk
    // producing error messages even when doing 100% static specialization.
    //
    // It is simpler to use a reasonable default size and treat any
    // type without an explicit attribute as using that size.
    //
    return kDefaultAnyValueSize;
}


bool SharedGenericsLoweringContext::doesTypeFitInAnyValue(
    IRType* concreteType,
    IRInterfaceType* interfaceType,
    IRIntegerValue* outTypeSize,
    IRIntegerValue* outLimit,
    bool* outIsTypeOpaque)
{
    auto anyValueSize = getInterfaceAnyValueSize(interfaceType, interfaceType->sourceLoc);
    if (outLimit)
        *outLimit = anyValueSize;

    if (!areResourceTypesBindlessOnTarget(targetProgram->getTargetReq()))
    {
        IRType* opaqueType = nullptr;
        if (isOpaqueType(concreteType, &opaqueType))
        {
            if (outIsTypeOpaque)
                *outIsTypeOpaque = true;
            return false;
        }
    }
    IRSizeAndAlignment sizeAndAlignment;
    Result result =
        getNaturalSizeAndAlignment(targetProgram->getOptionSet(), concreteType, &sizeAndAlignment);
    if (outTypeSize)
        *outTypeSize = sizeAndAlignment.size;

    if (SLANG_FAILED(result) || (sizeAndAlignment.size > anyValueSize))
    {
        // The value does not fit, either because it is too large,
        // or because it includes types that cannot be stored
        // in uniform/ordinary memory for this target.
        //
        return false;
    }

    return true;
}

} // namespace Slang
