#ifdef SLANG_IN_SPIRV_EMIT_CONTEXT
// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpUndef
template<typename T>
SpvInst* emitOpUndef(SpvInstParent* parent, IRInst* inst, const T& idResultType)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpUndef, idResultType, kResultID);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpName
template<typename T>
SpvInst* emitOpName(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    const UnownedStringSlice& name)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpName, target, name);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberName
template<typename T>
SpvInst* emitOpMemberName(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    int index,
    const UnownedStringSlice& name)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpMemberName, target, SpvLiteralInteger::from32(index), name);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpExtension
SpvInst* emitOpExtension(SpvInstParent* parent, IRInst* inst, const UnownedStringSlice& name)
{
    return emitInst(parent, inst, SpvOpExtension, name);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpExtInstImport
SpvInst* emitOpExtInstImport(SpvInstParent* parent, IRInst* inst, const UnownedStringSlice& name)
{
    return emitInst(parent, inst, SpvOpExtInstImport, kResultID, name);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemoryModel
SpvInst* emitOpMemoryModel(
    SpvInstParent* parent,
    IRInst* inst,
    SpvAddressingModel addressingModel,
    SpvMemoryModel memoryModel)
{
    return emitInst(parent, inst, SpvOpMemoryModel, addressingModel, memoryModel);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpEntryPoint
template<typename T, typename Ts>
SpvInst* emitOpEntryPoint(
    SpvInstParent* parent,
    IRInst* inst,
    SpvExecutionModel executionModel,
    const T& entryPoint,
    const UnownedStringSlice& name,
    const Ts& interfaces)
{
    static_assert(isSingular<T>);
    static_assert(isPlural<Ts>);
    return emitInst(parent, inst, SpvOpEntryPoint, executionModel, entryPoint, name, interfaces);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpCapability
SpvInst* emitOpCapability(SpvInstParent* parent, IRInst* inst, SpvCapability capability)
{
    return emitInst(parent, inst, SpvOpCapability, capability);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeVoid
SpvInst* emitOpTypeVoid(IRInst* inst)
{
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeVoid,
        kResultID);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeBool
SpvInst* emitOpTypeBool(IRInst* inst)
{
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeBool,
        kResultID);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeInt
SpvInst* emitOpTypeInt(
    IRInst* inst,
    const SpvLiteralInteger& width,
    const SpvLiteralInteger& signedness)
{
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeInt,
        kResultID,
        width,
        signedness);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeFloat
SpvInst* emitOpTypeFloat(IRInst* inst, const SpvLiteralInteger& width)
{
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeFloat,
        kResultID,
        width);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeVector
template<typename T>
SpvInst* emitOpTypeVector(
    IRInst* inst,
    const T& componentType,
    const SpvLiteralInteger& componentCount)
{
    static_assert(isSingular<T>);
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeVector,
        kResultID,
        componentType,
        componentCount);
}

template<typename T1, typename T2>
SpvInst* emitOpTypeCoopVec(IRInst* inst, const T1& componentType, const T2& componentCount)
{
    static_assert(isSingular<T1>);
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeCooperativeVectorNV,
        kResultID,
        componentType,
        componentCount);
}

template<typename T1, typename T2>
SpvInst* emitOpTypeCoopMat(
    IRInst* inst,
    const T1& componentType,
    const T2& scope,
    const T2& rowCount,
    const T2& columnCount,
    const T2& matrixUse)
{
    static_assert(isSingular<T1>);
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeCooperativeMatrixKHR,
        kResultID,
        componentType,
        scope,
        rowCount,
        columnCount,
        matrixUse);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeMatrix
template<typename T>
SpvInst* emitOpTypeMatrix(IRInst* inst, const T& columnType, const SpvLiteralInteger& columnCount)
{
    static_assert(isSingular<T>);
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeMatrix,
        kResultID,
        columnType,
        columnCount);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeImage
template<typename T>
SpvInst* emitOpTypeImage(
    IRInst* inst,
    const T& sampledType,
    SpvDim dim,
    const SpvLiteralInteger& depth,
    const SpvLiteralInteger& arrayed,
    const SpvLiteralInteger& mS,
    const SpvLiteralInteger& sampled,
    SpvImageFormat imageFormat,
    OptionalOperand<SpvAccessQualifier> accessQualifier = SkipThisOptionalOperand{})
{
    static_assert(isSingular<T>);
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeImage,
        kResultID,
        sampledType,
        dim,
        depth,
        arrayed,
        mS,
        sampled,
        imageFormat,
        accessQualifier);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeSampler
SpvInst* emitOpTypeSampler(IRInst* inst)
{
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeSampler,
        kResultID);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeSampler
template<typename T1>
SpvInst* emitOpTypeSampledImage(IRInst* inst, const T1& imageType)
{
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeSampledImage,
        kResultID,
        imageType);
}

SpvInst* emitOpTypeAccelerationStructure(IRInst* inst)
{
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeAccelerationStructureKHR,
        kResultID);
}

SpvInst* emitOpTypeRayQuery(IRInst* inst)
{
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeRayQueryKHR,
        kResultID);
}

SpvInst* emitOpTypeHitObject(IRInst* inst)
{
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeHitObjectNV,
        kResultID);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeArray
template<typename T1, typename T2>
SpvInst* emitOpTypeArray(IRInst* inst, const T1& elementType, const T2& length)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeArray,
        kResultID,
        elementType,
        length);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeRuntimeArray
template<typename T>
SpvInst* emitOpTypeRuntimeArray(IRInst* inst, const T& elementType)
{
    static_assert(isSingular<T>);
    return emitInst(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeRuntimeArray,
        kResultID,
        elementType);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeStruct
template<typename Ts>
SpvInst* emitOpTypeStruct(IRInst* inst, const Ts& member0TypeMember1TypeEtc)
{
    static_assert(isPlural<Ts>);
    return emitInst(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeStruct,
        kResultID,
        member0TypeMember1TypeEtc);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeForwardPointer
template<typename T>
SpvInst* emitOpTypeForwardPointer(const T& type, SpvStorageClass storageClass)
{
    static_assert(isSingular<T>);
    return emitInst(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        nullptr,
        SpvOpTypeForwardPointer,
        type,
        storageClass);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypePointer
template<typename T>
SpvInst* emitOpTypePointer(IRInst* inst, SpvStorageClass storageClass, const T& type)
{
    static_assert(isSingular<T>);
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypePointer,
        kResultID,
        storageClass,
        type);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeFunction
template<typename T, typename Ts>
SpvInst* emitOpTypeFunction(
    IRInst* inst,
    const T& returnType,
    const Ts& parameter0TypeParameter1TypeEtc)
{
    static_assert(isSingular<T>);
    static_assert(isPlural<Ts>);
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeFunction,
        kResultID,
        returnType,
        parameter0TypeParameter1TypeEtc);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpConstantTrue
template<typename T>
SpvInst* emitOpConstantTrue(IRInst* inst, const T& idResultType)
{
    static_assert(isSingular<T>);
    return emitInst(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpConstantTrue,
        idResultType,
        kResultID);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpConstantFalse
template<typename T>
SpvInst* emitOpConstantFalse(IRInst* inst, const T& idResultType)
{
    static_assert(isSingular<T>);
    return emitInst(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpConstantFalse,
        idResultType,
        kResultID);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpConstant
template<typename T>
SpvInst* emitOpConstant(IRInst* inst, const T& idResultType, const SpvLiteralBits& value)
{
    static_assert(isSingular<T>);
    return emitInst(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpConstant,
        idResultType,
        kResultID,
        value);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpConstantComposite
template<typename T, typename Ts>
SpvInst* emitOpConstantComposite(IRInst* inst, const T& idResultType, const Ts& constituents)
{
    static_assert(isSingular<T>);
    static_assert(isPlural<Ts>);
    return emitInst(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpConstantComposite,
        idResultType,
        kResultID,
        constituents);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpConstantNull
template<typename T>
SpvInst* emitOpConstantNull(IRInst* inst, const T& idResultType)
{
    static_assert(isSingular<T>);
    return emitInst(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpConstantNull,
        idResultType,
        kResultID);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFunction
template<typename T1, typename T2>
SpvInst* emitOpFunction(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    SpvFunctionControlMask functionControl,
    const T2& functionType)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(
        parent,
        inst,
        SpvOpFunction,
        idResultType,
        kResultID,
        functionControl,
        functionType);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFunctionParameter
template<typename T>
SpvInst* emitOpFunctionParameter(SpvInstParent* parent, IRInst* inst, const T& idResultType)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpFunctionParameter, idResultType, kResultID);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFunctionEnd
SpvInst* emitOpFunctionEnd(SpvInstParent* parent, IRInst* inst)
{
    return emitInst(parent, inst, SpvOpFunctionEnd);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFunctionCall
template<typename T1, typename T2, typename Ts>
SpvInst* emitOpFunctionCall(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& function,
    const Ts& argument0Argument1Etc)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isPlural<Ts>);
    return emitInst(
        parent,
        inst,
        SpvOpFunctionCall,
        idResultType,
        kResultID,
        function,
        argument0Argument1Etc);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpVariable
template<typename T, typename Opt = SkipThisOptionalOperand>
SpvInst* emitOpVariable(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvStorageClass storageClass,
    const Opt& initializer = SkipThisOptionalOperand{})
{
    static_assert(isSingular<T>);
    static_assert(isSingular<Opt>);
    return emitInst(
        parent,
        inst,
        SpvOpVariable,
        idResultType,
        kResultID,
        storageClass,
        initializer);
}

template<typename T, typename TOperand>
SpvInst* emitOpSpecConstant(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    TOperand operand)
{
    return emitInst(parent, inst, SpvOpSpecConstant, idResultType, kResultID, operand);
}

template<typename T, typename Ts>
SpvInst* emitOpSpecConstantComposite(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    const Ts& constituents)
{
    return emitInst(
        parent,
        inst,
        SpvOpSpecConstantComposite,
        idResultType,
        kResultID,
        constituents);
}

template<typename T>
SpvInst* emitOpSpecConstantTrue(SpvInstParent* parent, IRInst* inst, const T& idResultType)
{
    return emitInst(parent, inst, SpvOpSpecConstantTrue, idResultType, kResultID);
}

template<typename T>
SpvInst* emitOpSpecConstantFalse(SpvInstParent* parent, IRInst* inst, const T& idResultType)
{
    return emitInst(parent, inst, SpvOpSpecConstantFalse, idResultType, kResultID);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLoad
template<typename T1, typename T2>
SpvInst* emitOpLoad(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& pointer,
    OptionalOperand<SpvMemoryAccessMask> memoryAccess = SkipThisOptionalOperand{})
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpLoad, idResultType, kResultID, pointer, memoryAccess);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLoad
template<typename T1, typename T2>
SpvInst* emitOpLoadAligned(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& pointer,
    const SpvLiteralInteger& literalInteger)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(
        parent,
        inst,
        SpvOpLoad,
        idResultType,
        kResultID,
        pointer,
        SpvMemoryAccessAlignedMask,
        literalInteger);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpStore
template<typename T1, typename T2>
SpvInst* emitOpStore(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& pointer,
    const T2& object,
    OptionalOperand<SpvMemoryAccessMask> memoryAccess = SkipThisOptionalOperand{})
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpStore, pointer, object, memoryAccess);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpStore
template<typename T1, typename T2>
SpvInst* emitOpStoreAligned(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& pointer,
    const T2& object,
    const SpvLiteralInteger& literalInteger)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(
        parent,
        inst,
        SpvOpStore,
        pointer,
        object,
        SpvMemoryAccessAlignedMask,
        literalInteger);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpAccessChain
template<typename T1, typename T2, typename Ts>
SpvInst* emitOpAccessChain(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& base,
    const Ts& indexes)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isPlural<Ts>);
    return emitInst(parent, inst, SpvOpAccessChain, idResultType, kResultID, base, indexes);
}


// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpPtrAccessChain
template<typename T1, typename T2, typename T3>
SpvInst* emitOpPtrAccessChain(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& base,
    const T3& element)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpPtrAccessChain, idResultType, kResultID, base, element);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorate(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    SpvDecoration decoration)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpDecorate, target, decoration);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorateSpecId(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    const SpvLiteralInteger& specializationConstantID)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpDecorate,
        target,
        SpvDecorationSpecId,
        specializationConstantID);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorateArrayStride(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    const SpvLiteralInteger& arrayStride)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpDecorate, target, SpvDecorationArrayStride, arrayStride);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorateMatrixStride(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    const SpvLiteralInteger& matrixStride)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpDecorate, target, SpvDecorationMatrixStride, matrixStride);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorateBuiltIn(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    SpvBuiltIn builtIn)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpDecorate, target, SpvDecorationBuiltIn, builtIn);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpMemberDecorateString(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    const SpvLiteralInteger& index,
    SpvDecoration decoration,
    UnownedStringSlice text)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpMemberDecorateString, target, index, decoration, text);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorateString(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    SpvDecoration decoration,
    UnownedStringSlice text)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpDecorateString, target, decoration, text);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T1, typename T2>
SpvInst* emitOpDecorateUniformId(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& target,
    const T2& execution)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpDecorate, target, SpvDecorationUniformId, execution);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorateLocation(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    const SpvLiteralInteger& location)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpDecorate, target, SpvDecorationLocation, location);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorateComponent(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    const SpvLiteralInteger& component)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpDecorate, target, SpvDecorationComponent, component);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorateIndex(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    const SpvLiteralInteger& index)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpDecorate, target, SpvDecorationIndex, index);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorateBinding(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    const SpvLiteralInteger& bindingPoint)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpDecorate, target, SpvDecorationBinding, bindingPoint);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorateInputAttachmentIndex(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    const SpvLiteralInteger& bindingPoint)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpDecorate,
        target,
        SpvDecorationInputAttachmentIndex,
        bindingPoint);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorateDescriptorSet(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    const SpvLiteralInteger& descriptorSet)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpDecorate, target, SpvDecorationDescriptorSet, descriptorSet);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorateOffset(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    const SpvLiteralInteger& byteOffset)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpDecorate, target, SpvDecorationOffset, byteOffset);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorateFPRoundingMode(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    SpvFPRoundingMode floatingPointRoundingMode)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpDecorate,
        target,
        SpvDecorationFPRoundingMode,
        floatingPointRoundingMode);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T1, typename T2>
SpvInst* emitOpDecorateCounterBuffer(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& target,
    const T2& counterBuffer)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(
        parent,
        inst,
        SpvOpDecorateId,
        target,
        SpvDecorationCounterBuffer,
        counterBuffer);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpDecorate
template<typename T>
SpvInst* emitOpDecorateUserSemantic(
    SpvInstParent* parent,
    IRInst* inst,
    const T& target,
    const UnownedStringSlice& semantic)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpDecorate, target, SpvDecorationUserSemantic, semantic);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T>
SpvInst* emitOpMemberDecorate(
    SpvInstParent* parent,
    IRInst* inst,
    const T& structureType,
    const SpvLiteralInteger& member,
    SpvDecoration decoration)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpMemberDecorate, structureType, member, decoration);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T>
SpvInst* emitOpMemberDecorateSpecId(
    SpvInstParent* parent,
    IRInst* inst,
    const T& structureType,
    const SpvLiteralInteger& member,
    const SpvLiteralInteger& specializationConstantID)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpMemberDecorate,
        structureType,
        member,
        SpvDecorationSpecId,
        specializationConstantID);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T>
SpvInst* emitOpMemberDecorateArrayStride(
    SpvInstParent* parent,
    IRInst* inst,
    const T& structureType,
    const SpvLiteralInteger& member,
    const SpvLiteralInteger& arrayStride)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpMemberDecorate,
        structureType,
        member,
        SpvDecorationArrayStride,
        arrayStride);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T>
SpvInst* emitOpMemberDecorateMatrixStride(
    SpvInstParent* parent,
    IRInst* inst,
    const T& structureType,
    const SpvLiteralInteger& member,
    const SpvLiteralInteger& matrixStride)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpMemberDecorate,
        structureType,
        member,
        SpvDecorationMatrixStride,
        matrixStride);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T>
SpvInst* emitOpMemberDecorateBuiltIn(
    SpvInstParent* parent,
    IRInst* inst,
    const T& structureType,
    const SpvLiteralInteger& member,
    SpvBuiltIn builtIn)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpMemberDecorate,
        structureType,
        member,
        SpvDecorationBuiltIn,
        builtIn);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T1, typename T2>
SpvInst* emitOpMemberDecorateUniformId(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& structureType,
    const SpvLiteralInteger& member,
    const T2& execution)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(
        parent,
        inst,
        SpvOpMemberDecorate,
        structureType,
        member,
        SpvDecorationUniformId,
        execution);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T>
SpvInst* emitOpMemberDecorateLocation(
    SpvInstParent* parent,
    IRInst* inst,
    const T& structureType,
    const SpvLiteralInteger& member,
    const SpvLiteralInteger& location)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpMemberDecorate,
        structureType,
        member,
        SpvDecorationLocation,
        location);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T>
SpvInst* emitOpMemberDecorateComponent(
    SpvInstParent* parent,
    IRInst* inst,
    const T& structureType,
    const SpvLiteralInteger& member,
    const SpvLiteralInteger& component)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpMemberDecorate,
        structureType,
        member,
        SpvDecorationComponent,
        component);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T>
SpvInst* emitOpMemberDecorateIndex(
    SpvInstParent* parent,
    IRInst* inst,
    const T& structureType,
    const SpvLiteralInteger& member,
    const SpvLiteralInteger& index)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpMemberDecorate,
        structureType,
        member,
        SpvDecorationIndex,
        index);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T>
SpvInst* emitOpMemberDecorateBinding(
    SpvInstParent* parent,
    IRInst* inst,
    const T& structureType,
    const SpvLiteralInteger& member,
    const SpvLiteralInteger& bindingPoint)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpMemberDecorate,
        structureType,
        member,
        SpvDecorationBinding,
        bindingPoint);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T>
SpvInst* emitOpMemberDecorateDescriptorSet(
    SpvInstParent* parent,
    IRInst* inst,
    const T& structureType,
    const SpvLiteralInteger& member,
    const SpvLiteralInteger& descriptorSet)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpMemberDecorate,
        structureType,
        member,
        SpvDecorationDescriptorSet,
        descriptorSet);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T>
SpvInst* emitOpMemberDecorateOffset(
    SpvInstParent* parent,
    IRInst* inst,
    const T& structureType,
    const SpvLiteralInteger& member,
    const SpvLiteralInteger& byteOffset)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpMemberDecorate,
        structureType,
        member,
        SpvDecorationOffset,
        byteOffset);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T>
SpvInst* emitOpMemberDecorateFPRoundingMode(
    SpvInstParent* parent,
    IRInst* inst,
    const T& structureType,
    const SpvLiteralInteger& member,
    SpvFPRoundingMode floatingPointRoundingMode)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpMemberDecorate,
        structureType,
        member,
        SpvDecorationFPRoundingMode,
        floatingPointRoundingMode);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T1, typename T2>
SpvInst* emitOpMemberDecorateCounterBuffer(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& structureType,
    const SpvLiteralInteger& member,
    const T2& counterBuffer)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(
        parent,
        inst,
        SpvOpMemberDecorate,
        structureType,
        member,
        SpvDecorationCounterBuffer,
        counterBuffer);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpMemberDecorate
template<typename T>
SpvInst* emitOpMemberDecorateUserSemantic(
    SpvInstParent* parent,
    IRInst* inst,
    const T& structureType,
    const SpvLiteralInteger& member,
    const UnownedStringSlice& semantic)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpMemberDecorate,
        structureType,
        member,
        SpvDecorationUserSemantic,
        semantic);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpVectorShuffle
template<typename T1, typename T2, typename T3>
SpvInst* emitOpVectorShuffle(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& vector1,
    const T3& vector2,
    ArrayView<SpvLiteralInteger> components)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(
        parent,
        inst,
        SpvOpVectorShuffle,
        idResultType,
        kResultID,
        vector1,
        vector2,
        components);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpCompositeConstruct
template<typename T, typename Ts>
SpvInst* emitOpCompositeConstruct(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    const Ts& constituents)
{
    static_assert(isSingular<T>);
    static_assert(isPlural<Ts>);
    return emitInst(parent, inst, SpvOpCompositeConstruct, idResultType, kResultID, constituents);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpCompositeConstruct
template<typename T, typename T1, typename T2>
SpvInst* emitOpCompositeConstruct(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    const T1& constituent1,
    const T2& constituent2)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpCompositeConstruct,
        idResultType,
        kResultID,
        constituent1,
        constituent2);
}

template<typename T, typename Ts>
SpvInst* emitOpConstantComposite(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    const Ts& constituents)
{
    static_assert(isSingular<T>);
    static_assert(isPlural<Ts>);
    return emitInst(parent, inst, SpvOpConstantComposite, idResultType, kResultID, constituents);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpCompositeExtract
template<typename T1, typename T2, Index N>
SpvInst* emitOpCompositeExtract(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& composite,
    const Array<SpvLiteralInteger, N>& indexes)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(
        parent,
        inst,
        SpvOpCompositeExtract,
        idResultType,
        kResultID,
        composite,
        indexes);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpCompositeInsert
template<typename T1, typename T2, typename T3, Index N>
SpvInst* emitOpCompositeInsert(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& object,
    const T3& composite,
    const Array<SpvLiteralInteger, N>& indexes)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);

    return emitInst(
        parent,
        inst,
        SpvOpCompositeInsert,
        idResultType,
        kResultID,
        object,
        composite,
        indexes);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpVectorExtractDynamic
template<typename T1, typename T2, typename T3>
SpvInst* emitOpVectorExtractDynamic(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& composite,
    const T3& index)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(
        parent,
        inst,
        SpvOpVectorExtractDynamic,
        idResultType,
        kResultID,
        composite,
        index);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpCopyObject
template<typename T1, typename T2>
SpvInst* emitOpCopyObject(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpCopyObject, idResultType, kResultID, operand);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpConvertFToU
template<typename T1, typename T2>
SpvInst* emitOpConvertFToU(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& floatValue)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpConvertFToU, idResultType, kResultID, floatValue);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpConvertFToS
template<typename T1, typename T2>
SpvInst* emitOpConvertFToS(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& floatValue)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpConvertFToS, idResultType, kResultID, floatValue);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpConvertSToF
template<typename T1, typename T2>
SpvInst* emitOpConvertSToF(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& signedValue)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpConvertSToF, idResultType, kResultID, signedValue);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpConvertUToF
template<typename T1, typename T2>
SpvInst* emitOpConvertUToF(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& unsignedValue)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpConvertUToF, idResultType, kResultID, unsignedValue);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpUConvert
template<typename T1, typename T2>
SpvInst* emitOpUConvert(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& unsignedValue)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpUConvert, idResultType, kResultID, unsignedValue);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSConvert
template<typename T1, typename T2>
SpvInst* emitOpSConvert(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& signedValue)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpSConvert, idResultType, kResultID, signedValue);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFConvert
template<typename T1, typename T2>
SpvInst* emitOpFConvert(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& floatValue)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpFConvert, idResultType, kResultID, floatValue);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpBitcast
template<typename T1, typename T2>
SpvInst* emitOpBitcast(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpBitcast, idResultType, kResultID, operand);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpCopyLogical
template<typename T1, typename T2>
SpvInst* emitOpCopyLogical(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpCopyLogical, idResultType, kResultID, operand);
}


// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSNegate
template<typename T1, typename T2>
SpvInst* emitOpSNegate(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpSNegate, idResultType, kResultID, operand);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFNegate
template<typename T1, typename T2>
SpvInst* emitOpFNegate(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpFNegate, idResultType, kResultID, operand);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpIAdd
template<typename T1, typename T2, typename T3>
SpvInst* emitOpIAdd(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpIAdd, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFAdd
template<typename T1, typename T2, typename T3>
SpvInst* emitOpFAdd(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpFAdd, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpISub
template<typename T1, typename T2, typename T3>
SpvInst* emitOpISub(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpISub, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFSub
template<typename T1, typename T2, typename T3>
SpvInst* emitOpFSub(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpFSub, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpIMul
template<typename T1, typename T2, typename T3>
SpvInst* emitOpIMul(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpIMul, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFMul
template<typename T1, typename T2, typename T3>
SpvInst* emitOpFMul(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpFMul, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpUDiv
template<typename T1, typename T2, typename T3>
SpvInst* emitOpUDiv(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpUDiv, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSDiv
template<typename T1, typename T2, typename T3>
SpvInst* emitOpSDiv(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpSDiv, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFDiv
template<typename T1, typename T2, typename T3>
SpvInst* emitOpFDiv(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpFDiv, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpUMod
template<typename T1, typename T2, typename T3>
SpvInst* emitOpUMod(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpUMod, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSRem
template<typename T1, typename T2, typename T3>
SpvInst* emitOpSRem(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpSRem, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFRem
template<typename T1, typename T2, typename T3>
SpvInst* emitOpFRem(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpFRem, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpIAddCarry
template<typename T1, typename T2, typename T3>
SpvInst* emitOpIAddCarry(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpIAddCarry, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpISubBorrow
template<typename T1, typename T2, typename T3>
SpvInst* emitOpISubBorrow(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpISubBorrow, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLogicalEqual
template<typename T1, typename T2, typename T3>
SpvInst* emitOpLogicalEqual(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpLogicalEqual, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLogicalNotEqual
template<typename T1, typename T2, typename T3>
SpvInst* emitOpLogicalNotEqual(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(
        parent,
        inst,
        SpvOpLogicalNotEqual,
        idResultType,
        kResultID,
        operand1,
        operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLogicalOr
template<typename T1, typename T2, typename T3>
SpvInst* emitOpLogicalOr(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpLogicalOr, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLogicalAnd
template<typename T1, typename T2, typename T3>
SpvInst* emitOpLogicalAnd(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpLogicalAnd, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLogicalNot
template<typename T1, typename T2>
SpvInst* emitOpLogicalNot(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpLogicalNot, idResultType, kResultID, operand);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpIEqual
template<typename T1, typename T2, typename T3>
SpvInst* emitOpIEqual(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpIEqual, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpINotEqual
template<typename T1, typename T2, typename T3>
SpvInst* emitOpINotEqual(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpINotEqual, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpUGreaterThan
template<typename T1, typename T2, typename T3>
SpvInst* emitOpUGreaterThan(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpUGreaterThan, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSGreaterThan
template<typename T1, typename T2, typename T3>
SpvInst* emitOpSGreaterThan(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpSGreaterThan, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpUGreaterThanEqual
template<typename T1, typename T2, typename T3>
SpvInst* emitOpUGreaterThanEqual(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(
        parent,
        inst,
        SpvOpUGreaterThanEqual,
        idResultType,
        kResultID,
        operand1,
        operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSGreaterThanEqual
template<typename T1, typename T2, typename T3>
SpvInst* emitOpSGreaterThanEqual(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(
        parent,
        inst,
        SpvOpSGreaterThanEqual,
        idResultType,
        kResultID,
        operand1,
        operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpULessThan
template<typename T1, typename T2, typename T3>
SpvInst* emitOpULessThan(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpULessThan, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSLessThan
template<typename T1, typename T2, typename T3>
SpvInst* emitOpSLessThan(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpSLessThan, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpULessThanEqual
template<typename T1, typename T2, typename T3>
SpvInst* emitOpULessThanEqual(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpULessThanEqual, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSLessThanEqual
template<typename T1, typename T2, typename T3>
SpvInst* emitOpSLessThanEqual(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpSLessThanEqual, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFOrdEqual
template<typename T1, typename T2, typename T3>
SpvInst* emitOpFOrdEqual(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpFOrdEqual, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFOrdNotEqual
template<typename T1, typename T2, typename T3>
SpvInst* emitOpFOrdNotEqual(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpFOrdNotEqual, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFOrdLessThan
template<typename T1, typename T2, typename T3>
SpvInst* emitOpFOrdLessThan(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpFOrdLessThan, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFOrdGreaterThan
template<typename T1, typename T2, typename T3>
SpvInst* emitOpFOrdGreaterThan(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(
        parent,
        inst,
        SpvOpFOrdGreaterThan,
        idResultType,
        kResultID,
        operand1,
        operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFOrdLessThanEqual
template<typename T1, typename T2, typename T3>
SpvInst* emitOpFOrdLessThanEqual(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(
        parent,
        inst,
        SpvOpFOrdLessThanEqual,
        idResultType,
        kResultID,
        operand1,
        operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFOrdGreaterThanEqual
template<typename T1, typename T2, typename T3>
SpvInst* emitOpFOrdGreaterThanEqual(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(
        parent,
        inst,
        SpvOpFOrdGreaterThanEqual,
        idResultType,
        kResultID,
        operand1,
        operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpShiftRightLogical
template<typename T1, typename T2, typename T3>
SpvInst* emitOpShiftRightLogical(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& base,
    const T3& shift)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpShiftRightLogical, idResultType, kResultID, base, shift);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpShiftRightArithmetic
template<typename T1, typename T2, typename T3>
SpvInst* emitOpShiftRightArithmetic(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& base,
    const T3& shift)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpShiftRightArithmetic, idResultType, kResultID, base, shift);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpShiftLeftLogical
template<typename T1, typename T2, typename T3>
SpvInst* emitOpShiftLeftLogical(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& base,
    const T3& shift)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpShiftLeftLogical, idResultType, kResultID, base, shift);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpBitwiseOr
template<typename T1, typename T2, typename T3>
SpvInst* emitOpBitwiseOr(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpBitwiseOr, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpBitwiseXor
template<typename T1, typename T2, typename T3>
SpvInst* emitOpBitwiseXor(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpBitwiseXor, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpBitwiseAnd
template<typename T1, typename T2, typename T3>
SpvInst* emitOpBitwiseAnd(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& operand1,
    const T3& operand2)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(parent, inst, SpvOpBitwiseAnd, idResultType, kResultID, operand1, operand2);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpBitReverse
template<typename T1, typename T2>
SpvInst* emitOpBitReverse(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& base)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpBitReverse, idResultType, kResultID, base);
}

// OpPhi elided, please use emitInst directly

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLoopMerge
template<typename T1, typename T2>
SpvInst* emitOpLoopMerge(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& mergeBlock,
    const T2& continueTarget,
    SpvLoopControlMask loopControl)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(parent, inst, SpvOpLoopMerge, mergeBlock, continueTarget, loopControl);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLoopMerge
template<typename T1, typename T2>
SpvInst* emitOpLoopMergeDependencyLength(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& mergeBlock,
    const T2& continueTarget,
    const SpvLiteralInteger& literalInteger)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(
        parent,
        inst,
        SpvOpLoopMerge,
        mergeBlock,
        continueTarget,
        SpvLoopControlDependencyLengthMask,
        literalInteger);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLoopMerge
template<typename T1, typename T2>
SpvInst* emitOpLoopMergeMinIterations(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& mergeBlock,
    const T2& continueTarget,
    const SpvLiteralInteger& literalInteger)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(
        parent,
        inst,
        SpvOpLoopMerge,
        mergeBlock,
        continueTarget,
        SpvLoopControlMinIterationsMask,
        literalInteger);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLoopMerge
template<typename T1, typename T2>
SpvInst* emitOpLoopMergeMaxIterations(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& mergeBlock,
    const T2& continueTarget,
    const SpvLiteralInteger& literalInteger)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(
        parent,
        inst,
        SpvOpLoopMerge,
        mergeBlock,
        continueTarget,
        SpvLoopControlMaxIterationsMask,
        literalInteger);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLoopMerge
template<typename T1, typename T2>
SpvInst* emitOpLoopMergeIterationMultiple(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& mergeBlock,
    const T2& continueTarget,
    const SpvLiteralInteger& literalInteger)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(
        parent,
        inst,
        SpvOpLoopMerge,
        mergeBlock,
        continueTarget,
        SpvLoopControlIterationMultipleMask,
        literalInteger);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLoopMerge
template<typename T1, typename T2>
SpvInst* emitOpLoopMergePeelCount(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& mergeBlock,
    const T2& continueTarget,
    const SpvLiteralInteger& literalInteger)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(
        parent,
        inst,
        SpvOpLoopMerge,
        mergeBlock,
        continueTarget,
        SpvLoopControlPeelCountMask,
        literalInteger);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLoopMerge
template<typename T1, typename T2>
SpvInst* emitOpLoopMergePartialCount(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& mergeBlock,
    const T2& continueTarget,
    const SpvLiteralInteger& literalInteger)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    return emitInst(
        parent,
        inst,
        SpvOpLoopMerge,
        mergeBlock,
        continueTarget,
        SpvLoopControlPartialCountMask,
        literalInteger);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSelectionMerge
template<typename T>
SpvInst* emitOpSelectionMerge(
    SpvInstParent* parent,
    IRInst* inst,
    const T& mergeBlock,
    SpvSelectionControlMask selectionControl)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpSelectionMerge, mergeBlock, selectionControl);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpLabel
SpvInst* emitOpLabel(SpvInstParent* parent, IRInst* inst)
{
    return emitInst(parent, inst, SpvOpLabel, kResultID);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpBranch
template<typename T>
SpvInst* emitOpBranch(SpvInstParent* parent, IRInst* inst, const T& targetLabel)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpBranch, targetLabel);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpBranchConditional
template<typename T1, typename T2, typename T3, Index N>
SpvInst* emitOpBranchConditional(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& condition,
    const T2& trueLabel,
    const T3& falseLabel,
    const Array<SpvLiteralInteger, N>& branchWeights)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    return emitInst(
        parent,
        inst,
        SpvOpBranchConditional,
        condition,
        trueLabel,
        falseLabel,
        branchWeights);
}

// OpSwitch elided, please use emitInst directly

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpKill
SpvInst* emitOpKill(SpvInstParent* parent, IRInst* inst)
{
    return emitInst(parent, inst, SpvOpKill);
}

SpvInst* emitOpDemoteToHelperInvocation(SpvInstParent* parent, IRInst* inst)
{
    return emitInst(parent, inst, SpvOpDemoteToHelperInvocation);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpReturn
SpvInst* emitOpReturn(SpvInstParent* parent, IRInst* inst)
{
    return emitInst(parent, inst, SpvOpReturn);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpReturnValue
template<typename T>
SpvInst* emitOpReturnValue(SpvInstParent* parent, IRInst* inst, const T& value)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpReturnValue, value);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpUnreachable
SpvInst* emitOpUnreachable(SpvInstParent* parent, IRInst* inst)
{
    return emitInst(parent, inst, SpvOpUnreachable);
}

// https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/EXT/SPV_EXT_fragment_shader_interlock.html#shaders-fragment-shader-interlock
SpvInst* emitOpBeginInvocationInterlockEXT(SpvInstParent* parent, IRInst* inst)
{
    return emitInst(parent, inst, SpvOpBeginInvocationInterlockEXT);
}

// https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/EXT/SPV_EXT_fragment_shader_interlock.html#shaders-fragment-shader-interlock
SpvInst* emitOpEndInvocationInterlockEXT(SpvInstParent* parent, IRInst* inst)
{
    return emitInst(parent, inst, SpvOpEndInvocationInterlockEXT);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpExecutionModeId
template<typename T>
SpvInst* emitOpExecutionModeId(
    SpvInstParent* parent,
    IRInst* inst,
    const T& entryPoint,
    SpvExecutionMode mode)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpExecutionModeId, entryPoint, mode);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpExecutionModeId
template<typename T>
SpvInst* emitOpExecutionModeIdLocalSize(
    SpvInstParent* parent,
    IRInst* inst,
    const T& entryPoint,
    const SpvLiteralInteger& xSize,
    const SpvLiteralInteger& ySize,
    const SpvLiteralInteger& zSize)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpExecutionModeId,
        entryPoint,
        SpvExecutionModeLocalSize,
        xSize,
        ySize,
        zSize);
}

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpExecutionModeId
template<typename T1, typename T2, typename T3, typename T4>
SpvInst* emitOpExecutionModeIdLocalSizeId(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& entryPoint,
    const T2& xSize,
    const T3& ySize,
    const T4& zSize)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    static_assert(isSingular<T4>);
    return emitInst(
        parent,
        inst,
        SpvOpExecutionModeId,
        entryPoint,
        SpvExecutionModeLocalSizeId,
        xSize,
        ySize,
        zSize);
}

template<typename T1, typename T2, typename T3, typename T4>
SpvInst* emitOpAtomicLoad(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& pointer,
    const T3& memory,
    const T4& semantics)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    static_assert(isSingular<T4>);
    return emitInst(
        parent,
        inst,
        SpvOpAtomicLoad,
        idResultType,
        kResultID,
        pointer,
        memory,
        semantics);
}

template<typename T1, typename T2, typename T3, typename T4>
SpvInst* emitOpAtomicStore(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& pointer,
    const T2& memory,
    const T3& semantics,
    const T4& value)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    static_assert(isSingular<T4>);
    return emitInst(parent, inst, SpvOpAtomicStore, pointer, memory, semantics, value);
}

template<typename T1, typename T2, typename T3, typename T4, typename T5>
SpvInst* emitOpAtomicExchange(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& pointer,
    const T3& memory,
    const T4& semantics,
    const T5& value)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    static_assert(isSingular<T4>);
    static_assert(isSingular<T5>);
    return emitInst(
        parent,
        inst,
        SpvOpAtomicExchange,
        idResultType,
        kResultID,
        pointer,
        memory,
        semantics,
        value);
}

template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
SpvInst* emitOpAtomicCompareExchange(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& pointer,
    const T3& memory,
    const T4& semanticsEqual,
    const T5& semanticsUnequal,
    const T6& value,
    const T7& comparator)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    static_assert(isSingular<T4>);
    static_assert(isSingular<T5>);
    static_assert(isSingular<T6>);
    static_assert(isSingular<T7>);

    return emitInst(
        parent,
        inst,
        SpvOpAtomicCompareExchange,
        idResultType,
        kResultID,
        pointer,
        memory,
        semanticsEqual,
        semanticsUnequal,
        value,
        comparator);
}

template<typename T1, typename T2, typename T3, typename T4, typename T5>
SpvInst* emitOpAtomicOp(
    SpvInstParent* parent,
    IRInst* inst,
    SpvOp op,
    const T1& idResultType,
    const T2& pointer,
    const T3& memory,
    const T4& semantics,
    const T5& value)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    static_assert(isSingular<T4>);
    static_assert(isSingular<T5>);
    return emitInst(parent, inst, op, idResultType, kResultID, pointer, memory, semantics, value);
}

template<typename T1, typename T2, typename T3, typename T4>
SpvInst* emitOpAtomicIIncrement(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& pointer,
    const T3& memory,
    const T4& semantics)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    static_assert(isSingular<T4>);
    return emitInst(
        parent,
        inst,
        SpvOpAtomicIIncrement,
        idResultType,
        kResultID,
        pointer,
        memory,
        semantics);
}

template<typename T1, typename T2, typename T3, typename T4>
SpvInst* emitOpAtomicIDecrement(
    SpvInstParent* parent,
    IRInst* inst,
    const T1& idResultType,
    const T2& pointer,
    const T3& memory,
    const T4& semantics)
{
    static_assert(isSingular<T1>);
    static_assert(isSingular<T2>);
    static_assert(isSingular<T3>);
    static_assert(isSingular<T4>);
    return emitInst(
        parent,
        inst,
        SpvOpAtomicIDecrement,
        idResultType,
        kResultID,
        pointer,
        memory,
        semantics);
}

// https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/AMD/SPV_AMDX_shader_enqueue.html#OpTypeNodePayloadArrayAMDX
template<typename T>
SpvInst* emitOpTypeNodePayloadArray(IRInst* inst, const T& type)
{
    static_assert(isSingular<T>);
    return emitInstMemoized(
        getSection(SpvLogicalSectionID::ConstantsAndTypes),
        inst,
        SpvOpTypeNodePayloadArrayAMDX,
        kResultID,
        type);
}

#endif // SLANG_IN_SPIRV_EMIT_CONTEXT
