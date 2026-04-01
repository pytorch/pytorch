// slang-ir-inst-defs.h

// clang-format off

#ifndef INST
#error Must #define `INST` before including `ir-inst-defs.h`
#endif

#ifndef INST_RANGE
#define INST_RANGE(BASE, FIRST, LAST) /* empty */
#endif

#define PARENT kIROpFlag_Parent
#define USE_OTHER kIROpFlag_UseOther
#define HOISTABLE kIROpFlag_Hoistable
#define GLOBAL kIROpFlag_Global

INST(Nop, nop, 0, 0)

/* Types */

    /* Basic Types */

    #define DEFINE_BASE_TYPE_INST(NAME) INST(NAME ## Type, NAME, 0, HOISTABLE)
    FOREACH_BASE_TYPE(DEFINE_BASE_TYPE_INST)
    #undef DEFINE_BASE_TYPE_INST
    INST(AfterBaseType, afterBaseType, 0, 0)

    INST_RANGE(BasicType, VoidType, AfterBaseType)

    /* StringTypeBase */
        INST(StringType, String, 0, HOISTABLE)
        INST(NativeStringType, NativeString, 0, HOISTABLE)
    INST_RANGE(StringTypeBase, StringType, NativeStringType)

    INST(CapabilitySetType, CapabilitySet, 0, HOISTABLE)

    INST(DynamicType, DynamicType, 0, HOISTABLE)

    INST(AnyValueType, AnyValueType, 1, HOISTABLE)

    INST(RawPointerType, RawPointerType, 0, HOISTABLE)
    INST(RTTIPointerType, RTTIPointerType, 1, HOISTABLE)
    INST(AfterRawPointerTypeBase, AfterRawPointerTypeBase, 0, 0)
    INST_RANGE(RawPointerTypeBase, RawPointerType, AfterRawPointerTypeBase)


    /* ArrayTypeBase */
        INST(ArrayType, Array, 2, HOISTABLE)
        INST(UnsizedArrayType, UnsizedArray, 1, HOISTABLE)
    INST_RANGE(ArrayTypeBase, ArrayType, UnsizedArrayType)

    INST(FuncType, Func, 0, HOISTABLE)
    INST(BasicBlockType, BasicBlock, 0, HOISTABLE)

    INST(VectorType, Vec, 2, HOISTABLE)
    INST(MatrixType, Mat, 4, HOISTABLE)

    INST(ConjunctionType, Conjunction, 0, HOISTABLE)
    INST(AttributedType, Attributed, 0, HOISTABLE)
    INST(ResultType, Result, 2, HOISTABLE)
    INST(OptionalType, Optional, 1, HOISTABLE)
    INST(EnumType, Enum, 1, PARENT)

    INST(DifferentialPairType, DiffPair, 1, HOISTABLE)
    INST(DifferentialPairUserCodeType, DiffPairUserCode, 1, HOISTABLE)
    INST(DifferentialPtrPairType, DiffRefPair, 1, HOISTABLE)
    INST_RANGE(DifferentialPairTypeBase, DifferentialPairType, DifferentialPtrPairType)

    INST(BackwardDiffIntermediateContextType, BwdDiffIntermediateCtxType, 1, HOISTABLE)

    INST(TensorViewType, TensorView, 1, HOISTABLE)
    INST(TorchTensorType, TorchTensor, 0, HOISTABLE)
    INST(ArrayListType, ArrayListVector, 1, HOISTABLE)

    INST(AtomicType, Atomic, 1, HOISTABLE)

    /* BindExistentialsTypeBase */

        // A `BindExistentials<B, T0,w0, T1,w1, ...>` represents
        // taking type `B` and binding each of its existential type
        // parameters, recursively, with the specified arguments,
        // where each `Ti, wi` pair represents the concrete type
        // and witness table to plug in for parameter `i`.
        //
        INST(BindExistentialsType, BindExistentials, 1, HOISTABLE)

        // An `BindInterface<B, T0, w0>` represents the special case
        // of a `BindExistentials` where the type `B` is known to be
        // an interface type.
        //
        INST(BoundInterfaceType, BoundInterface, 3, HOISTABLE)

    INST_RANGE(BindExistentialsTypeBase, BindExistentialsType, BoundInterfaceType)

    /* Rate */
        INST(ConstExprRate, ConstExpr, 0, HOISTABLE)
        INST(GroupSharedRate, GroupShared, 0, HOISTABLE)
        INST(ActualGlobalRate, ActualGlobalRate, 0, HOISTABLE)
    INST_RANGE(Rate, ConstExprRate, GroupSharedRate)

    INST(RateQualifiedType, RateQualified, 2, HOISTABLE)

    // Kinds represent the "types of types."
    // They should not really be nested under `IRType`
    // in the overall hierarchy, but we can fix that later.
    //
    /* Kind */
        INST(TypeKind, Type, 0, HOISTABLE)
        INST(TypeParameterPackKind, TypeParameterPack, 0, HOISTABLE)
        INST(RateKind, Rate, 0, HOISTABLE)
        INST(GenericKind, Generic, 0, HOISTABLE)
    INST_RANGE(Kind, TypeKind, GenericKind)

    /* PtrTypeBase */
        INST(PtrType, Ptr, 1, HOISTABLE)
        INST(RefType, Ref, 1, HOISTABLE)
        INST(ConstRefType, ConstRef, 1, HOISTABLE)
        // A `PsuedoPtr<T>` logically represents a pointer to a value of type
        // `T` on a platform that cannot support pointers. The expectation
        // is that the "pointer" will be legalized away by storing a value
        // of type `T` somewhere out-of-line.

        INST(PseudoPtrType, PseudoPtr, 1, HOISTABLE)

        /* OutTypeBase */
            INST(OutType, Out, 1, HOISTABLE)
            INST(InOutType, InOut, 1, HOISTABLE)
        INST_RANGE(OutTypeBase, OutType, InOutType)
    INST_RANGE(PtrTypeBase, PtrType, InOutType)


    // A ComPtr<T> type is treated as a opaque type that represents a reference-counted handle to a COM object.
    INST(ComPtrType, ComPtr, 1, HOISTABLE)
    // A NativePtr<T> type represents a native pointer to a managed resource.
    INST(NativePtrType, NativePtr, 1, HOISTABLE)

    // A DescriptorHandle<T> type represents a bindless handle to an opaue resource type.
    INST(DescriptorHandleType, DescriptorHandle, 1, HOISTABLE)

    // An AtomicUint is a placeholder type for a storage buffer, and will be mangled during compiling.
    INST(GLSLAtomicUintType, GLSLAtomicUint, 0, HOISTABLE)

    /* SamplerStateTypeBase */
        INST(SamplerStateType, SamplerState, 0, HOISTABLE)
        INST(SamplerComparisonStateType, SamplerComparisonState, 0, HOISTABLE)
    INST_RANGE(SamplerStateTypeBase, SamplerStateType, SamplerComparisonStateType)

    INST(DefaultBufferLayoutType, DefaultLayout, 0, HOISTABLE)
    INST(Std140BufferLayoutType, Std140Layout, 0, HOISTABLE)
    INST(Std430BufferLayoutType, Std430Layout, 0, HOISTABLE)
    INST(ScalarBufferLayoutType, ScalarLayout, 0, HOISTABLE)

    INST(SubpassInputType, SubpassInputType, 2, HOISTABLE)

    INST(TextureFootprintType, TextureFootprintType, 1, HOISTABLE)

    INST(TextureShape1DType, TextureShape1DType, 0, HOISTABLE)
    INST(TextureShape2DType, TextureShape1DType, 0, HOISTABLE)
    INST(TextureShape3DType, TextureShape1DType, 0, HOISTABLE)
    INST(TextureShapeCubeType, TextureShape1DType, 0, HOISTABLE)
    INST(TextureShapeBufferType, TextureShapeBufferType, 0, HOISTABLE)

    // TODO: Why do we have all this hierarchy here, when everything
    // that actually matters is currently nested under `TextureTypeBase`?
    /* ResourceTypeBase */
        /* ResourceType */
            /* TextureTypeBase */
                /* TextureType */
                INST(TextureType, TextureType, 8, HOISTABLE)
                /* GLSLImageType */
                INST(GLSLImageType, GLSLImageType, 0, USE_OTHER | HOISTABLE)
            INST_RANGE(TextureTypeBase, TextureType, GLSLImageType)
        INST_RANGE(ResourceType, TextureType, GLSLImageType)
    INST_RANGE(ResourceTypeBase, TextureType, GLSLImageType)

    /* UntypedBufferResourceType */
        /* ByteAddressBufferTypeBase */
            INST(HLSLByteAddressBufferType,                     ByteAddressBuffer,   0, HOISTABLE)
            INST(HLSLRWByteAddressBufferType,                   RWByteAddressBuffer, 0, HOISTABLE)
            INST(HLSLRasterizerOrderedByteAddressBufferType,    RasterizerOrderedByteAddressBuffer, 0, HOISTABLE)
        INST_RANGE(ByteAddressBufferTypeBase, HLSLByteAddressBufferType, HLSLRasterizerOrderedByteAddressBufferType)
        INST(RaytracingAccelerationStructureType, RaytracingAccelerationStructure, 0, HOISTABLE)
    INST_RANGE(UntypedBufferResourceType, HLSLByteAddressBufferType, RaytracingAccelerationStructureType)

    /* HLSLPatchType */
        INST(HLSLInputPatchType,    InputPatch,     2, HOISTABLE)
        INST(HLSLOutputPatchType,   OutputPatch,    2, HOISTABLE)
    INST_RANGE(HLSLPatchType, HLSLInputPatchType, HLSLOutputPatchType)

    INST(GLSLInputAttachmentType, GLSLInputAttachment, 0, HOISTABLE)

    /* BuiltinGenericType */
        /* HLSLStreamOutputType */
            INST(HLSLPointStreamType,       PointStream,    1, HOISTABLE)
            INST(HLSLLineStreamType,        LineStream,     1, HOISTABLE)
            INST(HLSLTriangleStreamType,    TriangleStream, 1, HOISTABLE)
        INST_RANGE(HLSLStreamOutputType, HLSLPointStreamType, HLSLTriangleStreamType)

        /* MeshOutputType */
            INST(VerticesType,   Vertices, 2, HOISTABLE)
            INST(IndicesType,    Indices,  2, HOISTABLE)
            INST(PrimitivesType, Primitives, 2, HOISTABLE)
        INST_RANGE(MeshOutputType, VerticesType, PrimitivesType)

        /* Metal Mesh Type */
            INST(MetalMeshType, metal::mesh, 5, HOISTABLE)
        /* Metal Mesh Grid Properties */
            INST(MetalMeshGridPropertiesType, mesh_grid_properties, 0, HOISTABLE)

        /* HLSLStructuredBufferTypeBase */
            INST(HLSLStructuredBufferType,                  StructuredBuffer,                   0, HOISTABLE)
            INST(HLSLRWStructuredBufferType,                RWStructuredBuffer,                 0, HOISTABLE)
            INST(HLSLRasterizerOrderedStructuredBufferType, RasterizerOrderedStructuredBuffer,  0, HOISTABLE)
            INST(HLSLAppendStructuredBufferType,            AppendStructuredBuffer,             0, HOISTABLE)
            INST(HLSLConsumeStructuredBufferType,           ConsumeStructuredBuffer,            0, HOISTABLE)
        INST_RANGE(HLSLStructuredBufferTypeBase, HLSLStructuredBufferType, HLSLConsumeStructuredBufferType)

        /* PointerLikeType */
            /* ParameterGroupType */
                /* UniformParameterGroupType */
                    INST(ConstantBufferType, ConstantBuffer, 1, HOISTABLE)
                    INST(TextureBufferType, TextureBuffer, 1, HOISTABLE)
                    INST(ParameterBlockType, ParameterBlock, 1, HOISTABLE)
                INST_RANGE(UniformParameterGroupType, ConstantBufferType, ParameterBlockType)
            
                /* VaryingParameterGroupType */
                    INST(GLSLInputParameterGroupType, GLSLInputParameterGroup, 0, HOISTABLE)
                    INST(GLSLOutputParameterGroupType, GLSLOutputParameterGroup, 0, HOISTABLE)
                INST_RANGE(VaryingParameterGroupType, GLSLInputParameterGroupType, GLSLOutputParameterGroupType)
                INST(GLSLShaderStorageBufferType, GLSLShaderStorageBuffer, 1, HOISTABLE)
            INST_RANGE(ParameterGroupType, ConstantBufferType, GLSLShaderStorageBufferType)
        INST_RANGE(PointerLikeType, ConstantBufferType, GLSLShaderStorageBufferType)
    INST_RANGE(BuiltinGenericType, HLSLPointStreamType, GLSLShaderStorageBufferType)

INST(RayQueryType, RayQuery, 1, HOISTABLE)
INST(HitObjectType, HitObject, 0, HOISTABLE)
INST(CoopVectorType, CoopVectorType, 2, HOISTABLE)
INST(CoopMatrixType, CoopMatrixType, 5, HOISTABLE)

// Opaque type that can be dynamically cast to other resource types.
INST(DynamicResourceType, DynamicResource, 0, HOISTABLE)

// A user-defined structure declaration at the IR level.
// Unlike in the AST where there is a distinction between
// a `StructDecl` and a `DeclRefType` that refers to it,
// at the IR level the struct declaration and the type
// are the same IR instruction.
//
// This is a parent instruction that holds zero or more
// `field` instructions.
//
INST(StructType, struct, 0, PARENT)
INST(ClassType, class, 0, PARENT)
INST(InterfaceType, interface, 0, GLOBAL)
INST(AssociatedType, associated_type, 0, HOISTABLE)
INST(ThisType, this_type, 0, HOISTABLE)
INST(RTTIType, rtti_type, 0, HOISTABLE)
INST(RTTIHandleType, rtti_handle_type, 0, HOISTABLE)
/*TupleTypeBase*/
    INST(TupleType, tuple_type, 0, HOISTABLE)
    INST(TypePack, TypePack, 0, HOISTABLE)
INST_RANGE(TupleTypeBase, TupleType, TypePack)
INST(TargetTupleType, TargetTuple, 0, HOISTABLE)
INST(ExpandTypeOrVal, ExpandTypeOrVal, 1, HOISTABLE)

// A type that identifies it's contained type as being emittable as `spirv_literal.
INST(SPIRVLiteralType, spirvLiteralType, 1, HOISTABLE)

// A TypeType-typed IRValue represents a IRType.
// It is used to represent a type parameter/argument in a generics.
INST(TypeType, type_t, 0, HOISTABLE)

/*IRWitnessTableTypeBase*/
    // An `IRWitnessTable` has type `WitnessTableType`.
    INST(WitnessTableType, witness_table_t, 1, HOISTABLE)
    // An integer type representing a witness table for targets where
    // witness tables are represented as integer IDs. This type is used
    // during the lower-generics pass while generating dynamic dispatch
    // code and will eventually lower into an uint type.
    INST(WitnessTableIDType, witness_table_id_t, 1, HOISTABLE)
INST_RANGE(WitnessTableTypeBase, WitnessTableType, WitnessTableIDType)
INST_RANGE(Type, VoidType, WitnessTableIDType)

/*IRGlobalValueWithCode*/
    /* IRGlobalValueWithParams*/
        INST(Func, func, 0, PARENT)
        INST(Generic, generic, 0, PARENT)
    INST_RANGE(GlobalValueWithParams, Func, Generic)

    INST(GlobalVar, global_var, 0, GLOBAL)
INST_RANGE(GlobalValueWithCode, Func, GlobalVar)

INST(GlobalParam, global_param, 0, GLOBAL)
INST(GlobalConstant, globalConstant, 0, GLOBAL)

INST(StructKey, key, 0, GLOBAL)
INST(GlobalGenericParam, global_generic_param, 0, GLOBAL)
INST(WitnessTable, witness_table, 0, HOISTABLE)

INST(IndexedFieldKey, indexedFieldKey, 2, HOISTABLE)

// A placeholder witness that ThisType implements the enclosing interface.
// Used only in interface definitions.
INST(ThisTypeWitness, thisTypeWitness, 1, 0)

// A placeholder witness for the fact that two types are equal.
INST(TypeEqualityWitness, TypeEqualityWitness, 2, HOISTABLE)

INST(GlobalHashedStringLiterals, global_hashed_string_literals, 0, 0)

INST(Module, module, 0, PARENT)

INST(Block, block, 0, PARENT)

/* IRConstant */
    INST(BoolLit, boolConst, 0, 0)
    INST(IntLit, integer_constant, 0, 0)
    INST(FloatLit, float_constant, 0, 0)
    INST(PtrLit, ptr_constant, 0, 0)
    INST(StringLit, string_constant, 0, 0)
    INST(BlobLit, string_constant, 0, 0)
    INST(VoidLit, void_constant, 0, 0)
INST_RANGE(Constant, BoolLit, VoidLit)

INST(CapabilityConjunction, capabilityConjunction, 0, HOISTABLE)
INST(CapabilityDisjunction, capabilityDisjunction, 0, HOISTABLE)
INST_RANGE(CapabilitySet, CapabilityConjunction, CapabilityDisjunction)

INST(undefined, undefined, 0, 0)

// A `defaultConstruct` operation creates an initialized
// value of the result type, and can only be used for types
// where default construction is a meaningful thing to do.
//
INST(DefaultConstruct, defaultConstruct, 0, 0)

INST(MakeDifferentialPair, MakeDiffPair, 2, 0)
INST(MakeDifferentialPairUserCode, MakeDiffPairUserCode, 2, 0)
INST(MakeDifferentialPtrPair, MakeDiffRefPair, 2, 0)
INST_RANGE(MakeDifferentialPairBase, MakeDifferentialPair, MakeDifferentialPtrPair)

INST(DifferentialPairGetDifferential, GetDifferential, 1, 0)
INST(DifferentialPairGetDifferentialUserCode, GetDifferentialUserCode, 1, 0)
INST(DifferentialPtrPairGetDifferential, GetDifferentialPtr, 1, 0)
INST_RANGE(DifferentialPairGetDifferentialBase, DifferentialPairGetDifferential, DifferentialPtrPairGetDifferential)

INST(DifferentialPairGetPrimal, GetPrimal, 1, 0)
INST(DifferentialPairGetPrimalUserCode, GetPrimalUserCode, 1, 0)
INST(DifferentialPtrPairGetPrimal, GetPrimalRef, 1, 0)
INST_RANGE(DifferentialPairGetPrimalBase, DifferentialPairGetPrimal, DifferentialPtrPairGetPrimal)

INST(Specialize, specialize, 2, HOISTABLE)
INST(LookupWitness, lookupWitness, 2, HOISTABLE)
INST(GetSequentialID, GetSequentialID, 1, HOISTABLE)
INST(BindGlobalGenericParam, bind_global_generic_param, 2, 0)
INST(AllocObj, allocObj, 0, 0)

INST(GlobalValueRef, globalValueRef, 1, 0)

INST(MakeUInt64, makeUInt64, 2, 0)
INST(MakeVector, makeVector, 0, 0)
INST(MakeMatrix, makeMatrix, 0, 0)
INST(MakeMatrixFromScalar, makeMatrixFromScalar, 1, 0)
INST(MatrixReshape, matrixReshape, 1, 0)
INST(VectorReshape, vectorReshape, 1, 0)
INST(MakeArray, makeArray, 0, 0)
INST(MakeArrayFromElement, makeArrayFromElement, 1, 0)
INST(MakeCoopVector, makeCoopVector, 0, 0)
INST(MakeCoopVectorFromValuePack, makeCoopVectorFromValuePack, 1, 0)
INST(MakeStruct, makeStruct, 0, 0)
INST(MakeTuple, makeTuple, 0, 0)
INST(MakeTargetTuple, makeTuple, 0, 0)
INST(MakeValuePack, makeValuePack, 0, 0)
INST(GetTargetTupleElement, getTargetTupleElement, 0, 0)
INST(GetTupleElement, getTupleElement, 2, 0)
INST(LoadResourceDescriptorFromHeap, LoadResourceDescriptorFromHeap, 1, 0)
INST(LoadSamplerDescriptorFromHeap, LoadSamplerDescriptorFromHeap, 1, 0)
INST(MakeCombinedTextureSamplerFromHandle, MakeCombinedTextureSamplerFromHandle, 1, 0)
INST(MakeWitnessPack, MakeWitnessPack, 0, HOISTABLE)
INST(Expand, Expand, 1, 0)
INST(Each, Each, 1, HOISTABLE)
INST(MakeResultValue, makeResultValue, 1, 0)
INST(MakeResultError, makeResultError, 1, 0)
INST(IsResultError, isResultError, 1, 0)
INST(GetResultError, getResultError, 1, 0)
INST(GetResultValue, getResultValue, 1, 0)
INST(GetOptionalValue, getOptionalValue, 1, 0)
INST(OptionalHasValue, optionalHasValue, 1, 0)
INST(MakeOptionalValue, makeOptionalValue, 1, 0)
INST(MakeOptionalNone, makeOptionalNone, 1, 0)
INST(CombinedTextureSamplerGetTexture, CombinedTextureSamplerGetTexture, 1, 0)
INST(CombinedTextureSamplerGetSampler, CombinedTextureSamplerGetSampler, 1, 0)
INST(Call, call, 1, 0)

INST(RTTIObject, rtti_object, 0, 0)
INST(Alloca, alloca, 1, 0)

INST(UpdateElement, updateElement, 2, 0)
INST(DetachDerivative, detachDerivative, 1, 0)

INST(BitfieldExtract, bitfieldExtract, 3, 0)
INST(BitfieldInsert, bitfieldInsert, 4, 0)

INST(PackAnyValue, packAnyValue, 1, 0)
INST(UnpackAnyValue, unpackAnyValue, 1, 0)

INST(WitnessTableEntry, witness_table_entry, 2, 0)
INST(InterfaceRequirementEntry, interface_req_entry, 2, GLOBAL)

// An inst to represent the workgroup size of the calling entry point.
// We will materialize this inst during `translateGlobalVaryingVar`.
INST(GetWorkGroupSize, GetWorkGroupSize, 0, HOISTABLE)

// An inst that returns the current stage of the calling entry point.
INST(GetCurrentStage, GetCurrentStage, 0, 0)

INST(Param, param, 0, 0)
INST(StructField, field, 2, 0)
INST(Var, var, 0, 0)

INST(Load, load, 1, 0)
INST(Store, store, 2, 0)

// Atomic Operations
INST(AtomicLoad, atomicLoad, 1, 0)
INST(AtomicStore, atomicStore, 2, 0)
INST(AtomicExchange, atomicExchange, 2, 0)
INST(AtomicCompareExchange, atomicCompareExchange, 3, 0)
INST(AtomicAdd, atomicAdd, 2, 0)
INST(AtomicSub, atomicSub, 2, 0)
INST(AtomicAnd, atomicAnd, 2, 0)
INST(AtomicOr, atomicOr, 2, 0)
INST(AtomicXor, atomicXor, 2, 0)
INST(AtomicMin, atomicMin, 2, 0)
INST(AtomicMax, atomicMax, 2, 0)
INST(AtomicInc, atomicInc, 1, 0)
INST(AtomicDec, atomicDec, 1, 0)

// Produced and removed during backward auto-diff pass as a temporary placeholder representing the
// currently accumulated derivative to pass to some dOut argument in a nested call.
INST(LoadReverseGradient, LoadReverseGradient, 1, 0)

// Produced and removed during backward auto-diff pass as a temporary placeholder containing the
// primal and accumulated derivative values to pass to an inout argument in a nested call.
INST(ReverseGradientDiffPairRef, ReverseGradientDiffPairRef, 2, 0)

// Produced and removed during backward auto-diff pass. This inst is generated by the splitting step
// to represent a reference to an inout parameter for use in the primal part of the computation.
INST(PrimalParamRef, PrimalParamRef, 1, 0)

// Produced and removed during backward auto-diff pass. This inst is generated by the splitting step
// to represent a reference to an inout parameter for use in the back-prop part of the computation.
INST(DiffParamRef, DiffParamRef, 1, 0)

// Check that the value is a differential null value.
INST(IsDifferentialNull, IsDifferentialNull, 1, 0)

INST(FieldExtract, get_field, 2, 0)
INST(FieldAddress, get_field_addr, 2, 0)

INST(GetElement, getElement, 2, 0)
INST(GetElementPtr, getElementPtr, 2, 0)
// Pointer offset: computes pBase + offset_in_elements
INST(GetOffsetPtr, getOffsetPtr, 2, 0) 
INST(GetAddr, getAddr, 1, 0)

INST(CastDynamicResource, castDynamicResource, 1, 0)

// Get an unowned NativeString from a String.
INST(getNativeStr, getNativeStr, 1, 0)

// Make String from a NativeString.
INST(MakeString, makeString, 1, 0)

// Get a native ptr from a ComPtr or RefPtr
INST(GetNativePtr, getNativePtr, 1, 0)

// Get a write reference to a managed ptr var (operand must be Ptr<ComPtr<T>> or Ptr<RefPtr<T>>).
INST(GetManagedPtrWriteRef, getManagedPtrWriteRef, 1, 0)

// Attach a managedPtr var to a NativePtr without changing its ref count.
INST(ManagedPtrAttach, ManagedPtrAttach, 1, 0)

// Attach a managedPtr var to a NativePtr without changing its ref count.
INST(ManagedPtrDetach, ManagedPtrDetach, 1, 0)

// "Subscript" an image at a pixel coordinate to get pointer
INST(ImageSubscript, imageSubscript, 2, 0)

// Load from an Image.
INST(ImageLoad, imageLoad, 2, 0)
// Store into an Image.
INST(ImageStore, imageStore, 3, 0)

// Load (almost) arbitrary-type data from a byte-address buffer
//
// %dst = byteAddressBufferLoad(%buffer, %offset, %alignment)
//
// where
// - `buffer` is a value of some `ByteAddressBufferTypeBase` type
// - `offset` is an `int`
// - `alignment` is an `int`
// - `dst` is a value of some type containing only ordinary data
//
INST(ByteAddressBufferLoad, byteAddressBufferLoad, 3, 0)

// Store (almost) arbitrary-type data to a byte-address buffer
//
// byteAddressBufferLoad(%buffer, %offset, %alignment, %src)
//
// where
// - `buffer` is a value of some `ByteAddressBufferTypeBase` type
// - `offset` is an `int`
// - `alignment` is an `int`
// - `src` is a value of some type containing only ordinary data
//
INST(ByteAddressBufferStore, byteAddressBufferStore, 4, 0)

// Load data from a structured buffer
//
// %dst = structuredBufferLoad(%buffer, %index)
//
// where
// - `buffer` is a value of some `StructuredBufferTypeBase` type with element type T
// - `offset` is an `int`
// - `dst` is a value of type T
//
INST(StructuredBufferLoad, structuredBufferLoad, 2, 0)
INST(StructuredBufferLoadStatus, structuredBufferLoadStatus, 3, 0)
INST(RWStructuredBufferLoad, rwstructuredBufferLoad, 2, 0)
INST(RWStructuredBufferLoadStatus, rwstructuredBufferLoadStatus, 3, 0)

// Store data to a structured buffer
//
// structuredBufferLoad(%buffer, %offset, %src)
//
// where
// - `buffer` is a value of some `StructuredBufferTypeBase` type with element type T
// - `offset` is an `int`
// - `src` is a value of type T
//
INST(RWStructuredBufferStore, rwstructuredBufferStore, 3, 0)

INST(RWStructuredBufferGetElementPtr, rwstructuredBufferGetElementPtr, 2, 0)

// Append/Consume-StructuredBuffer operations
INST(StructuredBufferAppend, StructuredBufferAppend, 1, 0)
INST(StructuredBufferConsume, StructuredBufferConsume, 1, 0)
INST(StructuredBufferGetDimensions, StructuredBufferGetDimensions, 1, 0)

// Resource qualifiers for dynamically varying index
INST(NonUniformResourceIndex, nonUniformResourceIndex, 1, 0)

INST(GetNaturalStride, getNaturalStride, 1, 0)

INST(MeshOutputRef, meshOutputRef, 2, 0)
INST(MeshOutputSet, meshOutputSet, 3, 0)

// only two parameters as they are effectively static
// TODO: make them reference the _slang_mesh object directly
INST(MetalSetVertex, metalSetVertex, 2, 0)
INST(MetalSetPrimitive, metalSetPrimitive, 2, 0)
INST(MetalSetIndices, metalSetIndices, 2, 0)

INST(MetalCastToDepthTexture, MetalCastToDepthTexture, 1, 0)

// Construct a vector from a scalar
//
// %dst = MakeVectorFromScalar %T %N %val
//
// where
// - `T` is a `Type`
// - `N` is a (compile-time) `Int`
// - `val` is a `T`
// - dst is a `Vec<T,N>`
//
INST(MakeVectorFromScalar, MakeVectorFromScalar, 3, 0)

// A swizzle of a vector:
//
// %dst = swizzle %src %idx0 %idx1 ...
//
// where:
// - `src` is a vector<T,N>
// - `dst` is a vector<T,M>
// - `idx0` through `idx[M-1]` are literal integers
//
INST(swizzle, swizzle, 1, 0)

// Setting a vector via swizzle
//
// %dst = swizzle %base %src %idx0 %idx1 ...
//
// where:
// - `base` is a vector<T,N>
// - `dst` is a vector<T,N>
// - `src` is a vector<T,M>
// - `idx0` through `idx[M-1]` are literal integers
//
// The semantics of the op is:
//
//     dst = base;
//     for(ii : 0 ... M-1 )
//         dst[ii] = src[idx[ii]];
//
INST(swizzleSet, swizzleSet, 2, 0)

// Store to memory with a swizzle
//
// TODO: eventually this should be reduced to just
// a write mask by moving the actual swizzle to the RHS.
//
// swizzleStore %dst %src %idx0 %idx1 ...
//
// where:
// - `dst` is a vector<T,N>
// - `src` is a vector<T,M>
// - `idx0` through `idx[M-1]` are literal integers
//
// The semantics of the op is:
//
//     for(ii : 0 ... M-1 )
//         dst[ii] = src[idx[ii]];
//
INST(SwizzledStore, swizzledStore, 2, 0)


/* IRTerminatorInst */

    INST(Return, return_val, 1, 0)
    INST(Yield, yield, 1, 0)
    /* IRUnconditionalBranch */
        // unconditionalBranch <target>
        INST(unconditionalBranch, unconditionalBranch, 1, 0)

        // loop <target> <breakLabel> <continueLabel>
        INST(loop, loop, 3, 0)
    INST_RANGE(UnconditionalBranch, unconditionalBranch, loop)

    /* IRConditionalbranch */

        // conditionalBranch <condition> <trueBlock> <falseBlock>
INST(conditionalBranch, conditionalBranch, 3, 0)

// ifElse <condition> <trueBlock> <falseBlock> <mergeBlock>
INST(ifElse, ifElse, 4, 0)
INST_RANGE(ConditionalBranch, conditionalBranch, ifElse)

INST(Throw, throw, 1, 0)
// tryCall <successBlock> <failBlock> <callee> <args>...
INST(TryCall, tryCall, 3, 0)
// switch <val> <break> <default> <caseVal1> <caseBlock1> ...
INST(Switch, switch, 3, 0)
// target_switch <break> <targetName1> <block1> ...
INST(TargetSwitch, targetSwitch, 1, 0)

// A generic asm inst has an return semantics that terminates the control flow.
INST(GenericAsm, GenericAsm, 1, 0)

/* IRUnreachable */
INST(MissingReturn, missingReturn, 0, 0)
INST(Unreachable, unreachable, 0, 0)
INST_RANGE(Unreachable, MissingReturn, Unreachable)

INST(Defer, defer, 3, 0)

INST_RANGE(TerminatorInst, Return, Defer)

INST(discard, discard, 0, 0)

INST(RequirePrelude, RequirePrelude, 1, 0)
INST(RequireTargetExtension, RequireTargetExtension, 1, 0)
INST(RequireComputeDerivative, RequireComputeDerivative, 0, 0)
INST(StaticAssert, StaticAssert, 2, 0)
INST(Printf, Printf, 1, 0)

// Quad control execution modes.
INST(RequireMaximallyReconverges, RequireMaximallyReconverges, 0, 0)
INST(RequireQuadDerivatives, RequireQuadDerivatives, 0, 0)

// TODO: We should consider splitting the basic arithmetic/comparison
// ops into cases for signed integers, unsigned integers, and floating-point
// values, to better match downstream targets that want to treat them
// all differently (e.g., SPIR-V).

INST(Add, add, 2, 0)
INST(Sub, sub, 2, 0)
INST(Mul, mul, 2, 0)
INST(Div, div, 2, 0)

// Remainder of division.
//
// Note: this is distinct from modulus, and we should have a separate
// opcode for `mod` if we ever need to support it.
//
INST(IRem, irem, 2, 0) // integer (signed or unsigned)
INST(FRem, frem, 2, 0) // floating-point

INST(Lsh, shl, 2, 0)
INST(Rsh, shr, 2, 0)

INST(Eql, cmpEQ, 2, 0)
INST(Neq, cmpNE, 2, 0)
INST(Greater, cmpGT, 2, 0)
INST(Less, cmpLT, 2, 0)
INST(Geq, cmpGE, 2, 0)
INST(Leq, cmpLE, 2, 0)

INST(BitAnd, and, 2, 0)
INST(BitXor, xor, 2, 0)
INST(BitOr, or , 2, 0)

INST(And, logicalAnd, 2, 0)
INST(Or, logicalOr, 2, 0)

INST(Neg, neg, 1, 0)
INST(Not, not, 1, 0)
INST(BitNot, bitnot, 1, 0)

INST(Select, select, 3, 0)

INST(CheckpointObject, checkpointObj, 1, 0)
INST(LoopExitValue, loopExitValue, 1, 0)

INST(GetStringHash, getStringHash, 1, 0)

INST(WaveGetActiveMask, waveGetActiveMask, 0, 0)

/// trueMask = waveMaskBallot(mask, condition)
INST(WaveMaskBallot, waveMaskBallot, 2, 0)

/// matchMask = waveMaskBallot(mask, value)
INST(WaveMaskMatch, waveMaskMatch, 2, 0)

// Texture sampling operation of the form `t.Sample(s,u)`
INST(Sample, sample, 3, 0)

INST(SampleGrad, sampleGrad, 4, 0)

INST(GroupMemoryBarrierWithGroupSync, GroupMemoryBarrierWithGroupSync, 0, 0)

INST(ControlBarrier, ControlBarrier, 0, 0)

// GPU_FOREACH loop of the form 
INST(GpuForeach, gpuForeach, 3, 0)

// Wrapper for OptiX intrinsics used to load and store ray payload data using
// a pointer represented by two payload registers.
INST(GetOptiXRayPayloadPtr, getOptiXRayPayloadPtr, 0, 0)

// Wrapper for OptiX intrinsics used to load a single hit attribute
// Takes two arguments: the type (either float or int), and the hit 
// attribute index
INST(GetOptiXHitAttribute, getOptiXHitAttribute, 2, 0)

// Wrapper for OptiX intrinsics used to load shader binding table record data
// using a pointer. 
INST(GetOptiXSbtDataPtr, getOptiXSbtDataPointer, 0, 0)

INST(GetVulkanRayTracingPayloadLocation, GetVulkanRayTracingPayloadLocation, 1, 0)

INST(GetLegalizedSPIRVGlobalParamAddr, GetLegalizedSPIRVGlobalParamAddr, 1, 0)

INST(GetPerVertexInputArray, GetPerVertexInputArray, 1, HOISTABLE)
INST(ResolveVaryingInputRef, ResolveVaryingInputRef, 1, HOISTABLE)

INST(ForceVarIntoStructTemporarily, ForceVarIntoStructTemporarily, 1, 0)
INST(ForceVarIntoRayPayloadStructTemporarily, ForceVarIntoRayPayloadStructTemporarily, 1, 0)
INST_RANGE(ForceVarIntoStructTemporarily, ForceVarIntoStructTemporarily, ForceVarIntoRayPayloadStructTemporarily)

INST(MetalAtomicCast, MetalAtomicCast, 1, 0)

INST(IsTextureAccess, IsTextureAccess, 1, 0)
INST(IsTextureScalarAccess, IsTextureScalarAccess, 1, 0)
INST(IsTextureArrayAccess, IsTextureArrayAccess, 1, 0)
INST(ExtractTextureFromTextureAccess, ExtractTextureFromTextureAccess, 1, 0)
INST(ExtractCoordFromTextureAccess, ExtractCoordFromTextureAccess, 1, 0)
INST(ExtractArrayCoordFromTextureAccess, ExtractArrayCoordFromTextureAccess, 1, 0)

INST(MakeArrayList, makeArrayList, 0, 0)
INST(MakeTensorView, makeTensorView, 0, 0)
INST(AllocateTorchTensor, allocTorchTensor, 0, 0)
INST(TorchGetCudaStream, TorchGetCudaStream, 0, 0)
INST(TorchTensorGetView, TorchTensorGetView, 0, 0)

INST(AllocateOpaqueHandle, allocateOpaqueHandle, 0, 0)

    // Return the register index thtat a resource is bound to.
    INST(GetRegisterIndex, getRegisterIndex, 1, 0)

    // Return the registe space that a resource is bound to.
    INST(GetRegisterSpace, getRegisterSpace, 1, 0)

INST_RANGE(BindingQuery, GetRegisterIndex, GetRegisterSpace)

/* Decoration */

    INST(HighLevelDeclDecoration,           highLevelDecl,          1, 0)
    INST(LayoutDecoration,                  layout,                 1, 0)
    INST(BranchDecoration,                  branch,                 0, 0)
    INST(FlattenDecoration,                 flatten,                0, 0)
    INST(LoopControlDecoration,             loopControl,            1, 0)
    INST(LoopMaxItersDecoration,            loopMaxIters,           1, 0)
    INST(LoopExitPrimalValueDecoration,     loopExitPrimalValue,    2, 0)
    INST(IntrinsicOpDecoration, intrinsicOp, 1, 0)
    /* TargetSpecificDecoration */
        INST(TargetDecoration,              target,                 1, 0)
        INST(TargetIntrinsicDecoration,     targetIntrinsic,        2, 0)
        INST_RANGE(TargetSpecificDefinitionDecoration, TargetDecoration, TargetIntrinsicDecoration)
        INST(RequirePreludeDecoration, requirePrelude, 2, 0)
    INST_RANGE(TargetSpecificDecoration, TargetDecoration, RequirePreludeDecoration)
    INST(GLSLOuterArrayDecoration,          glslOuterArray,         1, 0)
    
    INST(TargetSystemValueDecoration,       TargetSystemValue,      2, 0)

    INST(InterpolationModeDecoration,       interpolationMode,      1, 0)
    INST(NameHintDecoration,                nameHint,               1, 0)

    INST(PhysicalTypeDecoration,            PhysicalType,           1, 0)

    // Mark an address instruction as aligned to a specific byte boundary.
    INST(AlignedAddressDecoration,          AlignedAddressDecoration, 1, 0)

    // Marks a type as being used as binary interface (e.g. shader parameters).
    // This prevents the legalizeEmptyType() pass from eliminating it on C++/CUDA targets.
    INST(BinaryInterfaceTypeDecoration,     BinaryInterfaceType, 0, 0)

    /**  The decorated _instruction_ is transitory. Such a decoration should NEVER be found on an output instruction a module. 
        Typically used mark an instruction so can be specially handled - say when creating a IRConstant literal, and the payload of 
        needs to be special cased for lookup. */
    INST(TransitoryDecoration,              transitory,             0, 0)

    // The result witness table that the functon's return type is a subtype of an interface.
    // This is used to keep track of the original witness table in a function that used to
    // return an existential value but now returns a concrete type after specialization.
    INST(ResultWitnessDecoration,           ResultWitness,          1, 0)

    INST(VulkanRayPayloadDecoration,        vulkanRayPayload,       0, 0)
    INST(VulkanRayPayloadInDecoration,      vulkanRayPayloadIn,       0, 0)
    INST(VulkanHitAttributesDecoration,     vulkanHitAttributes,    0, 0)
    INST(VulkanHitObjectAttributesDecoration, vulkanHitObjectAttributes, 0, 0)

    INST(GlobalVariableShadowingGlobalParameterDecoration, GlobalVariableShadowingGlobalParameterDecoration, 2, 0)

    INST(RequireSPIRVVersionDecoration,     requireSPIRVVersion,    1, 0)
    INST(RequireGLSLVersionDecoration,      requireGLSLVersion,     1, 0)
    INST(RequireGLSLExtensionDecoration,    requireGLSLExtension,   1, 0)
    INST(RequireWGSLExtensionDecoration,    requireWGSLExtension,   1, 0)
    INST(RequireCUDASMVersionDecoration,    requireCUDASMVersion,   1, 0)
    INST(RequireCapabilityAtomDecoration,   requireCapabilityAtom, 1, 0)

    INST(HasExplicitHLSLBindingDecoration, HasExplicitHLSLBinding, 0, 0)

    INST(DefaultValueDecoration,            DefaultValue,           1, 0)
    INST(ReadNoneDecoration,                readNone,               0, 0)
    INST(VulkanCallablePayloadDecoration,   vulkanCallablePayload,  0, 0)
    INST(VulkanCallablePayloadInDecoration, vulkanCallablePayloadIn,  0, 0)
    INST(EarlyDepthStencilDecoration,       earlyDepthStencil,      0, 0)
    INST(PreciseDecoration,                 precise,                0, 0)
    INST(PublicDecoration,                  public,                 0, 0)
    INST(HLSLExportDecoration,              hlslExport,             0, 0)
    INST(DownstreamModuleExportDecoration,  downstreamModuleExport, 0, 0)
    INST(DownstreamModuleImportDecoration,  downstreamModuleImport, 0, 0)
    INST(PatchConstantFuncDecoration,       patchConstantFunc,      1, 0)
    INST(MaxTessFactorDecoration,           maxTessFactor,          1, 0)
    INST(OutputControlPointsDecoration,     outputControlPoints,    1, 0)
    INST(OutputTopologyDecoration,          outputTopology,         2, 0)
    INST(PartitioningDecoration,            partioning,             1, 0)
    INST(DomainDecoration,                  domain,                 1, 0)
    INST(MaxVertexCountDecoration,          maxVertexCount,         1, 0)
    INST(InstanceDecoration,                instance,               1, 0)
    INST(NumThreadsDecoration,              numThreads,             3, 0)
    INST(WaveSizeDecoration,                waveSize,               1, 0)

    INST(AvailableInDownstreamIRDecoration, availableInDownstreamIR, 1, 0)

        // Added to IRParam parameters to an entry point
    /* GeometryInputPrimitiveTypeDecoration */
        INST(PointInputPrimitiveTypeDecoration,  pointPrimitiveType,     0, 0)
        INST(LineInputPrimitiveTypeDecoration,   linePrimitiveType,      0, 0)
        INST(TriangleInputPrimitiveTypeDecoration, trianglePrimitiveType, 0, 0)
        INST(LineAdjInputPrimitiveTypeDecoration,  lineAdjPrimitiveType,  0, 0)
        INST(TriangleAdjInputPrimitiveTypeDecoration, triangleAdjPrimitiveType, 0, 0)
    INST_RANGE(GeometryInputPrimitiveTypeDecoration, PointInputPrimitiveTypeDecoration, TriangleAdjInputPrimitiveTypeDecoration)

    INST(StreamOutputTypeDecoration,       streamOutputTypeDecoration,    1, 0)

        /// An `[entryPoint]` decoration marks a function that represents a shader entry point
    INST(EntryPointDecoration,              entryPoint,             2, 0)

    INST(CudaKernelDecoration,              CudaKernel,             0, 0)
    INST(CudaHostDecoration,                CudaHost,               0, 0)
    INST(TorchEntryPointDecoration,         TorchEntryPoint,        0, 0)
    INST(AutoPyBindCudaDecoration,          AutoPyBindCUDA,         0, 0)
    INST(CudaKernelForwardDerivativeDecoration,          CudaKernelFwdDiffRef,         0, 0)
    INST(CudaKernelBackwardDerivativeDecoration,         CudaKernelBwdDiffRef,         0, 0)
    INST(AutoPyBindExportInfoDecoration,    PyBindExportFuncInfo,   0, 0)
    INST(PyExportDecoration,    PyExportDecoration,   0, 0)
    
        /// Used to mark parameters that are moved from entry point parameters to global params as coming from the entry point.
    INST(EntryPointParamDecoration,         entryPointParam,        0, 0)

        /// A `[dependsOn(x)]` decoration indicates that the parent instruction depends on `x`
        /// even if it does not otherwise reference it.
    INST(DependsOnDecoration,               dependsOn,              1, 0)

        /// A `[keepAlive]` decoration marks an instruction that should not be eliminated.
    INST(KeepAliveDecoration,              keepAlive,             0, 0)

        /// A `[NoSideEffect]` decoration marks a callee to be side-effect free.
    INST(NoSideEffectDecoration,           noSideEffect, 0, 0)

    INST(BindExistentialSlotsDecoration, bindExistentialSlots, 0, 0)

        /// A `[format(f)]` decoration specifies that the format of an image should be `f`
    INST(FormatDecoration, format, 1, 0)

        /// An `[unsafeForceInlineEarly]` decoration specifies that calls to this function should be inline after initial codegen
    INST(UnsafeForceInlineEarlyDecoration, unsafeForceInlineEarly, 0, 0)

        /// A `[ForceInline]` decoration indicates the callee should be inlined by the Slang compiler.
    INST(ForceInlineDecoration, ForceInline, 0, 0)

        /// A `[ForceUnroll]` decoration indicates the loop should be unrolled by the Slang compiler.
    INST(ForceUnrollDecoration, ForceUnroll, 0, 0)

        /// A `[SizeAndAlignment(l,s,a)]` decoration is attached to a type to indicate that is has size `s` and alignment `a` under layout rules `l`.
    INST(SizeAndAlignmentDecoration, SizeAndAlignment, 3, 0)

        /// A `[Offset(l, o)]` decoration is attached to a field to indicate that it has offset `o` in the parent type under layout rules `l`.
    INST(OffsetDecoration, Offset, 2, 0)

    /* LinkageDecoration */
        INST(ImportDecoration, import, 1, 0)
        INST(ExportDecoration, export, 1, 0)
    INST_RANGE(LinkageDecoration, ImportDecoration, ExportDecoration)

        /// Mark a global variable as a target builtin variable.
    INST(TargetBuiltinVarDecoration, TargetBuiltinVar, 1, 0)

        /// Marks an inst as coming from an `extern` symbol defined in the user code.
    INST(UserExternDecoration, UserExtern, 0, 0)

        /// An extern_cpp decoration marks the inst to emit its name without mangling for C++ interop.
    INST(ExternCppDecoration, externCpp, 1, 0)

        // An externC decoration marks a function should be emitted inside an extern "C" block.
    INST(ExternCDecoration, externC, 0, 0)

        /// An dllImport decoration marks a function as imported from a DLL. Slang will generate dynamic function loading logic to use this function at runtime.
    INST(DllImportDecoration, dllImport, 2, 0)
        /// An dllExport decoration marks a function as an export symbol. Slang will generate a native wrapper function that is exported to DLL.
    INST(DllExportDecoration, dllExport, 1, 0)
        /// An cudaDeviceExport decoration marks a function to be exported as a cuda __device__ function.
    INST(CudaDeviceExportDecoration, cudaDeviceExport, 1, 0)

        /// Marks an interface as a COM interface declaration.
    INST(ComInterfaceDecoration, COMInterface, 0, 0)

        /// Attaches a name to this instruction so that it can be identified
        /// later in the compiler reliably
    INST(KnownBuiltinDecoration, KnownBuiltinDecoration, 1, 0)

    /* Decorations for RTTI objects */
    INST(RTTITypeSizeDecoration, RTTI_typeSize, 1, 0)
    INST(AnyValueSizeDecoration, AnyValueSize, 1, 0)
    INST(SpecializeDecoration, SpecializeDecoration, 0, 0)
    INST(SequentialIDDecoration, SequentialIDDecoration, 1, 0)
    INST(DynamicDispatchWitnessDecoration, DynamicDispatchWitnessDecoration, 0, 0)
    INST(StaticRequirementDecoration, StaticRequirementDecoration, 0, 0)
    INST(DispatchFuncDecoration, DispatchFuncDecoration, 1, 0)
    INST(TypeConstraintDecoration, TypeConstraintDecoration, 1, 0)

    
    INST(BuiltinDecoration, BuiltinDecoration, 0, 0)

        /// The decorated instruction requires NVAPI to be included via prelude when compiling for D3D.
    INST(RequiresNVAPIDecoration, requiresNVAPI, 0, 0)

        /// The decorated instruction is part of the NVAPI "magic" and should always use its original name
    INST(NVAPIMagicDecoration, nvapiMagic, 1, 0)

        /// A decoration that applies to an entire IR module, and indicates the register/space binding
        /// that the NVAPI shader parameter intends to use.
    INST(NVAPISlotDecoration, nvapiSlot, 2, 0)

        /// Applie to an IR function and signals that inlining should not be performed unless unavoidable.
    INST(NoInlineDecoration, noInline, 0, 0)
    INST(NoRefInlineDecoration, noRefInline, 0, 0)

    INST(DerivativeGroupQuadDecoration, DerivativeGroupQuad, 0, 0)
    INST(DerivativeGroupLinearDecoration, DerivativeGroupLinear, 0, 0)

    INST(MaximallyReconvergesDecoration, MaximallyReconverges, 0, 0)
    INST(QuadDerivativesDecoration, QuadDerivatives, 0, 0)
    INST(RequireFullQuadsDecoration, RequireFullQuads, 0, 0)

        // Marks a type to be non copyable, causing SSA pass to skip turning variables of the the type into SSA values.
    INST(NonCopyableTypeDecoration, nonCopyable, 0, 0)

        // Marks a value to be dynamically uniform.
    INST(DynamicUniformDecoration, DynamicUniform, 0, 0)

        /// A call to the decorated function should always be folded into its use site.
    INST(AlwaysFoldIntoUseSiteDecoration, alwaysFold, 0, 0)

    INST(GlobalOutputDecoration, output, 0, 0)
    INST(GlobalInputDecoration, input, 0, 0)
    INST(GLSLLocationDecoration, glslLocation, 1, 0)
    INST(GLSLOffsetDecoration, glslOffset, 1, 0)
    INST(VkStructOffsetDecoration, vkStructOffset, 1, 0)
    INST(PayloadDecoration, payload, 0, 0)
    INST(RayPayloadDecoration, raypayload, 0, 0)

    /* Mesh Shader outputs */
        INST(VerticesDecoration, vertices, 1, 0)
        INST(IndicesDecoration, indices, 1, 0)
        INST(PrimitivesDecoration, primitives, 1, 0)
    INST_RANGE(MeshOutputDecoration, VerticesDecoration, PrimitivesDecoration)
    INST(HLSLMeshPayloadDecoration, payload, 0, 0)
    INST(GLSLPrimitivesRateDecoration, perprimitive, 0, 0)
        // Marks an inst that represents the gl_Position output.
    INST(GLPositionOutputDecoration, PositionOutput, 0, 0)
        // Marks an inst that represents the gl_Position input.
    INST(GLPositionInputDecoration, PositionInput, 0, 0)

        // Marks a fragment shader input as per-vertex.
    INST(PerVertexDecoration, PerVertex, 0, 0)

    /* StageAccessDecoration */
        INST(StageReadAccessDecoration, stageReadAccess, 0, 0)
        INST(StageWriteAccessDecoration, stageWriteAccess, 0, 0)
    INST_RANGE(StageAccessDecoration, StageReadAccessDecoration, StageWriteAccessDecoration)

    INST(SemanticDecoration, semantic, 2, 0)
    INST(ConstructorDecoration, constructor, 1, 0)
    INST(MethodDecoration, method, 0, 0)
    INST(PackOffsetDecoration, packoffset, 2, 0)
    INST(SpecializationConstantDecoration, SpecializationConstantDecoration, 1, 0)

        // Reflection metadata for a shader parameter that provides the original type name.
    INST(UserTypeNameDecoration, UserTypeName, 1, 0)
        // Reflection metadata for a shader parameter that refers to the associated counter buffer of a UAV.
    INST(CounterBufferDecoration, CounterBuffer, 1, 0)

    INST(RequireSPIRVDescriptorIndexingExtensionDecoration, RequireSPIRVDescriptorIndexingExtensionDecoration, 0, 0)
    INST(SPIRVOpDecoration, spirvOpDecoration, 1, 0)

        /// Decorated function is marked for the forward-mode differentiation pass.
    INST(ForwardDifferentiableDecoration, forwardDifferentiable, 0, 0)

        /// Decorates a auto-diff transcribed value with the original value that the inst is transcribed from.
    INST(AutoDiffOriginalValueDecoration, AutoDiffOriginalValueDecoration, 1, 0)

        /// Decorates a type as auto-diff builtin type.
    INST(AutoDiffBuiltinDecoration, AutoDiffBuiltinDecoration, 0, 0)

        /// Used by the auto-diff pass to hold a reference to the
        /// generated derivative function.
    INST(ForwardDerivativeDecoration, fwdDerivative, 1, 0)

        /// Used by the auto-diff pass to hold a reference to the
        /// generated derivative function.
    INST(BackwardDifferentiableDecoration, backwardDifferentiable, 1, 0)

        /// Used by the auto-diff pass to hold a reference to the
        /// primal substitute function.
    INST(PrimalSubstituteDecoration, primalSubstFunc, 1, 0)

        /// Decorations to associate an original function with compiler generated backward derivative functions.
    INST(BackwardDerivativePrimalDecoration, backwardDiffPrimalReference, 1, 0)
    INST(BackwardDerivativePropagateDecoration, backwardDiffPropagateReference, 1, 0)
    INST(BackwardDerivativeIntermediateTypeDecoration, backwardDiffIntermediateTypeReference, 1, 0)
    INST(BackwardDerivativeDecoration, backwardDiffReference, 1, 0)

    INST(UserDefinedBackwardDerivativeDecoration, userDefinedBackwardDiffReference, 1, 0)
    INST(BackwardDerivativePrimalContextDecoration, BackwardDerivativePrimalContextDecoration, 1, 0)
    INST(BackwardDerivativePrimalReturnDecoration, BackwardDerivativePrimalReturnDecoration, 1, 0)

        // Mark a parameter as autodiff primal context.
    INST(PrimalContextDecoration, PrimalContextDecoration, 0, 0)
    INST(LoopCounterDecoration, loopCounterDecoration, 0, 0)
    INST(LoopCounterUpdateDecoration, loopCounterUpdateDecoration, 0, 0)

    /* Auto-diff inst decorations */
        /// Used by the auto-diff pass to mark insts that compute
        /// a primal value.
        INST(PrimalInstDecoration, primalInstDecoration, 0, 0)

        /// Used by the auto-diff pass to mark insts that compute
        /// a differential value.
        INST(DifferentialInstDecoration, diffInstDecoration, 1, 0)

        /// Used by the auto-diff pass to mark insts that compute
        /// BOTH a differential and a primal value.
        INST(MixedDifferentialInstDecoration, mixedDiffInstDecoration, 1, 0)

        INST(RecomputeBlockDecoration, RecomputeBlockDecoration, 0, 0)
    INST_RANGE(AutodiffInstDecoration, PrimalInstDecoration, RecomputeBlockDecoration)

        /// Used by the auto-diff pass to mark insts whose result is stored
        /// in an intermediary struct for reuse in backward propagation phase.
    INST(PrimalValueStructKeyDecoration, primalValueKey, 1, 0)

        /// Used by the auto-diff pass to mark the primal element type of an
        /// forward-differentiated updateElement inst.
    INST(PrimalElementTypeDecoration, primalElementType, 1, 0)

        /// Used by the auto-diff pass to mark the differential type of an intermediate context field.
    INST(IntermediateContextFieldDifferentialTypeDecoration, IntermediateContextFieldDifferentialTypeDecoration, 1, 0)

        /// Used by the auto-diff pass to hold a reference to a
        /// differential member of a type in its associated differential type.
    INST(DerivativeMemberDecoration, derivativeMemberDecoration, 1, 0)

        /// Treat a function as differentiable function
    INST(TreatAsDifferentiableDecoration, treatAsDifferentiableDecoration, 0, 0)

        /// Treat a call to arbitrary function as a differentiable call.
    INST(TreatCallAsDifferentiableDecoration, treatCallAsDifferentiableDecoration, 0, 0)

        /// Mark a call as explicitly calling a differentiable function.
    INST(DifferentiableCallDecoration, differentiableCallDecoration, 0, 0)

        /// Mark a type as being eligible for trimming if necessary. If
        /// any fields don't have any effective loads from them, they can be 
        /// removed.
        ///
    INST(OptimizableTypeDecoration, optimizableTypeDecoration, 0, 0)

        /// Informs the DCE pass to ignore side-effects on this call for
        /// the purposes of dead code elimination, even if the call does have
        /// side-effects.
        ///
    INST(IgnoreSideEffectsDecoration, ignoreSideEffectsDecoration, 0, 0)

        /// Hint that the result from a call to the decorated function should be stored in backward prop function.
    INST(PreferCheckpointDecoration, PreferCheckpointDecoration, 0, 0)

        /// Hint that the result from a call to the decorated function should be recomputed in backward prop function.
    INST(PreferRecomputeDecoration, PreferRecomputeDecoration, 0, 0)

        /// Hint that a struct is used for reverse mode checkpointing
    INST(CheckpointIntermediateDecoration, CheckpointIntermediateDecoration, 1, 0)

    INST_RANGE(CheckpointHintDecoration, PreferCheckpointDecoration, PreferRecomputeDecoration)

        /// Marks a function whose return value is never dynamic uniform.
    INST(NonDynamicUniformReturnDecoration, NonDynamicUniformReturnDecoration, 0, 0)

        /// Marks a class type as a COM interface implementation, which enables
        /// the witness table to be easily picked up by emit.
    INST(COMWitnessDecoration, COMWitnessDecoration, 1, 0)

    /* Differentiable Type Dictionary */
    INST(DifferentiableTypeDictionaryDecoration, DifferentiableTypeDictionaryDecoration, 0, PARENT)

        /// Overrides the floating mode for the target function
    INST(FloatingPointModeOverrideDecoration, FloatingPointModeOverride, 1, 0)

        /// Recognized by SPIRV-emit pass so we can emit a SPIRV `BufferBlock` decoration.
    INST(SPIRVBufferBlockDecoration, spvBufferBlock, 0, 0)

        /// Decorates an inst with a debug source location (IRDebugSource, IRIntLit(line), IRIntLit(col)).
    INST(DebugLocationDecoration, DebugLocation, 3, 0)

        /// Recognized by SPIRV-emit pass so we can emit a SPIRV `Block` decoration.
    INST(SPIRVBlockDecoration, spvBlock, 0, 0)

        /// Decorates a SPIRV-inst as `NonUniformResource` to guarantee non-uniform index lookup of
        /// - a resource within an array of resources via IRGetElement.
        /// - an IRLoad that takes a pointer within a memory buffer via IRGetElementPtr.
        /// - an IRIntCast to a resource that is casted from signed to unsigned or viceversa.
        /// - an IRGetElementPtr itself when using the pointer on an intrinsic operation.
    INST(SPIRVNonUniformResourceDecoration, NonUniformResource, 0, 0)

        // Stores flag bits of which memory qualifiers an object has
    INST(MemoryQualifierSetDecoration, MemoryQualifierSetDecoration, 1, 0)

        /// Marks a function as one which access a bitfield with the specified
        /// backing value key, width and offset
    INST(BitFieldAccessorDecoration, BitFieldAccessorDecoration, 3, 0)

 INST_RANGE(Decoration, HighLevelDeclDecoration, BitFieldAccessorDecoration)

    //

// A `makeExistential(v : C, w) : I` instruction takes a value `v` of type `C`
// and produces a value of interface type `I` by using the witness `w` which
// shows that `C` conforms to `I`.
//
INST(MakeExistential,                   makeExistential,                2, 0)
// A `MakeExistentialWithRTTI(v, w, t)` is the same with `MakeExistential`,
// but with the type of `v` being an explict operand.
INST(MakeExistentialWithRTTI,           makeExistentialWithRTTI,        3, 0)

// A 'CreateExistentialObject<I>(typeID, T)` packs user-provided `typeID` and a
// value of any type, and constructs an existential value of type `I`.
INST(CreateExistentialObject,           createExistentialObject,        2, 0)

// A `wrapExistential(v, T0,w0, T1,w0) : T` instruction is similar to `makeExistential`.
// but applies to a value `v` that is of type `BindExistentials(T, T0,w0, ...)`. The
// result of the `wrapExistentials` operation is a value of type `T`, allowing us to
// "smuggle" a value of specialized type into computations that expect an unspecialized type.
//
INST(WrapExistential,                   wrapExistential,                1, 0)

// A `GetValueFromBoundInterface` takes a `BindInterface<I, T, w0>` value and returns the
// value of concrete type `T` value that is being stored.
//
INST(GetValueFromBoundInterface,        getValueFromBoundInterface,     1, 0)

INST(ExtractExistentialValue,           extractExistentialValue,        1, 0)
INST(ExtractExistentialType,            extractExistentialType,         1, HOISTABLE)
INST(ExtractExistentialWitnessTable,    extractExistentialWitnessTable, 1, HOISTABLE)

INST(ExtractTaggedUnionTag,             extractTaggedUnionTag,      1, 0)
INST(ExtractTaggedUnionPayload,         extractTaggedUnionPayload,  1, 0)

INST(BuiltinCast,                       BuiltinCast,                1, 0)
INST(BitCast,                           bitCast,                    1, 0)
INST(Reinterpret,                       reinterpret,                1, 0)
INST(Unmodified,                        unmodified,                1, 0)
INST(OutImplicitCast,                   outImplicitCast,           1, 0)
INST(InOutImplicitCast,                 inOutImplicitCast,         1, 0)
INST(IntCast, intCast, 1, 0)
INST(FloatCast, floatCast, 1, 0)
INST(CastIntToFloat, castIntToFloat, 1, 0)
INST(CastFloatToInt, castFloatToInt, 1, 0)
INST(CastPtrToBool, CastPtrToBool, 1, 0)
INST(CastPtrToInt, CastPtrToInt, 1, 0)
INST(CastIntToPtr, CastIntToPtr, 1, 0)
INST(CastToVoid, castToVoid, 1, 0)
INST(PtrCast, PtrCast, 1, 0)
INST(CastEnumToInt, CastEnumToInt, 1, 0)
INST(CastIntToEnum, CastIntToEnum, 1, 0)
INST(EnumCast, EnumCast, 1, 0)
INST(CastUInt2ToDescriptorHandle, CastUInt2ToDescriptorHandle, 1, 0)
INST(CastDescriptorHandleToUInt2, CastDescriptorHandleToUInt2, 1, 0)

// Represents a no-op cast to convert a resource pointer to a resource on targets where the resource handles are already concrete types.
INST(CastDescriptorHandleToResource, CastDescriptorHandleToResource, 1, 0)

INST(TreatAsDynamicUniform, TreatAsDynamicUniform, 1, 0)

INST(SizeOf,                            sizeOf,                     1, 0)
INST(AlignOf,                           alignOf,                    1, 0)
INST(CountOf, countOf, 1, 0)

INST(GetArrayLength,                    GetArrayLength,             1, 0)
INST(IsType, IsType, 3, 0)
INST(TypeEquals, TypeEquals, 2, 0)
INST(IsInt, IsInt, 1, 0)
INST(IsBool, IsBool, 1, 0)
INST(IsFloat, IsFloat, 1, 0)
INST(IsHalf, IsHalf, 1, 0)
INST(IsUnsignedInt, IsUnsignedInt, 1, 0)
INST(IsSignedInt, IsSignedInt, 1, 0)
INST(IsVector, IsVector, 1, 0)
INST(GetDynamicResourceHeap, GetDynamicResourceHeap, 0, HOISTABLE)

INST(ForwardDifferentiate,                   ForwardDifferentiate,            1, 0)

// Produces the primal computation of backward derivatives, will return an intermediate context for
// backward derivative func.
INST(BackwardDifferentiatePrimal,            BackwardDifferentiatePrimal,     1, 0)

// Produces the actual backward derivative propagate function, using the intermediate context returned by the
// primal func produced from `BackwardDifferentiatePrimal`.
INST(BackwardDifferentiatePropagate,         BackwardDifferentiatePropagate,  1, 0)

// Represents the conceptual backward derivative function. Only produced by lower-to-ir and will be
// replaced with `BackwardDifferentiatePrimal` and `BackwardDifferentiatePropagate`.
INST(BackwardDifferentiate, BackwardDifferentiate, 1, 0)

INST(PrimalSubstitute, PrimalSubstitute, 1, 0)

INST(DispatchKernel, DispatchKernel, 3, 0)
INST(CudaKernelLaunch, CudaKernelLaunch, 6, 0)

// Converts other resources (such as ByteAddressBuffer) to the equivalent StructuredBuffer
INST(GetEquivalentStructuredBuffer,     getEquivalentStructuredBuffer, 1, 0)

// Gets a T[] pointer to the underlying data of a StructuredBuffer etc...
INST(GetStructuredBufferPtr,     getStructuredBufferPtr, 1, 0)
// Gets a uint[] pointer to the underlying data of a ByteAddressBuffer etc...
INST(GetUntypedBufferPtr,     getUntypedBufferPtr, 1, 0)

/* Layout */
    INST(VarLayout, varLayout, 1, HOISTABLE)

    /* TypeLayout */
        INST(TypeLayoutBase, typeLayout, 0, HOISTABLE)
        INST(ParameterGroupTypeLayout, parameterGroupTypeLayout, 2, HOISTABLE)
        INST(ArrayTypeLayout, arrayTypeLayout, 1, HOISTABLE)
        INST(StreamOutputTypeLayout, streamOutputTypeLayout, 1, HOISTABLE)
        INST(MatrixTypeLayout, matrixTypeLayout, 1, HOISTABLE)
        INST(ExistentialTypeLayout, existentialTypeLayout, 0, HOISTABLE)
        INST(StructTypeLayout, structTypeLayout, 0, HOISTABLE)
        INST(TupleTypeLayout, tupleTypeLayout, 0, HOISTABLE)
        INST(StructuredBufferTypeLayout, structuredBufferTypeLayout, 1, HOISTABLE)
        // TODO(JS): Ideally we'd have the layout to the pointed to value type (ie 1 instead of 0 here). But to avoid infinite recursion we don't.
        INST(PointerTypeLayout, ptrTypeLayout, 0, HOISTABLE)
    INST_RANGE(TypeLayout, TypeLayoutBase, PointerTypeLayout)

    INST(EntryPointLayout, EntryPointLayout, 1, HOISTABLE)
INST_RANGE(Layout, VarLayout, EntryPointLayout)

/* Attr */
    INST(PendingLayoutAttr, pendingLayout, 1, HOISTABLE)
    INST(StageAttr, stage, 1, HOISTABLE)
    INST(StructFieldLayoutAttr, fieldLayout, 2, HOISTABLE)
    INST(TupleFieldLayoutAttr, fieldLayout, 1, HOISTABLE)
    INST(CaseTypeLayoutAttr, caseLayout, 1, HOISTABLE)
    INST(UNormAttr, unorm, 0, HOISTABLE)
    INST(SNormAttr, snorm, 0, HOISTABLE)
    INST(NoDiffAttr, no_diff, 0, HOISTABLE)
    INST(NonUniformAttr, nonuniform, 0, HOISTABLE)
    INST(AlignedAttr, Aligned, 1, HOISTABLE)

    /* SemanticAttr */
        INST(UserSemanticAttr, userSemantic, 2, HOISTABLE)
        INST(SystemValueSemanticAttr, systemValueSemantic, 2, HOISTABLE)
    INST_RANGE(SemanticAttr, UserSemanticAttr, SystemValueSemanticAttr)
    /* LayoutResourceInfoAttr */
        INST(TypeSizeAttr, size, 2, HOISTABLE)
        INST(VarOffsetAttr, offset, 2, HOISTABLE)
    INST_RANGE(LayoutResourceInfoAttr, TypeSizeAttr, VarOffsetAttr)
    INST(FuncThrowTypeAttr, FuncThrowType, 1, HOISTABLE)
    
INST_RANGE(Attr, PendingLayoutAttr, FuncThrowTypeAttr)

/* Liveness */
    INST(LiveRangeStart, liveRangeStart, 2, 0)
    INST(LiveRangeEnd, liveRangeEnd, 0, 0)
INST_RANGE(LiveRangeMarker, LiveRangeStart, LiveRangeEnd)

/* IRSpecialization */
INST(SpecializationDictionaryItem, SpecializationDictionaryItem, 0, 0)
INST(GenericSpecializationDictionary, GenericSpecializationDictionary, 0, PARENT)
INST(ExistentialFuncSpecializationDictionary, ExistentialFuncSpecializationDictionary, 0, PARENT)
INST(ExistentialTypeSpecializationDictionary, ExistentialTypeSpecializationDictionary, 0, PARENT)

/* Differentiable Type Dictionary */
INST(DifferentiableTypeDictionaryItem, DifferentiableTypeDictionaryItem, 0, 0)

/* Differentiable Type Annotation (for run-time types)*/
INST(DifferentiableTypeAnnotation, DifferentiableTypeAnnotation, 2, HOISTABLE)

INST(BeginFragmentShaderInterlock, BeginFragmentShaderInterlock, 0, 0)
INST(EndFragmentShaderInterlock, BeginFragmentShaderInterlock, 0, 0)

/* DebugInfo */
INST(DebugSource, DebugSource, 2, HOISTABLE)
INST(DebugLine, DebugLine, 5, 0)
INST(DebugVar, DebugVar, 4, 0)
INST(DebugValue, DebugValue, 2, 0)

/* Embedded Precompiled Libraries */
INST(EmbeddedDownstreamIR, EmbeddedDownstreamIR, 2, 0)

/* Inline assembly */

INST(SPIRVAsm, SPIRVAsm, 0, PARENT)
INST(SPIRVAsmInst, SPIRVAsmInst, 1, 0)
    // These instruction serve to inform the backend precisely how to emit each
    // instruction, consider the difference between emitting a literal integer
    // and a reference to a literal integer instruction
    //
    // A literal string or 32-bit integer to be passed as operands
    INST(SPIRVAsmOperandLiteral, SPIRVAsmOperandLiteral, 1, HOISTABLE)
    // A reference to a slang IRInst, either a value or a type
    // This isn't hoistable, as we sometimes need to change the used value and
    // instructions around the specific asm block
    INST(SPIRVAsmOperandInst, SPIRVAsmOperandInst, 1, 0)    
    INST(SPIRVAsmOperandConvertTexel, SPIRVAsmOperandConvertTexel, 1, 0)
    //a late resolving type to handle the case of ray objects (resolving late due to constexpr data requirment)
    INST(SPIRVAsmOperandRayPayloadFromLocation, SPIRVAsmOperandRayPayloadFromLocation, 1, 0)
    INST(SPIRVAsmOperandRayAttributeFromLocation, SPIRVAsmOperandRayAttributeFromLocation, 1, 0)
    INST(SPIRVAsmOperandRayCallableFromLocation, SPIRVAsmOperandRayCallableFromLocation, 1, 0)
    // A named enumerator, the value is stored as a constant operand
    // It may have a second operand, which if present is a type with which to
    // construct a constant id to pass, instead of a literal constant
    INST(SPIRVAsmOperandEnum, SPIRVAsmOperandEnum, 1, HOISTABLE)
    // A reference to a builtin variable.
    INST(SPIRVAsmOperandBuiltinVar, SPIRVAsmOperandBuiltinVar, 1, HOISTABLE)
    // A reference to the glsl450 instruction set.
    INST(SPIRVAsmOperandGLSL450Set, SPIRVAsmOperandGLSL450Set, 0, HOISTABLE)
    INST(SPIRVAsmOperandDebugPrintfSet, SPIRVAsmOperandDebugPrintfSet, 0, HOISTABLE)
    // A string which is given a unique ID in the backend, used to refer to
    // results of other instrucions in the same asm block
    INST(SPIRVAsmOperandId, SPIRVAsmOperandId, 1, HOISTABLE)
    // A special instruction which marks the place to insert the generated
    // result operand
    INST(SPIRVAsmOperandResult, SPIRVAsmOperandResult, 0, HOISTABLE)
    // A special instruction which represents a type directed truncation
    // operation where extra components are dropped
    INST(SPIRVAsmOperandTruncate, __truncate, 0, HOISTABLE)

    // A special instruction which represents an ID of an entry point that references the current function.
    INST(SPIRVAsmOperandEntryPoint, __entryPoint, 0, HOISTABLE)

    // A type function which returns the result type of sampling an image of
    // this component type
    INST(SPIRVAsmOperandSampledType, __sampledType, 1, HOISTABLE)

    // A type function which returns the equivalent OpTypeImage type of sampled image value
    INST(SPIRVAsmOperandImageType, __imageType, 1, HOISTABLE)

    // A type function which returns the equivalent OpTypeImage type of sampled image value
    INST(SPIRVAsmOperandSampledImageType, __sampledImageType, 1, HOISTABLE)

INST_RANGE(SPIRVAsmOperand, SPIRVAsmOperandLiteral, SPIRVAsmOperandSampledImageType)


#undef PARENT
#undef USE_OTHER
#undef INST_RANGE
#undef INST
