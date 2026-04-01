// slang-ast-type.h
#pragma once

#include "slang-ast-base.h"
#include "slang-ast-type.h.fiddle"

FIDDLE()
namespace Slang
{

// Syntax class definitions for types.

// The type of a reference to an overloaded name
FIDDLE()
class OverloadGroupType : public Type
{
    FIDDLE(...)
    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();
};

// The type of an initializer-list expression (before it has
// been coerced to some other type)
FIDDLE()
class InitializerListType : public Type
{
    FIDDLE(...)
    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();
};

// The type of an expression that was erroneous
FIDDLE()
class ErrorType : public Type
{
    FIDDLE(...)
    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

// The bottom/empty type that has no values.
FIDDLE()
class BottomType : public Type
{
    FIDDLE(...)
    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

// A type that takes the form of a reference to some declaration
FIDDLE()
class DeclRefType : public Type
{
    FIDDLE(...)
    static Type* create(ASTBuilder* astBuilder, DeclRef<Decl> declRef);

    DeclRef<Decl> getDeclRef() const { return DeclRef<Decl>(as<DeclRefBase>(getOperand(0))); }
    DeclRefBase* getDeclRefBase() const { return as<DeclRefBase>(getOperand(0)); }

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);

    DeclRefType(DeclRefBase* declRefBase) { setOperands(declRefBase); }
};

template<typename T>
DeclRef<T> isDeclRefTypeOf(Val* type)
{
    if (auto declRefType = as<DeclRefType>(type))
    {
        return declRefType->getDeclRef().template as<T>();
    }
    return DeclRef<T>();
}

bool isTypePack(Type* type);
bool isAbstractTypePack(Type* type);

// Base class for types that can be used in arithmetic expressions
FIDDLE(abstract)
class ArithmeticExpressionType : public DeclRefType
{
    FIDDLE(...)
    BasicExpressionType* getScalarType();

    // Overrides should be public so base classes can access
    BasicExpressionType* _getScalarTypeOverride();
};

FIDDLE()
class BasicExpressionType : public ArithmeticExpressionType
{
    FIDDLE(...)
    BaseType getBaseType() const;

    // Overrides should be public so base classes can access
    BasicExpressionType* _getScalarTypeOverride();

    BasicExpressionType(DeclRefBase* inDeclRef) { setOperands(inDeclRef); }
};

// Base type for things that are built in to the compiler,
// and will usually have special behavior or a custom
// mapping to the IR level.
FIDDLE(abstract)
class BuiltinType : public DeclRefType
{
    FIDDLE(...)
};

FIDDLE(abstract)
class DataLayoutType : public BuiltinType
{
    FIDDLE(...)
};

FIDDLE()
class IBufferDataLayoutType : public BuiltinType
{
    FIDDLE(...)
};

FIDDLE()
class DefaultDataLayoutType : public DataLayoutType
{
    FIDDLE(...)
};

FIDDLE()
class Std430DataLayoutType : public DataLayoutType
{
    FIDDLE(...)
};

FIDDLE()
class Std140DataLayoutType : public DataLayoutType
{
    FIDDLE(...)
};

FIDDLE()
class ScalarDataLayoutType : public DataLayoutType
{
    FIDDLE(...)
};

FIDDLE()
class FeedbackType : public BuiltinType
{
    FIDDLE(...)
    enum class Kind : uint8_t
    {
        MinMip,        /// SAMPLER_FEEDBACK_MIN_MIP
        MipRegionUsed, /// SAMPLER_FEEDBACK_MIP_REGION_USED
    };

    Kind getKind() const;
};

FIDDLE(abstract)
class TextureShapeType : public BuiltinType
{
    FIDDLE(...)
};

FIDDLE()
class TextureShape1DType : public TextureShapeType
{
    FIDDLE(...)
};
FIDDLE()
class TextureShape2DType : public TextureShapeType
{
    FIDDLE(...)
};
FIDDLE()
class TextureShape3DType : public TextureShapeType
{
    FIDDLE(...)
};
FIDDLE()
class TextureShapeCubeType : public TextureShapeType
{
    FIDDLE(...)
};
FIDDLE()
class TextureShapeBufferType : public TextureShapeType
{
    FIDDLE(...)
};

// Resources that contain "elements" that can be fetched
FIDDLE(abstract)
class ResourceType : public BuiltinType
{
    FIDDLE(...)
    bool isMultisample();
    bool isArray();
    bool isShadow();
    bool isFeedback();
    bool isCombined();
    SlangResourceShape getBaseShape();
    SlangResourceShape getShape();
    SlangResourceAccess getAccess();
    Type* getElementType();
    void _toTextOverride(StringBuilder& out);
};

FIDDLE(abstract)
class TextureTypeBase : public ResourceType
{
    FIDDLE(...)
    Val* getSampleCount();
    Val* getFormat();
};

FIDDLE()
class TextureType : public TextureTypeBase
{
    FIDDLE(...)
};

// This is a base type for `image*` types, as they exist in GLSL
FIDDLE()
class GLSLImageType : public TextureTypeBase
{
    FIDDLE(...)
};

FIDDLE()
class SubpassInputType : public BuiltinType
{
    FIDDLE(...)
    bool isMultisample();
    Type* getElementType();
};

FIDDLE()
class SamplerStateType : public BuiltinType
{
    FIDDLE(...)
    // Returns flavor of sampler state of this type.
    SamplerStateFlavor getFlavor() const;
};

// Other cases of generic types known to the compiler
FIDDLE()
class BuiltinGenericType : public BuiltinType
{
    FIDDLE(...)
    Type* getElementType() const;
};

// Types that behave like pointers, in that they can be
// dereferenced (implicitly) to access members defined
// in the element type.
FIDDLE(abstract)
class PointerLikeType : public BuiltinGenericType
{
    FIDDLE(...)
};

FIDDLE()
class DynamicResourceType : public BuiltinType
{
    FIDDLE(...)
};

// HLSL buffer-type resources

FIDDLE(abstract)
class HLSLStructuredBufferTypeBase : public BuiltinGenericType
{
    FIDDLE(...)
};

FIDDLE()
class HLSLStructuredBufferType : public HLSLStructuredBufferTypeBase
{
    FIDDLE(...)
};

FIDDLE()
class HLSLRWStructuredBufferType : public HLSLStructuredBufferTypeBase
{
    FIDDLE(...)
};

FIDDLE()
class HLSLRasterizerOrderedStructuredBufferType : public HLSLStructuredBufferTypeBase
{
    FIDDLE(...)
};


FIDDLE()
class UntypedBufferResourceType : public BuiltinType
{
    FIDDLE(...)
};

FIDDLE()
class HLSLByteAddressBufferType : public UntypedBufferResourceType
{
    FIDDLE(...)
};

FIDDLE()
class HLSLRWByteAddressBufferType : public UntypedBufferResourceType
{
    FIDDLE(...)
};

FIDDLE()
class HLSLRasterizerOrderedByteAddressBufferType : public UntypedBufferResourceType
{
    FIDDLE(...)
};

FIDDLE()
class RaytracingAccelerationStructureType : public UntypedBufferResourceType
{
    FIDDLE(...)
};


FIDDLE()
class HLSLAppendStructuredBufferType : public HLSLStructuredBufferTypeBase
{
    FIDDLE(...)
};

FIDDLE()
class HLSLConsumeStructuredBufferType : public HLSLStructuredBufferTypeBase
{
    FIDDLE(...)
};

FIDDLE()
class GLSLAtomicUintType : public BuiltinType
{
    FIDDLE(...)
};

FIDDLE()
class HLSLPatchType : public BuiltinType
{
    FIDDLE(...)
    Type* getElementType();
    IntVal* getElementCount();
};

FIDDLE()
class HLSLInputPatchType : public HLSLPatchType
{
    FIDDLE(...)
};

FIDDLE()
class HLSLOutputPatchType : public HLSLPatchType
{
    FIDDLE(...)
};


// HLSL geometry shader output stream types

FIDDLE()
class HLSLStreamOutputType : public BuiltinGenericType
{
    FIDDLE(...)
};

FIDDLE()
class HLSLPointStreamType : public HLSLStreamOutputType
{
    FIDDLE(...)
};

FIDDLE()
class HLSLLineStreamType : public HLSLStreamOutputType
{
    FIDDLE(...)
};

FIDDLE()
class HLSLTriangleStreamType : public HLSLStreamOutputType
{
    FIDDLE(...)
};

// mesh shader output types

FIDDLE()
class MeshOutputType : public BuiltinGenericType
{
    FIDDLE(...)
    Type* getElementType();

    IntVal* getMaxElementCount();
};

FIDDLE()
class VerticesType : public MeshOutputType
{
    FIDDLE(...)
};

FIDDLE()
class IndicesType : public MeshOutputType
{
    FIDDLE(...)
};

FIDDLE()
class PrimitivesType : public MeshOutputType
{
    FIDDLE(...)
};


//
FIDDLE()
class GLSLInputAttachmentType : public BuiltinType
{
    FIDDLE(...)
};


FIDDLE()
class DescriptorHandleType : public PointerLikeType
{
    FIDDLE(...)
};

// Base class for types used when desugaring parameter block
// declarations, includeing HLSL `cbuffer` or GLSL `uniform` blocks.
FIDDLE(abstract)
class ParameterGroupType : public PointerLikeType
{
    FIDDLE(...)
};

FIDDLE()
class UniformParameterGroupType : public ParameterGroupType
{
    FIDDLE(...)
    Type* getLayoutType();
};

FIDDLE()
class VaryingParameterGroupType : public ParameterGroupType
{
    FIDDLE(...)
};


// type for HLSL `cbuffer` declarations, and `ConstantBuffer<T>`
// ALso used for GLSL `uniform` blocks.
FIDDLE()
class ConstantBufferType : public UniformParameterGroupType
{
    FIDDLE(...)
};


// type for HLSL `tbuffer` declarations, and `TextureBuffer<T>`
FIDDLE()
class TextureBufferType : public UniformParameterGroupType
{
    FIDDLE(...)
};


// type for GLSL `in` and `out` blocks
FIDDLE()
class GLSLInputParameterGroupType : public VaryingParameterGroupType
{
    FIDDLE(...)
};

FIDDLE()
class GLSLOutputParameterGroupType : public VaryingParameterGroupType
{
    FIDDLE(...)
};


// type for GLSL `buffer` blocks
FIDDLE()
class GLSLShaderStorageBufferType : public PointerLikeType
{
    FIDDLE(...)
};


// type for Slang `ParameterBlock<T>` type
FIDDLE()
class ParameterBlockType : public UniformParameterGroupType
{
    FIDDLE(...)
};

FIDDLE()
class ArrayExpressionType : public DeclRefType
{
    FIDDLE(...)
    bool isUnsized();
    void _toTextOverride(StringBuilder& out);
    Type* getElementType();
    IntVal* getElementCount();
};

FIDDLE()
class AtomicType : public DeclRefType
{
    FIDDLE(...)
    Type* getElementType();
};

FIDDLE()
class CoopVectorExpressionType : public ArithmeticExpressionType
{
    FIDDLE(...)
    void _toTextOverride(StringBuilder& out);
    BasicExpressionType* _getScalarTypeOverride();

    Type* getElementType();
    IntVal* getElementCount();
};

// The "type" of an expression that resolves to a type.
// For example, in the expression `float(2)` the sub-expression,
// `float` would have the type `TypeType(float)`.
FIDDLE()
class TypeType : public Type
{
    FIDDLE(...)
    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();

    Type* getType() { return as<Type>(getOperand(0)); }

    TypeType(Type* type) { setOperands(type); }
};

// A differential pair type, e.g., `__DifferentialPair<T>`
FIDDLE()
class DifferentialPairType : public ArithmeticExpressionType
{
    FIDDLE(...)
    Type* getPrimalType();
};

FIDDLE()
class DifferentialPtrPairType : public ArithmeticExpressionType
{
    FIDDLE(...)
    Type* getPrimalRefType();
};

FIDDLE()
class DifferentiableType : public BuiltinType
{
    FIDDLE(...)
};

FIDDLE()
class DifferentiablePtrType : public BuiltinType
{
    FIDDLE(...)
};

FIDDLE()
class DefaultInitializableType : public BuiltinType
{
    FIDDLE(...)
};

// A vector type, e.g., `vector<T,N>`
FIDDLE()
class VectorExpressionType : public ArithmeticExpressionType
{
    FIDDLE(...)
    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    BasicExpressionType* _getScalarTypeOverride();

    Type* getElementType();
    IntVal* getElementCount();
};

// A matrix type, e.g., `matrix<T,R,C,L>`
FIDDLE()
class MatrixExpressionType : public ArithmeticExpressionType
{
    FIDDLE(...)
    Type* getElementType();
    IntVal* getRowCount();
    IntVal* getColumnCount();
    IntVal* getLayout();

    Type* getRowType();

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    BasicExpressionType* _getScalarTypeOverride();

private:
    SLANG_UNREFLECTED Type* rowType = nullptr;
};

FIDDLE()
class TensorViewType : public BuiltinType
{
    FIDDLE(...)
    Type* getElementType();
};

// Base class for built in string types
FIDDLE(abstract)
class StringTypeBase : public BuiltinType
{
    FIDDLE(...)
};

// The regular built-in `String` type
FIDDLE()
class StringType : public StringTypeBase
{
    FIDDLE(...)
};

// The string type native to the target
FIDDLE()
class NativeStringType : public StringTypeBase
{
    FIDDLE(...)
};

// The built-in `__Dynamic` type
FIDDLE()
class DynamicType : public BuiltinType
{
    FIDDLE(...)
};

// Type built-in `__EnumType` type
FIDDLE()
class EnumTypeType : public BuiltinType
{
    FIDDLE(...)
    // TODO: provide accessors for the declaration, the "tag" type, etc.
};

// Base class for types that map down to
// simple pointers as part of code generation.
FIDDLE()
class PtrTypeBase : public BuiltinType
{
    FIDDLE(...)
    // Get the type of the pointed-to value.
    Type* getValueType();

    Val* getAddressSpace();
};

FIDDLE()
class NoneType : public BuiltinType
{
    FIDDLE(...)
};

FIDDLE()
class NullPtrType : public BuiltinType
{
    FIDDLE(...)
};

// A true (user-visible) pointer type, e.g., `T*`
FIDDLE()
class PtrType : public PtrTypeBase
{
    FIDDLE(...)
    void _toTextOverride(StringBuilder& out);
};

/// A pointer-like type used to represent a parameter "direction"
FIDDLE()
class ParamDirectionType : public PtrTypeBase
{
    FIDDLE(...)
};

// A type that represents the behind-the-scenes
// logical pointer that is passed for an `out`
// or `in out` parameter
FIDDLE(abstract)
class OutTypeBase : public ParamDirectionType
{
    FIDDLE(...)
};

// The type for an `out` parameter, e.g., `out T`
FIDDLE()
class OutType : public OutTypeBase
{
    FIDDLE(...)
};

// The type for an `in out` parameter, e.g., `in out T`
FIDDLE()
class InOutType : public OutTypeBase
{
    FIDDLE(...)
};

FIDDLE(abstract)
class RefTypeBase : public ParamDirectionType
{
    FIDDLE(...)
};

// The type for an `ref` parameter, e.g., `ref T`
FIDDLE()
class RefType : public RefTypeBase
{
    FIDDLE(...)
    void _toTextOverride(StringBuilder& out);
};

// The type for an `constref` parameter, e.g., `constref T`
FIDDLE()
class ConstRefType : public RefTypeBase
{
    FIDDLE(...)
};

FIDDLE()
class OptionalType : public BuiltinType
{
    FIDDLE(...)
    Type* getValueType();
};

// A raw-pointer reference to an managed value.
FIDDLE()
class NativeRefType : public BuiltinType
{
    FIDDLE(...)
    Type* getValueType();
};

// A type alias of some kind (e.g., via `typedef`)
FIDDLE()
class NamedExpressionType : public Type
{
    FIDDLE(...)
    DeclRef<TypeDefDecl> getDeclRef() { return as<DeclRefBase>(getOperand(0)); }

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();

    NamedExpressionType(DeclRef<TypeDefDecl> inDeclRef) { setOperands(inDeclRef); }
};

// A function type is defined by its parameter types
// and its result type.
FIDDLE()
class FuncType : public Type
{
    FIDDLE(...)
    // Construct a unary function
    FuncType(Type* paramType, Type* resultType, Type* errorType)
    {
        setOperands(paramType, resultType, errorType);
    }

    FuncType(ArrayView<Type*> parameters, Type* result, Type* error)
    {
        for (auto paramType : parameters)
            m_operands.add(ValNodeOperand(paramType));
        m_operands.add(ValNodeOperand(result));
        m_operands.add(ValNodeOperand(error));
    }

    OperandView<Type> getParamTypes() { return OperandView<Type>(this, 0, getOperandCount() - 2); }

    Index getParamCount() { return m_operands.getCount() - 2; }
    Type* getParamType(Index index) { return as<Type>(getOperand(index)); }
    Type* getResultType() { return as<Type>(getOperand(m_operands.getCount() - 2)); }
    Type* getErrorType() { return as<Type>(getOperand(m_operands.getCount() - 1)); }

    ParameterDirection getParamDirection(Index index);

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

// A tuple is a product of its member types
FIDDLE()
class TupleType : public DeclRefType
{
    FIDDLE(...)
    Index getMemberCount() const;
    Type* getMember(Index i) const;
    Type* getTypePack() const;
};

FIDDLE()
class EachType : public Type
{
    FIDDLE(...)
    Type* getElementType() const { return as<Type>(getOperand(0)); }
    DeclRefType* getElementDeclRefType() const { return as<DeclRefType>(getOperand(0)); }

    EachType(Type* elementType) { m_operands.add(ValNodeOperand(elementType)); }
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

FIDDLE()
class ExpandType : public Type
{
    FIDDLE(...)
    Type* getPatternType() const { return as<Type>(getOperand(0)); }
    Index getCapturedTypePackCount() { return getOperandCount() - 1; }
    Type* getCapturedTypePack(Index i) { return as<Type>(getOperand(i + 1)); }
    ExpandType(Type* patternType, ArrayView<Type*> capturedPacks)
    {
        m_operands.add(ValNodeOperand(patternType));
        for (auto t : capturedPacks)
            m_operands.add(ValNodeOperand(t));
    }
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

// A concrete pack of types.
FIDDLE()
class ConcreteTypePack : public Type
{
    FIDDLE(...)
    ConcreteTypePack(ArrayView<Type*> types)
    {
        for (auto t : types)
            m_operands.add(ValNodeOperand(t));
    }
    Index getTypeCount() { return getOperandCount(); }
    Type* getElementType(Index i) { return as<Type>(getOperand(i)); }
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

// The "type" of an expression that names a generic declaration.
FIDDLE()
class GenericDeclRefType : public Type
{
    FIDDLE(...)
    DeclRef<GenericDecl> getDeclRef() const { return as<DeclRefBase>(getOperand(0)); }

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();

    GenericDeclRefType(DeclRef<GenericDecl> declRef) { setOperands(declRef); }
};

// The "type" of a reference to a module or namespace
FIDDLE()
class NamespaceType : public Type
{
    FIDDLE(...)
    DeclRef<NamespaceDeclBase> getDeclRef() const { return as<DeclRefBase>(getOperand(0)); }

    NamespaceType(DeclRef<NamespaceDeclBase> inDeclRef) { setOperands(inDeclRef); }

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();
};

// The concrete type for a value wrapped in an existential, accessible
// when the existential is "opened" in some context.
FIDDLE()
class ExtractExistentialType : public Type
{
    FIDDLE(...)
    DeclRef<VarDeclBase> getDeclRef() const { return as<DeclRefBase>(getOperand(0)); }

    // A reference to the original interface this type is known
    // to be a subtype of.
    //
    Type* getOriginalInterfaceType() { return as<Type>(getOperand(1)); }
    DeclRef<InterfaceDecl> getOriginalInterfaceDeclRef() { return as<DeclRefBase>(getOperand(2)); }

    ExtractExistentialType(
        DeclRef<VarDeclBase> inDeclRef,
        Type* inOriginalInterfaceType,
        DeclRef<InterfaceDecl> inOriginalInterfaceDeclRef)
    {
        setOperands(inDeclRef, inOriginalInterfaceType, inOriginalInterfaceDeclRef);
    }

    // Following fields will not be reflected (and thus won't be serialized, etc.)
    SLANG_UNREFLECTED

    // A cached decl-ref to the original interface's ThisType Decl, with
    // a witness that refers to the type extracted here.
    //
    // This field is optional and can be filled in on-demand. It does *not*
    // represent part of the logical value of this `Type`, and should not
    // be serialized, included in hashes, etc.
    //
    DeclRef<ThisTypeDecl> cachedThisTypeDeclRef;

    // A cached pointer to a witness that shows how this type is a subtype
    // of `originalInterfaceType`.
    //
    SubtypeWitness* cachedSubtypeWitness = nullptr;

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);

    /// Get a witness that shows how this type is a subtype of `originalInterfaceType`.
    ///
    /// This operation may create the witness on demand and cache it.
    ///
    SubtypeWitness* getSubtypeWitness();

    /// Get a decl-ref to the interface's ThisType decl, which represents a substitutable type
    /// from which lookup can be performed.
    ///
    /// This operation may create the decl-ref on demand and cache it.
    ///
    DeclRef<ThisTypeDecl> getThisTypeDeclRef();
};

FIDDLE()
class ExistentialSpecializedType : public Type
{
    FIDDLE(...)
    Type* getBaseType() { return as<Type>(getOperand(0)); }
    ExpandedSpecializationArg getArg(Index i)
    {
        ExpandedSpecializationArg arg;
        arg.val = getOperand(i * 2 + 1);
        arg.witness = getOperand(i * 2 + 2);
        return arg;
    }
    Index getArgCount() { return (getOperandCount() - 1) / 2; }

    ExistentialSpecializedType(Type* inBaseType, ExpandedSpecializationArgs const& inArgs)
    {
        m_operands.add(ValNodeOperand(inBaseType));
        for (auto arg : inArgs)
        {
            m_operands.add(ValNodeOperand(arg.val));
            m_operands.add(ValNodeOperand(arg.witness));
        }
    }

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

/// The type of `this` within a polymorphic declaration
FIDDLE()
class ThisType : public DeclRefType
{
    FIDDLE(...)
    ThisType(DeclRefBase* declRef)
        : DeclRefType(declRef)
    {
    }

    DeclRef<InterfaceDecl> getInterfaceDeclRef();
};

/// The type of `A & B` where `A` and `B` are types
///
/// A value `v` is of type `A & B` if it is both of type `A` and of type `B`.
FIDDLE()
class AndType : public Type
{
    FIDDLE(...)
    Type* getLeft() { return as<Type>(getOperand(0)); }
    Type* getRight() { return as<Type>(getOperand(1)); }

    AndType(Type* leftType, Type* rightType) { setOperands(leftType, rightType); }

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

FIDDLE()
class ModifiedType : public Type
{
    FIDDLE(...)
    Type* getBase() { return as<Type>(getOperand(0)); }

    Index getModifierCount() { return getOperandCount() - 1; }
    Val* getModifier(Index index) { return getOperand(index + 1); }

    ModifiedType(Type* inBase, ArrayView<Val*> inModifiers)
    {
        m_operands.add(ValNodeOperand(inBase));
        for (auto modifier : inModifiers)
            m_operands.add(ValNodeOperand(modifier));
    }

    template<typename T>
    T* findModifier()
    {
        for (Index i = 1; i < getOperandCount(); i++)
            if (auto rs = as<T>(getOperand(i)))
                return rs;
        return nullptr;
    }

    // Overrides should be public so base classes can access
    void _toTextOverride(StringBuilder& out);
    Type* _createCanonicalTypeOverride();
    Val* _substituteImplOverride(ASTBuilder* astBuilder, SubstitutionSet subst, int* ioDiff);
};

Type* removeParamDirType(Type* type);
bool isNonCopyableType(Type* type);

} // namespace Slang
