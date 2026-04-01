// slang-reflection-api.cpp

#include "../core/slang-basic.h"
#include "slang-check-impl.h"
#include "slang-check.h"
#include "slang-compiler.h"
#include "slang-syntax.h"
#include "slang-type-layout.h"
#include "slang.h"

#include <assert.h>

// Don't signal errors for stuff we don't implement here,
// and instead just try to return things defensively
//
// Slang developers can switch this when debugging.
#define SLANG_REFLECTION_UNEXPECTED() \
    do                                \
    {                                 \
    } while (0)

namespace Slang
{

// Conversion routines to help with strongly-typed reflection API

static inline Attribute* convert(SlangReflectionUserAttribute* attrib)
{
    return (Attribute*)attrib;
}
static inline SlangReflectionUserAttribute* convert(Attribute* attrib)
{
    return (SlangReflectionUserAttribute*)attrib;
}

static inline Type* convert(SlangReflectionType* type)
{
    return (Type*)type;
}

static inline SlangReflectionType* convert(Type* type)
{
    return (SlangReflectionType*)type;
}

static inline TypeLayout* convert(SlangReflectionTypeLayout* type)
{
    return (TypeLayout*)type;
}

static inline SlangReflectionTypeLayout* convert(TypeLayout* type)
{
    return (SlangReflectionTypeLayout*)type;
}

static inline SpecializationParamLayout* convert(SlangReflectionTypeParameter* typeParam)
{
    return (SpecializationParamLayout*)typeParam;
}

static inline DeclRef<Decl> convert(SlangReflectionVariable* var)
{
    return DeclRef<Decl>((DeclRefBase*)var);
}

static inline SlangReflectionVariable* convert(DeclRef<Decl> var)
{
    return (SlangReflectionVariable*)var.declRefBase;
}

static inline DeclRef<FunctionDeclBase> convertToFunc(SlangReflectionFunction* func)
{
    NodeBase* nodeBase = (NodeBase*)func;
    if (DeclRefBase* declRefBase = as<DeclRefBase>(nodeBase))
    {
        return DeclRef<FunctionDeclBase>(declRefBase);
    }

    return DeclRef<FunctionDeclBase>();
}

static inline OverloadedExpr* convertToOverloadedFunc(SlangReflectionFunction* func)
{
    NodeBase* nodeBase = (NodeBase*)func;
    return as<OverloadedExpr>(nodeBase);
}

static inline SlangReflectionFunction* convert(DeclRef<FunctionDeclBase> func)
{
    return (SlangReflectionFunction*)func.declRefBase;
}

static inline SlangReflectionFunction* convert(OverloadedExpr* overloadedFunc)
{
    return (SlangReflectionFunction*)overloadedFunc;
}

static inline DeclRef<Decl> convertGenericToDeclRef(SlangReflectionGeneric* func)
{
    DeclRefBase* declBase = (DeclRefBase*)func;
    return DeclRef<Decl>(declBase);
}

static inline SlangReflectionGeneric* convertDeclToGeneric(DeclRef<Decl> func)
{
    return (SlangReflectionGeneric*)func.declRefBase;
}

static inline VarLayout* convert(SlangReflectionVariableLayout* var)
{
    return (VarLayout*)var;
}

static inline SlangReflectionVariableLayout* convert(VarLayout* var)
{
    return (SlangReflectionVariableLayout*)var;
}

static inline EntryPointLayout* convert(SlangReflectionEntryPoint* entryPoint)
{
    return (EntryPointLayout*)entryPoint;
}

static inline SlangReflectionEntryPoint* convert(EntryPointLayout* entryPoint)
{
    return (SlangReflectionEntryPoint*)entryPoint;
}

static inline ProgramLayout* convert(SlangReflection* program)
{
    return (ProgramLayout*)program;
}

[[maybe_unused]] static inline SlangReflection* convert(ProgramLayout* program)
{
    return (SlangReflection*)program;
}

// user attribute

static unsigned int getUserAttributeCount(Decl* decl)
{
    unsigned int count = 0;
    for (auto x : decl->getModifiersOfType<UserDefinedAttribute>())
    {
        SLANG_UNUSED(x);
        count++;
    }
    return count;
}

static SlangReflectionUserAttribute* findUserAttributeByName(
    Session* session,
    Decl* decl,
    const char* name)
{
    auto nameObj = session->tryGetNameObj(name);
    if (!nameObj)
        return nullptr;
    for (auto x : decl->getModifiersOfType<Attribute>())
    {
        if (x->keywordName == nameObj)
            return (SlangReflectionUserAttribute*)(x);
    }
    return nullptr;
}

static SlangReflectionUserAttribute* getUserAttributeByIndex(Decl* decl, unsigned int index)
{
    unsigned int id = 0;
    for (auto x : decl->getModifiersOfType<UserDefinedAttribute>())
    {
        if (id == index)
            return convert(x);
        id++;
    }
    return nullptr;
}


// Attempt "do what I mean" remapping from the parameter category the user asked about,
// over to a parameter category that they might have meant.
static SlangParameterCategory maybeRemapParameterCategory(
    TypeLayout* typeLayout,
    SlangParameterCategory category)
{
    // Do we have an entry for the category they asked about? Then use that.
    if (typeLayout->FindResourceInfo(LayoutResourceKind(category)))
        return category;

    // Do we have an entry for the `DescriptorTableSlot` category?
    if (typeLayout->FindResourceInfo(LayoutResourceKind::DescriptorTableSlot))
    {
        // Is the category they were asking about one that makes sense for the type
        // of this variable?
        Type* type = typeLayout->getType();
        while (auto arrayType = as<ArrayExpressionType>(type))
            type = arrayType->getElementType();
        switch (spReflectionType_GetKind(convert(type)))
        {
        case SLANG_TYPE_KIND_CONSTANT_BUFFER:
            if (category == SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER)
                return SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT;
            break;

        case SLANG_TYPE_KIND_RESOURCE:
            if (category == SLANG_PARAMETER_CATEGORY_SHADER_RESOURCE)
                return SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT;
            break;

        case SLANG_TYPE_KIND_SAMPLER_STATE:
            if (category == SLANG_PARAMETER_CATEGORY_SAMPLER_STATE)
                return SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT;
            break;

        case SLANG_TYPE_KIND_SHADER_STORAGE_BUFFER:
            if (category == SLANG_PARAMETER_CATEGORY_UNIFORM)
                return SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT;
            break;

            // TODO: implement more helpers here

        default:
            break;
        }
    }

    return category;
}

// Helpers for getting parameter count

static unsigned getParameterCount(RefPtr<TypeLayout> typeLayout)
{
    if (auto parameterGroupLayout = as<ParameterGroupTypeLayout>(typeLayout))
    {
        typeLayout = parameterGroupLayout->offsetElementTypeLayout;
    }

    if (auto structLayout = as<StructTypeLayout>(typeLayout))
    {
        return (unsigned)structLayout->fields.getCount();
    }

    return 0;
}

static VarLayout* getParameterByIndex(RefPtr<TypeLayout> typeLayout, unsigned index)
{
    if (auto parameterGroupLayout = as<ParameterGroupTypeLayout>(typeLayout))
    {
        typeLayout = parameterGroupLayout->offsetElementTypeLayout;
    }

    if (auto structLayout = as<StructTypeLayout>(typeLayout))
    {
        return structLayout->fields[index];
    }

    return 0;
}

static SlangParameterCategory getParameterCategory(LayoutResourceKind kind)
{
    return SlangParameterCategory(kind);
}

static SlangParameterCategory getParameterCategory(TypeLayout* typeLayout)
{
    auto resourceInfoCount = typeLayout->resourceInfos.getCount();
    if (resourceInfoCount == 1)
    {
        return getParameterCategory(typeLayout->resourceInfos[0].kind);
    }
    else if (resourceInfoCount == 0)
    {
        // TODO: can this ever happen?
        return SLANG_PARAMETER_CATEGORY_NONE;
    }
    return SLANG_PARAMETER_CATEGORY_MIXED;
}

static bool hasDefaultConstantBuffer(ScopeLayout* layout)
{
    auto typeLayout = layout->parametersLayout->getTypeLayout();
    return as<ParameterGroupTypeLayout>(typeLayout) != nullptr;
}


} // namespace Slang

using namespace Slang;

// Implementation to back public-facing reflection API

SLANG_API char const* spReflectionUserAttribute_GetName(SlangReflectionUserAttribute* attrib)
{
    auto userAttr = convert(attrib);
    if (!userAttr)
        return nullptr;
    return userAttr->getKeywordName()->text.getBuffer();
}
SLANG_API unsigned int spReflectionUserAttribute_GetArgumentCount(
    SlangReflectionUserAttribute* attrib)
{
    auto userAttr = convert(attrib);
    if (!userAttr)
        return 0;
    return (unsigned int)userAttr->args.getCount();
}
SlangReflectionType* spReflectionUserAttribute_GetArgumentType(
    SlangReflectionUserAttribute* attrib,
    unsigned int index)
{
    auto userAttr = convert(attrib);
    if (!userAttr)
        return nullptr;
    return convert(userAttr->args[index]->type.type);
}
SLANG_API SlangResult spReflectionUserAttribute_GetArgumentValueInt(
    SlangReflectionUserAttribute* attrib,
    unsigned int index,
    int* rs)
{
    auto userAttr = convert(attrib);
    if (!userAttr)
        return SLANG_E_INVALID_ARG;
    if (index >= (unsigned int)userAttr->args.getCount())
        return SLANG_E_INVALID_ARG;

    if (userAttr->intArgVals.getCount() > (Index)index)
    {
        auto intVal = as<ConstantIntVal>(userAttr->intArgVals[index]);
        if (intVal)
        {
            *rs = (int)intVal->getValue();
            return 0;
        }
    }
    return SLANG_E_INVALID_ARG;
}
SLANG_API SlangResult spReflectionUserAttribute_GetArgumentValueFloat(
    SlangReflectionUserAttribute* attrib,
    unsigned int index,
    float* rs)
{
    auto userAttr = convert(attrib);
    if (!userAttr)
        return SLANG_E_INVALID_ARG;
    if (index >= (unsigned int)userAttr->args.getCount())
        return SLANG_E_INVALID_ARG;
    if (auto cexpr = as<FloatingPointLiteralExpr>(userAttr->args[index]))
    {
        *rs = (float)cexpr->value;
        return 0;
    }
    else if (auto implicitCastExpr = as<ImplicitCastExpr>(userAttr->args[index]))
    {
        auto base = implicitCastExpr->arguments[0];
        if (auto intLit = as<IntegerLiteralExpr>(base))
        {
            *rs = (float)intLit->value;
            return 0;
        }
    }
    return SLANG_E_INVALID_ARG;
}
SLANG_API const char* spReflectionUserAttribute_GetArgumentValueString(
    SlangReflectionUserAttribute* attrib,
    unsigned int index,
    size_t* bufLen)
{
    auto userAttr = convert(attrib);
    if (!userAttr)
        return nullptr;
    if (index >= (unsigned int)userAttr->args.getCount())
        return nullptr;
    if (auto cexpr = as<StringLiteralExpr>(userAttr->args[index]))
    {
        if (bufLen)
            *bufLen = cexpr->value.getLength();
        return cexpr->value.getBuffer();
    }
    return nullptr;
}

// type Reflection

SLANG_API SlangTypeKind spReflectionType_GetKind(SlangReflectionType* inType)
{
    auto type = convert(inType);
    if (!type)
        return SLANG_TYPE_KIND_NONE;

    // TODO(tfoley): Don't emit the same type more than once...

    if (const auto basicType = as<BasicExpressionType>(type))
    {
        return SLANG_TYPE_KIND_SCALAR;
    }
    else if (const auto vectorType = as<VectorExpressionType>(type))
    {
        return SLANG_TYPE_KIND_VECTOR;
    }
    else if (const auto matrixType = as<MatrixExpressionType>(type))
    {
        return SLANG_TYPE_KIND_MATRIX;
    }
    else if (const auto parameterBlockType = as<ParameterBlockType>(type))
    {
        return SLANG_TYPE_KIND_PARAMETER_BLOCK;
    }
    else if (const auto constantBufferType = as<ConstantBufferType>(type))
    {
        return SLANG_TYPE_KIND_CONSTANT_BUFFER;
    }
    else if (const auto streamOutputType = as<HLSLStreamOutputType>(type))
    {
        return SLANG_TYPE_KIND_OUTPUT_STREAM;
    }
    else if (as<MeshOutputType>(type))
    {
        return SLANG_TYPE_KIND_MESH_OUTPUT;
    }
    else if (as<TextureBufferType>(type))
    {
        return SLANG_TYPE_KIND_TEXTURE_BUFFER;
    }
    else if (as<GLSLShaderStorageBufferType>(type))
    {
        return SLANG_TYPE_KIND_SHADER_STORAGE_BUFFER;
    }
    else if (const auto samplerStateType = as<SamplerStateType>(type))
    {
        return SLANG_TYPE_KIND_SAMPLER_STATE;
    }
    else if (const auto textureType = as<TextureTypeBase>(type))
    {
        return SLANG_TYPE_KIND_RESOURCE;
    }
    else if (const auto feedbackType = as<FeedbackType>(type))
    {
        return SLANG_TYPE_KIND_FEEDBACK;
    }
    else if (const auto ptrType = as<PtrType>(type))
    {
        return SLANG_TYPE_KIND_POINTER;
    }
    else if (const auto dynamicResourceType = as<DynamicResourceType>(type))
    {
        return SLANG_TYPE_KIND_DYNAMIC_RESOURCE;
    }
    // TODO: need a better way to handle this stuff...
#define CASE(TYPE)                       \
    else if (as<TYPE>(type)) do          \
    {                                    \
        return SLANG_TYPE_KIND_RESOURCE; \
    }                                    \
    while (0)

    CASE(HLSLStructuredBufferType);
    CASE(HLSLRWStructuredBufferType);
    CASE(HLSLRasterizerOrderedStructuredBufferType);
    CASE(HLSLAppendStructuredBufferType);
    CASE(HLSLConsumeStructuredBufferType);
    CASE(HLSLByteAddressBufferType);
    CASE(HLSLRWByteAddressBufferType);
    CASE(HLSLRasterizerOrderedByteAddressBufferType);
    CASE(UntypedBufferResourceType);
    CASE(GLSLShaderStorageBufferType);
#undef CASE

    else if (const auto arrayType = as<ArrayExpressionType>(type))
    {
        return SLANG_TYPE_KIND_ARRAY;
    }
    else if (auto declRefType = as<DeclRefType>(type))
    {
        const auto& declRef = declRefType->getDeclRef();
        if (declRef.is<StructDecl>())
        {
            return SLANG_TYPE_KIND_STRUCT;
        }
        else if (declRef.is<GlobalGenericParamDecl>())
        {
            return SLANG_TYPE_KIND_GENERIC_TYPE_PARAMETER;
        }
        else if (declRef.is<InterfaceDecl>())
        {
            return SLANG_TYPE_KIND_INTERFACE;
        }
        else if (declRef.is<FuncDecl>())
        {
            // This is a reference to an entry point
            return SLANG_TYPE_KIND_STRUCT;
        }
    }
    else if (const auto specializedType = as<ExistentialSpecializedType>(type))
    {
        return SLANG_TYPE_KIND_SPECIALIZED;
    }
    else if (const auto errorType = as<ErrorType>(type))
    {
        // This means we saw a type we didn't understand in the user's code
        return SLANG_TYPE_KIND_NONE;
    }

    SLANG_REFLECTION_UNEXPECTED();
    return SLANG_TYPE_KIND_NONE;
}

SLANG_API unsigned int spReflectionType_GetFieldCount(SlangReflectionType* inType)
{
    auto type = convert(inType);
    if (!type)
        return 0;

    // TODO: maybe filter based on kind

    if (auto declRefType = as<DeclRefType>(type))
    {
        auto declRef = declRefType->getDeclRef();
        if (auto structDeclRef = declRef.as<StructDecl>())
        {
            return (unsigned int)getFields(
                       getModule(declRef.getDecl())->getLinkage()->getASTBuilder(),
                       structDeclRef,
                       MemberFilterStyle::Instance)
                .getCount();
        }
    }

    return 0;
}

SLANG_API SlangReflectionVariable* spReflectionType_GetFieldByIndex(
    SlangReflectionType* inType,
    unsigned index)
{
    auto type = convert(inType);
    if (!type)
        return nullptr;

    // TODO: maybe filter based on kind

    if (auto declRefType = as<DeclRefType>(type))
    {
        auto declRef = declRefType->getDeclRef();
        if (auto structDeclRef = declRef.as<StructDecl>())
        {
            auto fields = getFields(
                getModule(declRef.getDecl())->getLinkage()->getASTBuilder(),
                structDeclRef,
                MemberFilterStyle::Instance);
            auto fieldDeclRef = fields[index];
            return convert(fieldDeclRef);
        }
    }

    return nullptr;
}

SLANG_API size_t spReflectionType_GetElementCount(SlangReflectionType* inType)
{
    auto type = convert(inType);
    if (!type)
        return 0;

    if (auto arrayType = as<ArrayExpressionType>(type))
    {
        return !arrayType->isUnsized() ? (size_t)getIntVal(arrayType->getElementCount()) : 0;
    }
    else if (auto vectorType = as<VectorExpressionType>(type))
    {
        return (size_t)getIntVal(vectorType->getElementCount());
    }

    return 0;
}

SLANG_API SlangReflectionType* spReflectionType_GetElementType(SlangReflectionType* inType)
{
    auto type = convert(inType);
    if (!type)
        return nullptr;

    if (auto arrayType = as<ArrayExpressionType>(type))
    {
        return (SlangReflectionType*)arrayType->getElementType();
    }
    else if (auto parameterGroupType = as<ParameterGroupType>(type))
    {
        return convert(parameterGroupType->getElementType());
    }
    else if (auto structuredBufferType = as<HLSLStructuredBufferTypeBase>(type))
    {
        return convert(structuredBufferType->getElementType());
    }
    else if (auto vectorType = as<VectorExpressionType>(type))
    {
        return convert(vectorType->getElementType());
    }
    else if (auto matrixType = as<MatrixExpressionType>(type))
    {
        return convert(matrixType->getElementType());
    }

    return nullptr;
}

SLANG_API unsigned int spReflectionType_GetRowCount(SlangReflectionType* inType)
{
    auto type = convert(inType);
    if (!type)
        return 0;

    if (auto matrixType = as<MatrixExpressionType>(type))
    {
        return (unsigned int)getIntVal(matrixType->getRowCount());
    }
    else if (const auto vectorType = as<VectorExpressionType>(type))
    {
        return 1;
    }
    else if (const auto basicType = as<BasicExpressionType>(type))
    {
        return 1;
    }

    return 0;
}

SLANG_API unsigned int spReflectionType_GetColumnCount(SlangReflectionType* inType)
{
    auto type = convert(inType);
    if (!type)
        return 0;

    if (auto matrixType = as<MatrixExpressionType>(type))
    {
        return (unsigned int)getIntVal(matrixType->getColumnCount());
    }
    else if (auto vectorType = as<VectorExpressionType>(type))
    {
        return (unsigned int)getIntVal(vectorType->getElementCount());
    }
    else if (const auto basicType = as<BasicExpressionType>(type))
    {
        return 1;
    }

    return 0;
}

SLANG_API SlangScalarType spReflectionType_GetScalarType(SlangReflectionType* inType)
{
    auto type = convert(inType);
    if (!type)
        return SLANG_SCALAR_TYPE_NONE;

    if (auto matrixType = as<MatrixExpressionType>(type))
    {
        type = matrixType->getElementType();
    }
    else if (auto vectorType = as<VectorExpressionType>(type))
    {
        type = vectorType->getElementType();
    }

    if (auto basicType = as<BasicExpressionType>(type))
    {
        switch (basicType->getBaseType())
        {
#define CASE(BASE, TAG)  \
    case BaseType::BASE: \
        return SLANG_SCALAR_TYPE_##TAG

            CASE(Void, VOID);
            CASE(Bool, BOOL);
            CASE(Int8, INT8);
            CASE(Int16, INT16);
            CASE(Int, INT32);
            CASE(Int64, INT64);
            CASE(UInt8, UINT8);
            CASE(UInt16, UINT16);
            CASE(UInt, UINT32);
            CASE(UInt64, UINT64);
            CASE(Half, FLOAT16);
            CASE(Float, FLOAT32);
            CASE(Double, FLOAT64);

#undef CASE

        default:
            SLANG_REFLECTION_UNEXPECTED();
            return SLANG_SCALAR_TYPE_NONE;
            break;
        }
    }

    return SLANG_SCALAR_TYPE_NONE;
}

SLANG_API unsigned int spReflectionType_GetUserAttributeCount(SlangReflectionType* inType)
{
    auto type = convert(inType);
    if (!type)
        return 0;
    if (auto declRefType = as<DeclRefType>(type))
    {
        return getUserAttributeCount(declRefType->getDeclRef().getDecl());
    }
    return 0;
}
SLANG_API SlangReflectionUserAttribute* spReflectionType_GetUserAttribute(
    SlangReflectionType* inType,
    unsigned int index)
{
    auto type = convert(inType);
    if (!type)
        return 0;
    if (auto declRefType = as<DeclRefType>(type))
    {
        return getUserAttributeByIndex(declRefType->getDeclRef().getDecl(), index);
    }
    return 0;
}
SLANG_API SlangReflectionUserAttribute* spReflectionType_FindUserAttributeByName(
    SlangReflectionType* inType,
    char const* name)
{
    auto type = convert(inType);
    if (!type)
        return 0;
    if (auto declRefType = as<DeclRefType>(type))
    {
        ASTBuilder* astBuilder = declRefType->getASTBuilderForReflection();
        auto globalSession = astBuilder->getGlobalSession();

        return findUserAttributeByName(globalSession, declRefType->getDeclRef().getDecl(), name);
    }
    return 0;
}

SLANG_API SlangReflectionType* spReflectionType_applySpecializations(
    SlangReflectionType* inType,
    SlangReflectionGeneric* generic)
{
    auto type = convert(inType);
    auto genericDeclRef = convertGenericToDeclRef(generic);

    if (!type || !genericDeclRef)
        return nullptr;

    return convert(
        substituteType(SubstitutionSet(genericDeclRef), type->getASTBuilderForReflection(), type));
}

SLANG_API SlangResourceShape spReflectionType_GetResourceShape(SlangReflectionType* inType)
{
    auto type = convert(inType);
    if (!type)
        return SLANG_RESOURCE_NONE;

    while (auto arrayType = as<ArrayExpressionType>(type))
    {
        type = arrayType->getElementType();
    }

    if (auto textureType = as<TextureTypeBase>(type))
    {
        return textureType->getShape();
    }

    // TODO: need a better way to handle this stuff...
#define CASE(TYPE, SHAPE, ACCESS) \
    else if (as<TYPE>(type)) do   \
    {                             \
        return SHAPE;             \
    }                             \
    while (0)

    CASE(HLSLStructuredBufferType, SLANG_STRUCTURED_BUFFER, SLANG_RESOURCE_ACCESS_READ);
    CASE(HLSLRWStructuredBufferType, SLANG_STRUCTURED_BUFFER, SLANG_RESOURCE_ACCESS_READ_WRITE);
    CASE(
        HLSLRasterizerOrderedStructuredBufferType,
        SLANG_STRUCTURED_BUFFER,
        SLANG_RESOURCE_ACCESS_RASTER_ORDERED);
    CASE(HLSLAppendStructuredBufferType, SLANG_STRUCTURED_BUFFER, SLANG_RESOURCE_ACCESS_APPEND);
    CASE(HLSLConsumeStructuredBufferType, SLANG_STRUCTURED_BUFFER, SLANG_RESOURCE_ACCESS_CONSUME);
    CASE(HLSLByteAddressBufferType, SLANG_BYTE_ADDRESS_BUFFER, SLANG_RESOURCE_ACCESS_READ);
    CASE(HLSLRWByteAddressBufferType, SLANG_BYTE_ADDRESS_BUFFER, SLANG_RESOURCE_ACCESS_READ_WRITE);
    CASE(
        HLSLRasterizerOrderedByteAddressBufferType,
        SLANG_BYTE_ADDRESS_BUFFER,
        SLANG_RESOURCE_ACCESS_RASTER_ORDERED);
    CASE(
        RaytracingAccelerationStructureType,
        SLANG_ACCELERATION_STRUCTURE,
        SLANG_RESOURCE_ACCESS_READ);
    CASE(UntypedBufferResourceType, SLANG_BYTE_ADDRESS_BUFFER, SLANG_RESOURCE_ACCESS_READ);
    CASE(GLSLShaderStorageBufferType, SLANG_BYTE_ADDRESS_BUFFER, SLANG_RESOURCE_ACCESS_READ_WRITE);
#undef CASE

    return SLANG_RESOURCE_NONE;
}

SLANG_API SlangResourceAccess spReflectionType_GetResourceAccess(SlangReflectionType* inType)
{
    auto type = convert(inType);
    if (!type)
        return SLANG_RESOURCE_ACCESS_NONE;

    while (auto arrayType = as<ArrayExpressionType>(type))
    {
        type = arrayType->getElementType();
    }

    if (auto textureType = as<TextureTypeBase>(type))
    {
        return textureType->getAccess();
    }

    // TODO: need a better way to handle this stuff...
#define CASE(TYPE, SHAPE, ACCESS) \
    else if (as<TYPE>(type)) do   \
    {                             \
        return ACCESS;            \
    }                             \
    while (0)

    CASE(HLSLStructuredBufferType, SLANG_STRUCTURED_BUFFER, SLANG_RESOURCE_ACCESS_READ);
    CASE(HLSLRWStructuredBufferType, SLANG_STRUCTURED_BUFFER, SLANG_RESOURCE_ACCESS_READ_WRITE);
    CASE(
        HLSLRasterizerOrderedStructuredBufferType,
        SLANG_STRUCTURED_BUFFER,
        SLANG_RESOURCE_ACCESS_RASTER_ORDERED);
    CASE(HLSLAppendStructuredBufferType, SLANG_STRUCTURED_BUFFER, SLANG_RESOURCE_ACCESS_APPEND);
    CASE(HLSLConsumeStructuredBufferType, SLANG_STRUCTURED_BUFFER, SLANG_RESOURCE_ACCESS_CONSUME);
    CASE(HLSLByteAddressBufferType, SLANG_BYTE_ADDRESS_BUFFER, SLANG_RESOURCE_ACCESS_READ);
    CASE(HLSLRWByteAddressBufferType, SLANG_BYTE_ADDRESS_BUFFER, SLANG_RESOURCE_ACCESS_READ_WRITE);
    CASE(
        HLSLRasterizerOrderedByteAddressBufferType,
        SLANG_BYTE_ADDRESS_BUFFER,
        SLANG_RESOURCE_ACCESS_RASTER_ORDERED);
    CASE(UntypedBufferResourceType, SLANG_BYTE_ADDRESS_BUFFER, SLANG_RESOURCE_ACCESS_READ);
    CASE(GLSLShaderStorageBufferType, SLANG_BYTE_ADDRESS_BUFFER, SLANG_RESOURCE_ACCESS_READ_WRITE);
#undef CASE

    return SLANG_RESOURCE_ACCESS_NONE;
}

SLANG_API char const* spReflectionType_GetName(SlangReflectionType* inType)
{
    auto type = convert(inType);

    if (auto declRefType = as<DeclRefType>(type))
    {
        auto declRef = declRefType->getDeclRef();

        // Don't return a name for auto-generated anonymous types
        // that represent `cbuffer` members, etc.
        auto decl = declRef.getDecl();
        if (decl->hasModifier<ImplicitParameterGroupElementTypeModifier>())
            return nullptr;
        return getText(declRef.getName()).begin();
    }

    return nullptr;
}

SLANG_API SlangResult
spReflectionType_GetFullName(SlangReflectionType* inType, ISlangBlob** outNameBlob)
{
    auto type = convert(inType);

    if (!type)
        return SLANG_FAIL;

    StringBuilder sb;
    type->toText(sb);
    *outNameBlob = StringUtil::createStringBlob(sb.produceString()).detach();
    return SLANG_OK;
}

SlangReflectionFunction* tryConvertExprToFunctionReflection(ASTBuilder* astBuilder, Expr* expr)
{
    if (auto declRefExpr = as<DeclRefExpr>(expr))
    {
        auto declRef = declRefExpr->declRef;
        if (auto genericDeclRef = declRef.as<GenericDecl>())
        {
            auto innerDeclRef = createDefaultSubstitutionsIfNeeded(
                astBuilder,
                nullptr,
                genericDeclRef.getDecl()->inner);
            declRef = substituteDeclRef(SubstitutionSet(genericDeclRef), astBuilder, innerDeclRef);
        }

        if (auto funcDeclRef = declRef.as<FunctionDeclBase>())
            return convert(funcDeclRef);
    }
    else if (auto overloadedExpr = as<OverloadedExpr>(expr))
        return convert(overloadedExpr);

    return nullptr;
}

SLANG_API SlangReflectionFunction* spReflection_FindFunctionByName(
    SlangReflection* reflection,
    char const* name)
{
    auto programLayout = convert(reflection);
    auto program = programLayout->getProgram();

    // TODO: We should extend this API to support getting error messages
    // when type lookup fails.
    //
    Slang::DiagnosticSink sink(
        programLayout->getTargetReq()->getLinkage()->getSourceManager(),
        Lexer::sourceLocationLexer);

    auto astBuilder = program->getLinkage()->getASTBuilder();
    try
    {
        return tryConvertExprToFunctionReflection(
            astBuilder,
            program->findDeclFromString(name, &sink));
    }
    catch (...)
    {
    }
    return nullptr;
}

SLANG_API SlangReflectionFunction* spReflection_FindFunctionByNameInType(
    SlangReflection* reflection,
    SlangReflectionType* reflType,
    char const* name)
{
    auto programLayout = convert(reflection);
    auto program = programLayout->getProgram();

    auto type = convert(reflType);

    Slang::DiagnosticSink sink(
        programLayout->getTargetReq()->getLinkage()->getSourceManager(),
        Lexer::sourceLocationLexer);

    auto astBuilder = program->getLinkage()->getASTBuilder();

    try
    {
        auto result = program->findDeclFromStringInType(type, name, LookupMask::Function, &sink);
        return tryConvertExprToFunctionReflection(astBuilder, result);
    }
    catch (...)
    {
    }
    return nullptr;
}

SLANG_API SlangReflectionVariable* spReflection_FindVarByNameInType(
    SlangReflection* reflection,
    SlangReflectionType* reflType,
    char const* name)
{
    auto programLayout = convert(reflection);
    auto program = programLayout->getProgram();

    auto type = convert(reflType);

    Slang::DiagnosticSink sink(
        programLayout->getTargetReq()->getLinkage()->getSourceManager(),
        Lexer::sourceLocationLexer);

    try
    {
        auto result = program->findDeclFromStringInType(type, name, LookupMask::Value, &sink);
        if (auto declRefExpr = as<DeclRefExpr>(result))
        {
            if (auto varDeclRef = declRefExpr->declRef.as<VarDeclBase>())
                return convert(varDeclRef.as<Decl>());
        }
    }
    catch (...)
    {
    }
    return nullptr;
}

SLANG_API SlangReflectionType* spReflection_FindTypeByName(
    SlangReflection* reflection,
    char const* name)
{
    auto programLayout = convert(reflection);
    auto program = programLayout->getProgram();

    // TODO: We should extend this API to support getting error messages
    // when type lookup fails.
    //
    Slang::DiagnosticSink sink(
        programLayout->getTargetReq()->getLinkage()->getSourceManager(),
        Lexer::sourceLocationLexer);

    try
    {
        Type* result = program->getTypeFromString(name, &sink);

        ASTBuilder* astBuilder = program->getLinkage()->getASTBuilder();

        if (auto genericType = as<GenericDeclRefType>(result))
        {
            auto genericDeclRef = genericType->getDeclRef();
            auto innerDeclRef = substituteDeclRef(
                SubstitutionSet(genericDeclRef),
                astBuilder,
                genericDeclRef.getDecl()->inner);
            if (as<AggTypeDecl>(innerDeclRef.getDecl()) ||
                as<SimpleTypeDecl>(innerDeclRef.getDecl()))
                return convert(DeclRefType::create(astBuilder, innerDeclRef));
            return nullptr;
        }

        if (as<ErrorType>(result))
            return nullptr;
        return (SlangReflectionType*)result;
    }
    catch (...)
    {
        return nullptr;
    }
}


SLANG_API bool spReflection_isSubType(
    SlangReflection* reflection,
    SlangReflectionType* subType,
    SlangReflectionType* superType)
{
    auto programLayout = convert(reflection);
    auto program = programLayout->getProgram();

    // TODO: We should extend this API to support getting error messages
    // when type lookup fails.
    //
    Slang::DiagnosticSink sink(
        programLayout->getTargetReq()->getLinkage()->getSourceManager(),
        Lexer::sourceLocationLexer);

    try
    {
        auto sub = convert(subType);
        auto super = convert(superType);

        return program->isSubType(sub, super);
    }
    catch (...)
    {
        return false;
    }
}

DeclRef<Decl> getInnermostGenericParent(DeclRef<Decl> declRef)
{
    auto decl = declRef.getDecl();
    auto astBuilder = getModule(decl)->getLinkage()->getASTBuilder();
    auto parentDecl = decl;
    while (parentDecl)
    {
        if (parentDecl->parentDecl && as<GenericDecl>(parentDecl->parentDecl))
            return substituteDeclRef(
                SubstitutionSet(declRef),
                astBuilder,
                createDefaultSubstitutionsIfNeeded(astBuilder, nullptr, DeclRef(parentDecl)));
        parentDecl = parentDecl->parentDecl;
    }

    return DeclRef<Decl>();
}

SLANG_API SlangReflectionGeneric* spReflectionType_GetGenericContainer(SlangReflectionType* type)
{
    auto slangType = convert(type);
    if (auto declRefType = as<DeclRefType>(slangType))
    {
        return convertDeclToGeneric(getInnermostGenericParent(declRefType->getDeclRef()));
    }
    else if (auto genericDeclRefType = as<GenericDeclRefType>(slangType))
    {
        return convertDeclToGeneric(getInnermostGenericParent(genericDeclRefType->getDeclRef()));
    }

    return nullptr;
}

SLANG_API SlangReflectionTypeLayout* spReflection_GetTypeLayout(
    SlangReflection* reflection,
    SlangReflectionType* inType,
    SlangLayoutRules rules)
{
    auto context = convert(reflection);
    auto type = convert(inType);
    auto targetReq = context->getTargetReq();

    auto typeLayout = targetReq->getTypeLayout(type, (slang::LayoutRules)rules);
    return convert(typeLayout);
}

SLANG_API SlangReflectionType* spReflectionType_GetResourceResultType(SlangReflectionType* inType)
{
    auto type = convert(inType);
    if (!type)
        return nullptr;

    while (auto arrayType = as<ArrayExpressionType>(type))
    {
        type = arrayType->getElementType();
    }

    if (auto textureType = as<TextureTypeBase>(type))
    {
        return convert(textureType->getElementType());
    }

    // TODO: need a better way to handle this stuff...
#define CASE(TYPE, SHAPE, ACCESS)                         \
    else if (as<TYPE>(type)) do                           \
    {                                                     \
        return convert(as<TYPE>(type)->getElementType()); \
    }                                                     \
    while (0)

    // TODO: structured buffer needs to expose type layout!

    CASE(HLSLStructuredBufferType, SLANG_STRUCTURED_BUFFER, SLANG_RESOURCE_ACCESS_READ);
    CASE(HLSLRWStructuredBufferType, SLANG_STRUCTURED_BUFFER, SLANG_RESOURCE_ACCESS_READ_WRITE);
    CASE(
        HLSLRasterizerOrderedStructuredBufferType,
        SLANG_STRUCTURED_BUFFER,
        SLANG_RESOURCE_ACCESS_RASTER_ORDERED);
    CASE(HLSLAppendStructuredBufferType, SLANG_STRUCTURED_BUFFER, SLANG_RESOURCE_ACCESS_APPEND);
    CASE(HLSLConsumeStructuredBufferType, SLANG_STRUCTURED_BUFFER, SLANG_RESOURCE_ACCESS_CONSUME);
#undef CASE

    return nullptr;
}

// type Layout Reflection

SLANG_API SlangReflectionType* spReflectionTypeLayout_GetType(
    SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return nullptr;

    return (SlangReflectionType*)typeLayout->type;
}

SLANG_API SlangTypeKind spReflectionTypeLayout_getKind(SlangReflectionTypeLayout* inTypeLayout)
{
    if (!inTypeLayout)
        return SLANG_TYPE_KIND_NONE;

    if (auto type = spReflectionTypeLayout_GetType(inTypeLayout))
    {
        return spReflectionType_GetKind(type);
    }

    auto typeLayout = convert(inTypeLayout);
    if (as<StructTypeLayout>(typeLayout))
    {
        return SLANG_TYPE_KIND_STRUCT;
    }
    else if (as<ParameterGroupTypeLayout>(typeLayout))
    {
        return SLANG_TYPE_KIND_CONSTANT_BUFFER;
    }

    return SLANG_TYPE_KIND_NONE;
}

namespace
{
static size_t getReflectionSize(LayoutSize size)
{
    if (size.isFinite())
        return size.getFiniteValue();

    return SLANG_UNBOUNDED_SIZE;
}

static int32_t getAlignment(TypeLayout* typeLayout, SlangParameterCategory category)
{
    if (category == SLANG_PARAMETER_CATEGORY_UNIFORM)
    {
        return int32_t(typeLayout->uniformAlignment);
    }
    else
    {
        return 1;
    }
}

static size_t getStride(TypeLayout* typeLayout, SlangParameterCategory category)
{
    auto info = typeLayout->FindResourceInfo(LayoutResourceKind(category));
    if (!info)
        return 0;

    auto size = info->count;
    if (size.isInfinite())
        return SLANG_UNBOUNDED_SIZE;

    size_t finiteSize = size.getFiniteValue();
    size_t alignment = getAlignment(typeLayout, category);
    SLANG_ASSERT(alignment >= 1);

    auto stride = (finiteSize + (alignment - 1)) & ~(alignment - 1);
    return stride;
}
} // namespace

SLANG_API size_t spReflectionTypeLayout_GetSize(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangParameterCategory category)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto info = typeLayout->FindResourceInfo(LayoutResourceKind(category));
    if (!info)
        return 0;

    return getReflectionSize(info->count);
}

SLANG_API size_t spReflectionTypeLayout_GetStride(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangParameterCategory category)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    return getStride(typeLayout, category);
}

SLANG_API int32_t spReflectionTypeLayout_getAlignment(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangParameterCategory category)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    return getAlignment(typeLayout, category);
}

SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_GetFieldByIndex(
    SlangReflectionTypeLayout* inTypeLayout,
    unsigned index)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return nullptr;

    if (auto structTypeLayout = as<StructTypeLayout>(typeLayout))
    {
        return (SlangReflectionVariableLayout*)structTypeLayout->fields[index].Ptr();
    }

    return nullptr;
}

SLANG_API SlangInt spReflectionTypeLayout_findFieldIndexByName(
    SlangReflectionTypeLayout* inTypeLayout,
    const char* nameBegin,
    const char* nameEnd)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return -1;

    UnownedStringSlice name = nameEnd != nullptr ? UnownedStringSlice(nameBegin, nameEnd)
                                                 : UnownedTerminatedStringSlice(nameBegin);

    if (auto structTypeLayout = as<StructTypeLayout>(typeLayout))
    {
        Index fieldCount = structTypeLayout->fields.getCount();
        for (Index f = 0; f < fieldCount; ++f)
        {
            auto field = structTypeLayout->fields[f];
            if (getReflectionName(field->getVariable())->text.getUnownedSlice() == name)
                return f;
        }
    }

    return -1;
}

SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_GetExplicitCounter(
    SlangReflectionTypeLayout* inTypeLayout)
{
    const auto typeLayout = convert(inTypeLayout);
    if (const auto structuredBufferTypeLayout = as<StructuredBufferTypeLayout>(typeLayout))
        return (SlangReflectionVariableLayout*)structuredBufferTypeLayout->counterVarLayout.Ptr();
    return nullptr;
}

SLANG_API size_t spReflectionTypeLayout_GetElementStride(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangParameterCategory category)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    if (auto arrayTypeLayout = as<ArrayTypeLayout>(typeLayout))
    {
        switch (category)
        {
        // We store the stride explicitly for the uniform case
        case SLANG_PARAMETER_CATEGORY_UNIFORM:
            return arrayTypeLayout->uniformStride;

        // For most other cases (resource registers), the "stride"
        // of an array is simply the number of resources (if any)
        // consumed by its element type.
        default:
            {
                auto elementTypeLayout = arrayTypeLayout->elementTypeLayout;
                auto info = elementTypeLayout->FindResourceInfo(LayoutResourceKind(category));
                if (!info)
                    return 0;
                return getReflectionSize(info->count);
            }

        // An important special case, though, is Vulkan descriptor-table slots,
        // where an entire array will use a single `binding`, so that the
        // effective stride is zero:
        case SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT:
            return 0;
        }
    }
    else if (auto vectorTypeLayout = as<VectorTypeLayout>(typeLayout))
    {
        auto resInfo =
            vectorTypeLayout->elementTypeLayout->FindResourceInfo(LayoutResourceKind::Uniform);
        if (!resInfo)
            return 0;
        return resInfo->count.getFiniteValue();
    }

    return 0;
}

SLANG_API SlangReflectionTypeLayout* spReflectionTypeLayout_GetElementTypeLayout(
    SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return nullptr;

    if (auto arrayTypeLayout = as<ArrayTypeLayout>(typeLayout))
    {
        return (SlangReflectionTypeLayout*)arrayTypeLayout->elementTypeLayout.Ptr();
    }
    else if (auto constantBufferTypeLayout = as<ParameterGroupTypeLayout>(typeLayout))
    {
        return convert(constantBufferTypeLayout->offsetElementTypeLayout.Ptr());
    }
    else if (auto structuredBufferTypeLayout = as<StructuredBufferTypeLayout>(typeLayout))
    {
        return convert(structuredBufferTypeLayout->elementTypeLayout.Ptr());
    }
    else if (auto specializedTypeLayout = as<ExistentialSpecializedTypeLayout>(typeLayout))
    {
        return convert(specializedTypeLayout->baseTypeLayout.Ptr());
    }
    else if (auto vectorTypeLayout = as<VectorTypeLayout>(typeLayout))
    {
        return convert(vectorTypeLayout->elementTypeLayout);
    }
    else if (auto matrixTypeLayout = as<MatrixTypeLayout>(typeLayout))
    {
        return convert(matrixTypeLayout->elementTypeLayout);
    }
    else if (auto ptrTypeLayout = as<PointerTypeLayout>(typeLayout))
    {
        return convert(ptrTypeLayout->valueTypeLayout.Ptr());
    }
    return nullptr;
}

SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_GetElementVarLayout(
    SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return nullptr;

    if (auto parameterGroupTypeLayout = as<ParameterGroupTypeLayout>(typeLayout))
    {
        return convert(parameterGroupTypeLayout->elementVarLayout.Ptr());
    }

    return nullptr;
}

SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_getContainerVarLayout(
    SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return nullptr;

    if (auto parameterGroupTypeLayout = as<ParameterGroupTypeLayout>(typeLayout))
    {
        return convert(parameterGroupTypeLayout->containerVarLayout.Ptr());
    }

    return nullptr;
}

SLANG_API SlangParameterCategory
spReflectionTypeLayout_GetParameterCategory(SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return SLANG_PARAMETER_CATEGORY_NONE;

    return getParameterCategory(typeLayout);
}

SLANG_API uint32_t spReflectionTypeLayout_GetFieldCount(SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    if (auto structTypeLayout = as<StructTypeLayout>(typeLayout))
    {
        return (uint32_t)structTypeLayout->fields.getCount();
    }
    return 0;
}

SLANG_API unsigned spReflectionTypeLayout_GetCategoryCount(SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    return (unsigned)typeLayout->resourceInfos.getCount();
}

SLANG_API SlangParameterCategory
spReflectionTypeLayout_GetCategoryByIndex(SlangReflectionTypeLayout* inTypeLayout, unsigned index)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return SLANG_PARAMETER_CATEGORY_NONE;

    return SlangParameterCategory(typeLayout->resourceInfos[index].kind);
}

SLANG_API SlangMatrixLayoutMode
spReflectionTypeLayout_GetMatrixLayoutMode(SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return SLANG_MATRIX_LAYOUT_MODE_UNKNOWN;

    if (auto matrixLayout = as<MatrixTypeLayout>(typeLayout))
    {
        return SlangMatrixLayoutMode(matrixLayout->mode);
    }
    else
    {
        return SLANG_MATRIX_LAYOUT_MODE_UNKNOWN;
    }
}

SLANG_API int spReflectionTypeLayout_getGenericParamIndex(SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return -1;

    if (auto genericParamTypeLayout = as<GenericParamTypeLayout>(typeLayout))
    {
        return (int)genericParamTypeLayout->paramIndex;
    }
    else
    {
        return -1;
    }
}

SLANG_API SlangReflectionTypeLayout* spReflectionTypeLayout_getPendingDataTypeLayout(
    SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return nullptr;

    auto pendingDataTypeLayout = typeLayout->pendingDataTypeLayout.Ptr();
    return convert(pendingDataTypeLayout);
}

SLANG_API SlangReflectionVariableLayout* spReflectionVariableLayout_getPendingDataLayout(
    SlangReflectionVariableLayout* inVarLayout)
{
    auto varLayout = convert(inVarLayout);
    if (!varLayout)
        return nullptr;

    auto pendingDataLayout = varLayout->pendingVarLayout.Ptr();
    return convert(pendingDataLayout);
}

SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_getSpecializedTypePendingDataVarLayout(
    SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return nullptr;

    if (auto specializedTypeLayout = as<ExistentialSpecializedTypeLayout>(typeLayout))
    {
        auto pendingDataVarLayout = specializedTypeLayout->pendingDataVarLayout.Ptr();
        return convert(pendingDataVarLayout);
    }
    else
    {
        return nullptr;
    }
}

SLANG_API SlangInt spReflectionType_getSpecializedTypeArgCount(SlangReflectionType* inType)
{
    auto type = convert(inType);
    if (!type)
        return 0;

    auto specializedType = as<ExistentialSpecializedType>(type);
    if (!specializedType)
        return 0;

    return specializedType->getArgCount();
}

SLANG_API SlangReflectionType* spReflectionType_getSpecializedTypeArgType(
    SlangReflectionType* inType,
    SlangInt index)
{
    auto type = convert(inType);
    if (!type)
        return nullptr;

    auto specializedType = as<ExistentialSpecializedType>(type);
    if (!specializedType)
        return nullptr;

    if (index < 0)
        return nullptr;
    if (index >= specializedType->getArgCount())
        return nullptr;

    auto argType = as<Type>(specializedType->getArg(index).val);
    return convert(argType);
}

namespace Slang
{
/// A link in a chain of `VarLayout`s that can be used to compute offset information for a nested
/// field
struct BindingRangePathLink
{
    BindingRangePathLink() {}

    BindingRangePathLink(BindingRangePathLink* parent, VarLayout* var)
        : var(var), parent(parent)
    {
    }

    /// The inner-most variable that contributes to the offset along this path
    VarLayout* var = nullptr;

    /// The next outer link along the path
    BindingRangePathLink* parent = nullptr;
};

/// A path leading to some nested field, with both parimary and "pending" data offsets
struct BindingRangePath
{

    /// The chain of variables that defines the "primary" offset of a nested field
    BindingRangePathLink* primary = nullptr;

    /// The chain of variables that defines the offset for "pending" data of a nested field
    BindingRangePathLink* pending = nullptr;
};

/// A helper type to construct a `BindingRangePath` that extends an existing path
struct ExtendedBindingRangePath : BindingRangePath
{
    /// Construct a path that extends `parent` with offset information from `varLayout`
    ExtendedBindingRangePath(BindingRangePath const& parent, VarLayout* varLayout)
    {
        SLANG_ASSERT(varLayout);

        // We always add another link to the primary chain.
        //
        primaryLink = BindingRangePathLink(parent.primary, varLayout);
        primary = &primaryLink;

        // If the `varLayout` provided has any offset information
        // for pending data, then we also add a link to the pending
        // chain, but otherwise we re-use the pending chain from
        // the parent path.
        //
        if (auto pendingLayout = varLayout->pendingVarLayout)
        {
            pendingLink = BindingRangePathLink(parent.pending, pendingLayout);
            pending = &pendingLink;
        }
        else
        {
            pending = parent.pending;
        }
    }

    /// Storage for a link in the primary chain, if needed
    BindingRangePathLink primaryLink;

    /// Storage for a link in the pending chain, if needed
    BindingRangePathLink pendingLink;
};

/// Calculate the offset for resources of the given `kind` in the `path`.
Int _calcIndexOffset(BindingRangePathLink* path, LayoutResourceKind kind)
{
    Int result = 0;
    for (auto link = path; link; link = link->parent)
    {
        if (auto resInfo = link->var->FindResourceInfo(kind))
        {
            result += resInfo->index;
        }
    }
    return result;
}

/// Calculate the regsiter space / set for resources of the given `kind` in the `path`.
Int _calcSpaceOffset(BindingRangePathLink* path, LayoutResourceKind kind)
{
    Int result = 0;
    for (auto link = path; link; link = link->parent)
    {
        if (auto resInfo = link->var->FindResourceInfo(kind))
        {
            result += resInfo->space;
        }
    }
    return result;
}

SlangBindingType _calcResourceBindingType(Type* type)
{
    if (auto resourceType = as<ResourceType>(type))
    {
        if (resourceType->isCombined())
            return SlangBindingType(SLANG_BINDING_TYPE_COMBINED_TEXTURE_SAMPLER);

        auto shape = resourceType->getBaseShape();

        auto access = resourceType->getAccess();
        auto mutableFlag = access != SLANG_RESOURCE_ACCESS_READ ? SLANG_BINDING_TYPE_MUTABLE_FLAG
                                                                : SLANG_BINDING_TYPE_UNKNOWN;

        switch (SlangResourceShape(shape))
        {
        default:
            return SlangBindingType(SLANG_BINDING_TYPE_TEXTURE | mutableFlag);

        case SLANG_TEXTURE_BUFFER:
            return SlangBindingType(SLANG_BINDING_TYPE_TYPED_BUFFER | mutableFlag);
        }
    }
    else if (const auto structuredBufferType = as<HLSLStructuredBufferTypeBase>(type))
    {
        if (as<HLSLStructuredBufferType>(type))
        {
            return SLANG_BINDING_TYPE_RAW_BUFFER;
        }
        else
        {
            return SLANG_BINDING_TYPE_MUTABLE_RAW_BUFFER;
        }
    }
    else if (as<RaytracingAccelerationStructureType>(type))
    {
        return SLANG_BINDING_TYPE_RAY_TRACING_ACCELERATION_STRUCTURE;
    }
    else if (const auto untypedBufferType = as<UntypedBufferResourceType>(type))
    {
        if (as<HLSLByteAddressBufferType>(type))
        {
            return SLANG_BINDING_TYPE_RAW_BUFFER;
        }
        else
        {
            return SLANG_BINDING_TYPE_MUTABLE_RAW_BUFFER;
        }
    }
    else if (as<GLSLAtomicUintType>(type))
    {
        return SLANG_BINDING_TYPE_MUTABLE_RAW_BUFFER;
    }
    else if (as<GLSLShaderStorageBufferType>(type))
    {
        // TODO Immutable buffers
        return SLANG_BINDING_TYPE_MUTABLE_RAW_BUFFER;
    }
    else if (as<ConstantBufferType>(type))
    {
        return SLANG_BINDING_TYPE_CONSTANT_BUFFER;
    }
    else if (as<SamplerStateType>(type))
    {
        return SLANG_BINDING_TYPE_SAMPLER;
    }
    else if (as<ParameterBlockType>(type))
    {
        return SLANG_BINDING_TYPE_PARAMETER_BLOCK;
    }
    else
    {
        return SLANG_BINDING_TYPE_UNKNOWN;
    }
}

SlangBindingType _calcResourceBindingType(TypeLayout* typeLayout)
{
    if (auto type = typeLayout->getType())
    {
        return _calcResourceBindingType(type);
    }

    if (as<ParameterGroupTypeLayout>(typeLayout))
    {
        return SLANG_BINDING_TYPE_CONSTANT_BUFFER;
    }
    else
    {
        return SLANG_BINDING_TYPE_UNKNOWN;
    }
}

SlangBindingType _calcBindingType(LayoutResourceKind kind)
{
    switch (kind)
    {
    default:
        return SLANG_BINDING_TYPE_UNKNOWN;

        // Some cases of `LayoutResourceKind` can be mapped
        // directly to a `BindingType` because there is only
        // one case of types that have that resource kind.

#define CASE(FROM, TO)             \
    case LayoutResourceKind::FROM: \
        return SLANG_BINDING_TYPE_##TO

        CASE(ConstantBuffer, CONSTANT_BUFFER);
        CASE(SamplerState, SAMPLER);
        CASE(VaryingInput, VARYING_INPUT);
        CASE(VaryingOutput, VARYING_OUTPUT);
        CASE(ExistentialObjectParam, EXISTENTIAL_VALUE);
        CASE(PushConstantBuffer, PUSH_CONSTANT);
        CASE(Uniform, INLINE_UNIFORM_DATA);
        // TODO: register space

#undef CASE
    }
}

SlangBindingType _calcBindingType(Slang::TypeLayout* typeLayout, LayoutResourceKind kind)
{
    // At the type level, a push-constant buffer and a regular constant
    // buffer are currently not distinct, so we need to detect push
    // constant buffers/ranges before we inspect the `typeLayout` to
    // avoid reflecting them all as ordinary constant buffers.
    //
    switch (kind)
    {
    default:
        break;

    case LayoutResourceKind::PushConstantBuffer:
        return SLANG_BINDING_TYPE_PUSH_CONSTANT;
    }

    // If the type or type layout implies a specific binding type
    // (e.g., a `Texture2D` implies a texture binding), then we
    // will always favor the binding type implied.
    //
    if (auto bindingType = _calcResourceBindingType(typeLayout))
    {
        if (bindingType != SLANG_BINDING_TYPE_UNKNOWN)
            return bindingType;
    }

    // As a fallback, we may look at the kind of resources consumed
    // by a type layout, and use that to infer the type of binding
    // used. Note that, for example, a `float4` might represent
    // multiple different kinds of binding, depending on where/how
    // it is used (e.g., as a varying parameter, a root constant, etc.).
    //
    return _calcBindingType(kind);
}

static DeclRefType* asInterfaceType(Type* type)
{
    if (auto declRefType = as<DeclRefType>(type))
    {
        if (declRefType->getDeclRef().as<InterfaceDecl>())
        {
            return declRefType;
        }
    }
    return nullptr;
}

struct ExtendedTypeLayoutContext
{
    TypeLayout* m_typeLayout;
    TypeLayout::ExtendedInfo* m_extendedInfo;

    Dictionary<Int, Int> m_mapSpaceToDescriptorSetIndex;

    Int _findOrAddDescriptorSet(Int space)
    {
        Int index = 0;
        if (m_mapSpaceToDescriptorSetIndex.tryGetValue(space, index))
            return index;

        index = m_extendedInfo->m_descriptorSets.getCount();
        m_mapSpaceToDescriptorSetIndex.add(space, index);

        RefPtr<TypeLayout::ExtendedInfo::DescriptorSetInfo> descriptorSet =
            new TypeLayout::ExtendedInfo::DescriptorSetInfo();
        m_extendedInfo->m_descriptorSets.add(descriptorSet);

        return index;
    }

    /// Create a single `VarLayout` for `typeLayout` that summarizes all of the offset information
    /// in `path`.
    ///
    /// Note: This function does not handle "pending" layout information.
    RefPtr<VarLayout> _createSimpleOffsetVarLayout(
        TypeLayout* typeLayout,
        BindingRangePathLink* path)
    {
        SLANG_ASSERT(typeLayout);

        RefPtr<VarLayout> varLayout = new VarLayout();
        varLayout->typeLayout = typeLayout;
        varLayout->typeLayout.demoteToWeakReference();

        for (auto typeResInfo : typeLayout->resourceInfos)
        {
            auto kind = typeResInfo.kind;
            auto varResInfo = varLayout->findOrAddResourceInfo(kind);
            varResInfo->index = _calcIndexOffset(path, kind);
            varResInfo->space = _calcSpaceOffset(path, kind);
        }

        return varLayout;
    }

    /// Create a single `VarLayout` for `typeLayout` that summarizes all of the offset information
    /// in `path`.
    RefPtr<VarLayout> createOffsetVarLayout(TypeLayout* typeLayout, BindingRangePath const& path)
    {
        auto primaryVarLayout = _createSimpleOffsetVarLayout(typeLayout, path.primary);
        SLANG_ASSERT(primaryVarLayout);

        if (auto pendingDataTypeLayout = typeLayout->pendingDataTypeLayout)
        {
            primaryVarLayout->pendingVarLayout =
                _createSimpleOffsetVarLayout(pendingDataTypeLayout, path.pending);
        }

        return primaryVarLayout;
    }

    void addRangesRec(TypeLayout* typeLayout, BindingRangePath const& path, LayoutSize multiplier)
    {
        if (auto structTypeLayout = as<StructTypeLayout>(typeLayout))
        {
            // For a structure type, we need to recursively
            // add the ranges for each field.
            //
            // Along the way we will make sure to properly update
            // the offset information on the fields so that
            // they properly show their binding-range offset
            // within the parent type.
            //
            Index structBindingRangeIndex = m_extendedInfo->m_bindingRanges.getCount();
            for (auto fieldVarLayout : structTypeLayout->fields)
            {
                Index fieldBindingRangeIndex = m_extendedInfo->m_bindingRanges.getCount();
                fieldVarLayout->bindingRangeOffset =
                    fieldBindingRangeIndex - structBindingRangeIndex;

                auto fieldTypeLayout = fieldVarLayout->getTypeLayout();

                ExtendedBindingRangePath fieldPath(path, fieldVarLayout);
                addRangesRec(fieldTypeLayout, fieldPath, multiplier);
            }
            return;
        }
        else if (auto arrayTypeLayout = as<ArrayTypeLayout>(typeLayout))
        {
            // For an array, we need to recursively add the
            // element type of the array, but with an adjusted
            // `multiplier` to account for the element count.
            //
            auto elementTypeLayout = arrayTypeLayout->elementTypeLayout;
            LayoutSize elementCount = LayoutSize::infinite();
            if (auto arrayType = as<ArrayExpressionType>(arrayTypeLayout->type))
            {
                if (!arrayType->isUnsized())
                {
                    elementCount = LayoutSize::RawValue(getIntVal(arrayType->getElementCount()));
                }
            }
            addRangesRec(elementTypeLayout, path, multiplier * elementCount);
            return;
        }
        else if (auto parameterGroupTypeLayout = as<ParameterGroupTypeLayout>(typeLayout))
        {
            // A parameter group (whether a `ConstantBuffer<>` or `ParameterBlock<>`
            // introduces a separately-allocated "sub-object" in the application's
            // layout for shader objects.
            //
            // We will represent the parameter group with a single sub-object
            // binding range (and an associated sub-object range).
            //
            // We start out by looking at the resources consumed by the parameter group
            // itself, to determine what kind of binding range to report it as.
            //
            Index bindingRangeIndex = m_extendedInfo->m_bindingRanges.getCount();
            SlangBindingType bindingType = SLANG_BINDING_TYPE_CONSTANT_BUFFER;
            bool shouldAllocDescriptorSet = true;
            LayoutResourceKind kind = LayoutResourceKind::None;

            // If the parameter group container starts a new space,
            // we do not want to allocate a descriptor set from the current parent.
            if (parameterGroupTypeLayout->containerVarLayout->FindResourceInfo(
                    LayoutResourceKind::RegisterSpace))
            {
                kind = LayoutResourceKind::RegisterSpace;
                bindingType = SLANG_BINDING_TYPE_PARAMETER_BLOCK;
                shouldAllocDescriptorSet = false;
            }

            if (shouldAllocDescriptorSet)
            {
                // If this is not a parameter block, derive the binding type
                // from resource infos.
                for (auto& resInfo : parameterGroupTypeLayout->resourceInfos)
                {
                    kind = resInfo.kind;
                    switch (kind)
                    {
                    default:
                        continue;

                    case LayoutResourceKind::ConstantBuffer:
                    case LayoutResourceKind::PushConstantBuffer:
                    case LayoutResourceKind::DescriptorTableSlot:
                        break;

                        // Certain cases indicate a parameter block that
                        // actually involves indirection.
                        //
                        // Note: the only case where a parameter group should
                        // reflect as consuming `Uniform` storage is on CPU/CUDA,
                        // where that will be the only resource it contains.
                    case LayoutResourceKind::Uniform:
                        break;
                    }

                    bindingType = _calcBindingType(typeLayout, kind);
                    break;
                }
            }

            TypeLayout::ExtendedInfo::BindingRangeInfo bindingRange;
            bindingRange.leafTypeLayout = typeLayout;
            bindingRange.leafVariable = path.primary ? path.primary->var->getVariable() : nullptr;
            bindingRange.bindingType = bindingType;
            bindingRange.count = multiplier;
            bindingRange.descriptorSetIndex = -1;
            bindingRange.firstDescriptorRangeIndex = 0;
            bindingRange.descriptorRangeCount = 0;

            // Every parameter group will introduce a sub-object range,
            // which will include bindings based on the type of data
            // inside the sub-object.
            //
            TypeLayout::ExtendedInfo::SubObjectRangeInfo subObjectRange;
            subObjectRange.bindingRangeIndex = bindingRangeIndex;
            subObjectRange.offsetVarLayout = createOffsetVarLayout(typeLayout, path);
            subObjectRange.spaceOffset = 0;
            if (kind == LayoutResourceKind::SubElementRegisterSpace && path.primary)
            {
                if (auto resInfo = path.primary->var->FindResourceInfo(
                        LayoutResourceKind::SubElementRegisterSpace))
                {
                    subObjectRange.spaceOffset = resInfo->index;
                }
            }
            // It is possible that the sub-object has descriptor ranges
            // that will need to be exposed upward, into the parent.
            //
            // Note: it is a subtle point, but we are only going to expose
            // *descriptor ranges* upward and not *binding ranges*. The
            // distinction here comes down to:
            //
            // * Descriptor ranges are used to describe the entries that
            //   must be allocated in one or more API descriptor sets to
            //   physically hold a value of a given type (layout).
            //
            // * Binding ranges are used to describe the entries that must
            //   be allocated in an application shader object to logically
            //   hold a value of a given type (layout).
            //
            // In practice, a binding range might logically belong to a
            // sub-object, but physically belong to a parent. Consider:
            //
            //    cbuffer C { Texture2D a; float b; }
            //
            // Independent of the API we compile for, we expect the global
            // scope to have a sub-object for `C`, and for that sub-object
            // to have a binding range for `a` (that is, we bind the texture
            // into the sub-object).
            //
            // When compiling for D3D12 or Vulkan, we expect that the global
            // scope must have two descriptor ranges for `C`: one for the
            // constant buffer itself, and another for the texture `a`.
            // The reason for this is that `a` needs to be bound as part
            // of a descriptor set, and `C` doesn't create/allocate its own
            // descriptor set(s).
            //
            // When compiling for CPU or CUDA, we expect that the global scope
            // will have a descriptor range for `C` but *not* one for `C.a`,
            // because the physical storage for `C.a` is provided by the
            // memory allocation for `C` itself.

            if (shouldAllocDescriptorSet)
            {
                // The logic here assumes that when a parameter group consumes
                // resources that must "leak" into the outer scope (including
                // reosurces consumed by the group "container"), those resources
                // will amount to descriptor ranges that are part of the same
                // descriptor set.
                //
                // (If the contents of a group consume whole spaces/sets, then
                // those resources will be accounted for separately).
                //
                Int descriptorSetIndex = _findOrAddDescriptorSet(0);
                auto descriptorSet = m_extendedInfo->m_descriptorSets[descriptorSetIndex];
                auto firstDescriptorRangeIndex = descriptorSet->descriptorRanges.getCount();

                // First, we need to deal with any descriptor ranges that are
                // introduced by the "container" type itself.
                //
                switch (kind)
                {
                    // If the parameter group was allocated to consume one or
                    // more whole register spaces/sets, then nothing should
                    // leak through that is measured in descriptor sets.
                    //
                case LayoutResourceKind::SubElementRegisterSpace:
                case LayoutResourceKind::None:
                    break;

                default:
                    {
                        // In a constant-buffer-like case, then all the (non-space/set) resource
                        // usage of the "container" should be reflected as descriptor
                        // ranges in the parent scope.
                        //
                        for (auto resInfo : parameterGroupTypeLayout->containerVarLayout->typeLayout
                                                ->resourceInfos)
                        {
                            switch (resInfo.kind)
                            {
                            case LayoutResourceKind::SubElementRegisterSpace:
                                continue;

                            default:
                                break;
                            }

                            TypeLayout::ExtendedInfo::DescriptorRangeInfo descriptorRange;
                            descriptorRange.kind = resInfo.kind;
                            descriptorRange.bindingType =
                                _calcBindingType(typeLayout, resInfo.kind);
                            descriptorRange.count = multiplier;
                            descriptorRange.indexOffset =
                                _calcIndexOffset(path.primary, resInfo.kind);
                            descriptorSet->descriptorRanges.add(descriptorRange);
                        }
                    }
                }

                // Second, we need to consider resource usage from the "element"
                // type that might leak through to the parent.
                //
                switch (kind)
                {
                    // If the parameter group was allocated as a full register space/set,
                    // *or* if it was allocated as ordinary uniform storage (likely
                    // because it was compiled for CPU/CUDA), then there should
                    // be no "leakage" of descriptor ranges from the element type
                    // to the parent.
                    //
                case LayoutResourceKind::SubElementRegisterSpace:
                case LayoutResourceKind::Uniform:
                case LayoutResourceKind::None:
                    break;

                default:
                    {
                        // If we are in the constant-buffer-like case, on an API
                        // where constant bufers "leak" resource usage to the
                        // outer context, then we need to add the descriptor ranges
                        // implied by the element type.
                        //
                        // HACK: We enumerate these nested ranges by recurisvely
                        // calling `addRangesRec`, which adds all of descriptor ranges,
                        // binding ranges, and sub-object ranges, and then we trim
                        // the lists we don't actually care about as a post-process.
                        //
                        // TODO: We could try to consider a model where we first
                        // query the extended layout information of the element
                        // type (which might already be cached) and then enumerate
                        // the descriptor ranges and copy them over.
                        //
                        // TODO: It is possible that there could be cases where
                        // some, but not all, of the nested descriptor ranges ought
                        // to be enumerated here. In that case we might have to introduce
                        // a kind of "mask" parameter that is passed down into
                        // the recursive call so that only the appropriate ranges
                        // get added.

                        // We need to add a link to the "path" that is used when looking
                        // up binding information, to ensure that the descriptor ranges
                        // that get enumerated here have correct register/binding offsets.
                        //
                        ExtendedBindingRangePath elementPath(
                            path,
                            parameterGroupTypeLayout->elementVarLayout);

                        Index bindingRangeCountBefore = m_extendedInfo->m_bindingRanges.getCount();
                        Index subObjectRangeCountBefore =
                            m_extendedInfo->m_subObjectRanges.getCount();

                        addRangesRec(
                            parameterGroupTypeLayout->elementVarLayout->typeLayout,
                            elementPath,
                            multiplier);

                        m_extendedInfo->m_bindingRanges.setCount(bindingRangeCountBefore);
                        m_extendedInfo->m_subObjectRanges.setCount(subObjectRangeCountBefore);
                    }
                    break;
                }

                auto descriptorRangeCount =
                    descriptorSet->descriptorRanges.getCount() - firstDescriptorRangeIndex;
                bindingRange.descriptorSetIndex = descriptorSetIndex;
                bindingRange.firstDescriptorRangeIndex = firstDescriptorRangeIndex;
                bindingRange.descriptorRangeCount = descriptorRangeCount;
            }

            m_extendedInfo->m_bindingRanges.add(bindingRange);
            m_extendedInfo->m_subObjectRanges.add(subObjectRange);
            return;
        }
        else if (asInterfaceType(typeLayout->type))
        {
            // An `interface` type should introduce a binding range and a matching
            // sub-object range.
            //
            TypeLayout::ExtendedInfo::BindingRangeInfo bindingRange;
            bindingRange.leafTypeLayout = typeLayout;
            bindingRange.leafVariable = path.primary ? path.primary->var->getVariable() : nullptr;
            bindingRange.bindingType = SLANG_BINDING_TYPE_EXISTENTIAL_VALUE;
            bindingRange.count = multiplier;
            bindingRange.descriptorSetIndex = 0;
            bindingRange.descriptorRangeCount = 0;
            bindingRange.firstDescriptorRangeIndex = 0;

            TypeLayout::ExtendedInfo::SubObjectRangeInfo subObjectRange;
            subObjectRange.bindingRangeIndex = m_extendedInfo->m_bindingRanges.getCount();
            subObjectRange.offsetVarLayout = createOffsetVarLayout(typeLayout, path);

            m_extendedInfo->m_bindingRanges.add(bindingRange);
            m_extendedInfo->m_subObjectRanges.add(subObjectRange);
        }
        else if (const auto structuredBufferTypeLayout = as<StructuredBufferTypeLayout>(typeLayout))
        {
            // For structured buffers we expect them to consume a single
            // resource descriptor slot (not counting the possible counter
            // buffer)
            SLANG_ASSERT(typeLayout->resourceInfos.getCount() >= 1);
            TypeLayout::ResourceInfo resInfo;
            for (auto& info : typeLayout->resourceInfos)
            {
                switch (info.kind)
                {
                case LayoutResourceKind::UnorderedAccess:
                case LayoutResourceKind::ShaderResource:
                case LayoutResourceKind::DescriptorTableSlot:
                case LayoutResourceKind::Uniform:
                case LayoutResourceKind::ConstantBuffer: // for metal
                case LayoutResourceKind::MetalArgumentBufferElement:
                    resInfo = info;
                    break;
                }
            }
            SLANG_ASSERT(resInfo.kind != LayoutResourceKind::None);

            const auto bindingType = as<HLSLStructuredBufferType>(typeLayout->getType())
                                         ? SLANG_BINDING_TYPE_RAW_BUFFER
                                         : SLANG_BINDING_TYPE_MUTABLE_RAW_BUFFER;

            // We now allocate a descriptor range for this buffer
            TypeLayout::ExtendedInfo::DescriptorRangeInfo descriptorRange;
            descriptorRange.kind = resInfo.kind;
            descriptorRange.bindingType = bindingType;
            // Note that we don't use resInfo.count here, as each
            // structuredBufferType is essentially a struct of 2 fields
            // (elements, counter) and not an array of length 2.
            SLANG_ASSERT(resInfo.count != 2 || structuredBufferTypeLayout->counterVarLayout);
            SLANG_ASSERT(resInfo.count != 1 || !structuredBufferTypeLayout->counterVarLayout);
            descriptorRange.count = multiplier;
            descriptorRange.indexOffset = _calcIndexOffset(path.primary, resInfo.kind);

            Int descriptorSetIndex =
                _findOrAddDescriptorSet(_calcSpaceOffset(path.primary, resInfo.kind));
            const RefPtr<TypeLayout::ExtendedInfo::DescriptorSetInfo> descriptorSet =
                m_extendedInfo->m_descriptorSets[descriptorSetIndex];
            auto descriptorRangeIndex = descriptorSet->descriptorRanges.getCount();
            descriptorSet->descriptorRanges.add(descriptorRange);

            // We will map the elements buffer to a single binding range
            TypeLayout::ExtendedInfo::BindingRangeInfo bindingRange;
            bindingRange.leafTypeLayout = typeLayout;
            bindingRange.leafVariable = path.primary ? path.primary->var->getVariable() : nullptr;
            bindingRange.bindingType = bindingType;
            bindingRange.count = multiplier;
            bindingRange.descriptorSetIndex = descriptorSetIndex;
            bindingRange.firstDescriptorRangeIndex = descriptorRangeIndex;
            bindingRange.descriptorRangeCount = 1;

            auto bindingRangeIndex = m_extendedInfo->m_bindingRanges.getCount();
            m_extendedInfo->m_bindingRanges.add(bindingRange);

            // We also make sure to report it as a sub-object range.
            TypeLayout::ExtendedInfo::SubObjectRangeInfo subObjectRange;
            subObjectRange.bindingRangeIndex = bindingRangeIndex;
            subObjectRange.offsetVarLayout = createOffsetVarLayout(typeLayout, path);
            subObjectRange.spaceOffset = 0;
            m_extendedInfo->m_subObjectRanges.add(subObjectRange);

            // If we have an associated counter for this structured buffer,
            // add its ranges
            if (structuredBufferTypeLayout->counterVarLayout)
            {
                ExtendedBindingRangePath counterPath(
                    path,
                    structuredBufferTypeLayout->counterVarLayout);
                // This should always be 1, because it comes after the
                // single binding range we just added
                structuredBufferTypeLayout->counterVarLayout->bindingRangeOffset =
                    m_extendedInfo->m_bindingRanges.getCount() - bindingRangeIndex;
                addRangesRec(
                    structuredBufferTypeLayout->counterVarLayout->typeLayout,
                    counterPath,
                    multiplier);
            }
        }
        else
        {
            // Here we have the catch-all case that handles "leaf" fields
            // that might need to introduce a binding range and descriptor
            // ranges.
            //
            // First, we want to determine what type of binding this
            // leaf field should map to, if any. We being by querying
            // the type itself, since there are many distinct descriptor
            // types for textures/buffers that can only be determined
            // by type, rather than by a `LayoutResourceKind`.
            //
            auto bindingType = _calcResourceBindingType(typeLayout);

            // It is possible that the type alone isn't enough to tell
            // us a specific binding type, at which point we need to
            // start looking at the actual resources the type layout
            // consumes.
            //
            if (bindingType == SLANG_BINDING_TYPE_UNKNOWN)
            {
                // We will search through all the resource kinds that
                // the type layout consumes, to see if we can find
                // one that indicates a binding type we actually
                // want to reflect.
                //
                for (auto resInfo : typeLayout->resourceInfos)
                {
                    auto kind = resInfo.kind;
                    if (kind == LayoutResourceKind::Uniform)
                        continue;

                    auto kindBindingType = _calcBindingType(kind);
                    if (kindBindingType == SLANG_BINDING_TYPE_UNKNOWN)
                        continue;

                    // If we find a relevant binding type based on
                    // one of the resource kinds that are consumed,
                    // then we immediately stop the search and use
                    // the first one found (whether or not later
                    // entries might also provide something relevant).
                    //
                    bindingType = kindBindingType;
                    break;
                }
            }

            // After we've tried to determine a binding type, if
            // we have nothing to go on then we don't want to add
            // a binding range.
            //
            if (bindingType == SLANG_BINDING_TYPE_UNKNOWN)
                return;

            // We now know that the leaf field will map to a single binding range,
            // and zero or more descriptor ranges.
            //
            TypeLayout::ExtendedInfo::BindingRangeInfo bindingRange;
            bindingRange.leafTypeLayout = typeLayout;
            bindingRange.leafVariable = path.primary ? path.primary->var->getVariable() : nullptr;
            bindingRange.bindingType = bindingType;
            bindingRange.count = multiplier;
            bindingRange.descriptorSetIndex = 0;
            bindingRange.firstDescriptorRangeIndex = 0;
            bindingRange.descriptorRangeCount = 0;

            // We will associate the binding range with a specific descriptor
            // set on demand *if* we discover that it shold contain any
            // descriptor ranges.
            //
            RefPtr<TypeLayout::ExtendedInfo::DescriptorSetInfo> descriptorSet;


            // We will add a descriptor range for each relevant resource kind
            // that the type layout consumes.
            //
            for (auto resInfo : typeLayout->resourceInfos)
            {
                auto kind = resInfo.kind;
                switch (kind)
                {
                default:
                    break;


                    // There are many resource kinds that we do not want
                    // to expose as descriptor ranges simply because they
                    // do not actually allocate descriptors on our target
                    // APIs.
                    //
                    // Notably included here are uniform/ordinary data and
                    // varying input/output (including the ray-tracing cases).
                    //
                    // It is worth noting that we *do* allow root/push-constant
                    // ranges to be reflected as "descriptor" ranges here,
                    // despite the fact that they are not descriptor-bound
                    // under D3D12/Vulkan.
                    //
                    // In practice, even with us filtering out some cases here,
                    // an application/renderer layer will need to filter/translate
                    // or descriptor ranges into API-specific ones, and a one-to-one
                    // mapping should not be assumed.
                    //
                    // TODO: Make some clear decisions about what should and should
                    // not appear here.
                    //
                case LayoutResourceKind::SubElementRegisterSpace:
                case LayoutResourceKind::VaryingInput:
                case LayoutResourceKind::VaryingOutput:
                case LayoutResourceKind::HitAttributes:
                case LayoutResourceKind::RayPayload:
                case LayoutResourceKind::ExistentialTypeParam:
                case LayoutResourceKind::ExistentialObjectParam:
                    continue;
                }

                // We will prefer to use a binding type derived from the specific
                // resource kind, but will fall back to information from the
                // type layout when that is not available.
                //
                // TODO: This logic probably needs a bit more work to handle
                // the case of a combined texture-sampler field that is being
                // compiled for an API with separate textures and samplers.
                //
                auto kindBindingType = _calcBindingType(kind);
                if (kindBindingType == SLANG_BINDING_TYPE_UNKNOWN)
                {
                    kindBindingType = bindingType;
                }

                // We now expect to allocate a descriptor range for this
                // `resInfo` representing resouce usage.
                //
                auto count = resInfo.count * multiplier;
                auto indexOffset = _calcIndexOffset(path.primary, kind);
                auto spaceOffset = _calcSpaceOffset(path.primary, kind);

                TypeLayout::ExtendedInfo::DescriptorRangeInfo descriptorRange;
                descriptorRange.kind = kind;
                descriptorRange.bindingType = kindBindingType;
                descriptorRange.count = count;
                descriptorRange.indexOffset = indexOffset;

                if (!descriptorSet)
                {
                    Int descriptorSetIndex = _findOrAddDescriptorSet(spaceOffset);
                    descriptorSet = m_extendedInfo->m_descriptorSets[descriptorSetIndex];

                    bindingRange.descriptorSetIndex = descriptorSetIndex;
                    bindingRange.firstDescriptorRangeIndex =
                        descriptorSet->descriptorRanges.getCount();
                }

                descriptorSet->descriptorRanges.add(descriptorRange);
                bindingRange.descriptorRangeCount++;
            }

            m_extendedInfo->m_bindingRanges.add(bindingRange);
        }
    }
};

TypeLayout::ExtendedInfo* getExtendedTypeLayout(TypeLayout* typeLayout)
{
    if (!typeLayout->m_extendedInfo)
    {
        RefPtr<TypeLayout::ExtendedInfo> extendedInfo = new TypeLayout::ExtendedInfo;

        ExtendedTypeLayoutContext context;
        context.m_typeLayout = typeLayout;
        context.m_extendedInfo = extendedInfo;

        BindingRangePath rootPath;
        context.addRangesRec(typeLayout, rootPath, 1);

        typeLayout->m_extendedInfo = extendedInfo;
    }
    return typeLayout->m_extendedInfo;
}
} // namespace Slang

SLANG_API SlangInt
spReflectionTypeLayout_getBindingRangeCount(SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);
    return extTypeLayout->m_bindingRanges.getCount();
}

SLANG_API SlangBindingType
spReflectionTypeLayout_getBindingRangeType(SlangReflectionTypeLayout* inTypeLayout, SlangInt index)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return SLANG_BINDING_TYPE_UNKNOWN;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);
    if (index < 0)
        return SLANG_BINDING_TYPE_UNKNOWN;
    if (index >= extTypeLayout->m_bindingRanges.getCount())
        return SLANG_BINDING_TYPE_UNKNOWN;
    auto& bindingRange = extTypeLayout->m_bindingRanges[index];

    return bindingRange.bindingType;
}

SLANG_API SlangInt spReflectionTypeLayout_isBindingRangeSpecializable(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt index)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return SLANG_BINDING_TYPE_UNKNOWN;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);
    if (index < 0)
        return SLANG_BINDING_TYPE_UNKNOWN;
    if (index >= extTypeLayout->m_bindingRanges.getCount())
        return SLANG_BINDING_TYPE_UNKNOWN;
    auto& bindingRange = extTypeLayout->m_bindingRanges[index];
    auto type = bindingRange.leafTypeLayout->getType();
    if (asInterfaceType(type))
        return 1;
    if (auto parameterGroupType = as<ParameterGroupType>(type))
    {
        if (asInterfaceType(parameterGroupType->getElementType()))
            return 1;
    }
    return 0;
}

SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeBindingCount(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt index)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);
    if (index < 0)
        return 0;
    if (index >= extTypeLayout->m_bindingRanges.getCount())
        return 0;
    auto& bindingRange = extTypeLayout->m_bindingRanges[index];

    auto count = bindingRange.count;
    return count.isFinite() ? SlangInt(count.getFiniteValue()) : -1;
}

#if 0
SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeIndexOffset(SlangReflectionTypeLayout* inTypeLayout, SlangInt index)
{
    auto typeLayout = convert(inTypeLayout);
    if(!typeLayout) return 0;

    return Slang::_findBindingRange(typeLayout, index).indexOffset;
}

SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeSpaceOffset(SlangReflectionTypeLayout* inTypeLayout, SlangInt index)
{
    auto typeLayout = convert(inTypeLayout);
    if(!typeLayout) return 0;

    return Slang::_findBindingRange(typeLayout, index).spaceOffset;
}
#endif

SLANG_API SlangReflectionTypeLayout* spReflectionTypeLayout_getBindingRangeLeafTypeLayout(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt index)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);
    if (index < 0)
        return 0;
    if (index >= extTypeLayout->m_bindingRanges.getCount())
        return 0;
    auto& bindingRange = extTypeLayout->m_bindingRanges[index];

    return convert(bindingRange.leafTypeLayout);
}

SLANG_API SlangReflectionVariable* spReflectionTypeLayout_getBindingRangeLeafVariable(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt index)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);
    if (index < 0)
        return 0;
    if (index >= extTypeLayout->m_bindingRanges.getCount())
        return 0;
    auto& bindingRange = extTypeLayout->m_bindingRanges[index];

    return convert(DeclRef<Decl>(bindingRange.leafVariable));
}

SLANG_API SlangImageFormat spReflectionTypeLayout_getBindingRangeImageFormat(
    SlangReflectionTypeLayout* typeLayout,
    SlangInt index)
{
    auto typeLayout_ = convert(typeLayout);
    if (!typeLayout_)
        return SLANG_IMAGE_FORMAT_unknown;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout_);
    if (index < 0)
        return SLANG_IMAGE_FORMAT_unknown;
    if (index >= extTypeLayout->m_bindingRanges.getCount())
        return SLANG_IMAGE_FORMAT_unknown;
    auto& bindingRange = extTypeLayout->m_bindingRanges[index];

    auto leafVar = bindingRange.leafVariable;
    if (auto formatAttrib = leafVar->findModifier<FormatAttribute>())
    {
        return (SlangImageFormat)formatAttrib->format;
    }
    return SLANG_IMAGE_FORMAT_unknown;
}


SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeDescriptorSetIndex(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt index)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);
    if (index < 0)
        return 0;
    if (index >= extTypeLayout->m_bindingRanges.getCount())
        return 0;
    auto& bindingRange = extTypeLayout->m_bindingRanges[index];

    return bindingRange.descriptorSetIndex;
}

SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeFirstDescriptorRangeIndex(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt index)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);
    if (index < 0)
        return 0;
    if (index >= extTypeLayout->m_bindingRanges.getCount())
        return 0;
    auto& bindingRange = extTypeLayout->m_bindingRanges[index];

    return bindingRange.firstDescriptorRangeIndex;
}

SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeDescriptorRangeCount(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt index)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);
    if (index < 0)
        return 0;
    if (index >= extTypeLayout->m_bindingRanges.getCount())
        return 0;
    auto& bindingRange = extTypeLayout->m_bindingRanges[index];

    return bindingRange.descriptorRangeCount;
}

SLANG_API SlangInt
spReflectionTypeLayout_getDescriptorSetCount(SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);

    return extTypeLayout->m_descriptorSets.getCount();
}

SLANG_API SlangInt spReflectionTypeLayout_getDescriptorSetSpaceOffset(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt setIndex)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);

    if (setIndex < 0)
        return 0;
    if (setIndex >= extTypeLayout->m_descriptorSets.getCount())
        return 0;
    auto descriptorSet = extTypeLayout->m_descriptorSets[setIndex];

    return descriptorSet->spaceOffset;
}

SLANG_API SlangInt spReflectionTypeLayout_getDescriptorSetDescriptorRangeCount(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt setIndex)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);

    if (setIndex < 0)
        return 0;
    if (setIndex >= extTypeLayout->m_descriptorSets.getCount())
        return 0;
    auto descriptorSet = extTypeLayout->m_descriptorSets[setIndex];

    return descriptorSet->descriptorRanges.getCount();
}

SLANG_API SlangInt spReflectionTypeLayout_getDescriptorSetDescriptorRangeIndexOffset(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt setIndex,
    SlangInt rangeIndex)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);

    if (setIndex < 0)
        return 0;
    if (setIndex >= extTypeLayout->m_descriptorSets.getCount())
        return 0;
    auto descriptorSet = extTypeLayout->m_descriptorSets[setIndex];

    if (rangeIndex < 0)
        return 0;
    if (rangeIndex >= descriptorSet->descriptorRanges.getCount())
        return 0;
    auto& range = descriptorSet->descriptorRanges[rangeIndex];

    return range.indexOffset;
}

SLANG_API SlangInt spReflectionTypeLayout_getDescriptorSetDescriptorRangeDescriptorCount(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt setIndex,
    SlangInt rangeIndex)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);

    if (setIndex < 0)
        return 0;
    if (setIndex >= extTypeLayout->m_descriptorSets.getCount())
        return 0;
    auto descriptorSet = extTypeLayout->m_descriptorSets[setIndex];

    if (rangeIndex < 0)
        return 0;
    if (rangeIndex >= descriptorSet->descriptorRanges.getCount())
        return 0;
    auto& range = descriptorSet->descriptorRanges[rangeIndex];

    auto count = range.count;
    return count.isFinite() ? count.getFiniteValue() : -1;
}

SLANG_API SlangBindingType spReflectionTypeLayout_getDescriptorSetDescriptorRangeType(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt setIndex,
    SlangInt rangeIndex)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return SLANG_BINDING_TYPE_UNKNOWN;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);

    if (setIndex < 0)
        return SLANG_BINDING_TYPE_UNKNOWN;
    if (setIndex >= extTypeLayout->m_descriptorSets.getCount())
        return SLANG_BINDING_TYPE_UNKNOWN;
    auto descriptorSet = extTypeLayout->m_descriptorSets[setIndex];

    if (rangeIndex < 0)
        return SLANG_BINDING_TYPE_UNKNOWN;
    if (rangeIndex >= descriptorSet->descriptorRanges.getCount())
        return SLANG_BINDING_TYPE_UNKNOWN;
    auto& range = descriptorSet->descriptorRanges[rangeIndex];

    return range.bindingType;
}

SLANG_API SlangParameterCategory spReflectionTypeLayout_getDescriptorSetDescriptorRangeCategory(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt setIndex,
    SlangInt rangeIndex)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return SLANG_PARAMETER_CATEGORY_NONE;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);

    if (setIndex < 0)
        return SLANG_PARAMETER_CATEGORY_NONE;
    if (setIndex >= extTypeLayout->m_descriptorSets.getCount())
        return SLANG_PARAMETER_CATEGORY_NONE;
    auto descriptorSet = extTypeLayout->m_descriptorSets[setIndex];

    if (rangeIndex < 0)
        return SLANG_PARAMETER_CATEGORY_NONE;
    if (rangeIndex >= descriptorSet->descriptorRanges.getCount())
        return SLANG_PARAMETER_CATEGORY_NONE;
    auto& range = descriptorSet->descriptorRanges[rangeIndex];

    return SlangParameterCategory(range.kind);
}

SLANG_API SlangInt
spReflectionTypeLayout_getSubObjectRangeCount(SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);

    return extTypeLayout->m_subObjectRanges.getCount();
}

SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeBindingRangeIndex(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt subObjectRangeIndex)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);

    if (subObjectRangeIndex < 0)
        return 0;
    if (subObjectRangeIndex >= extTypeLayout->m_subObjectRanges.getCount())
        return 0;

    return extTypeLayout->m_subObjectRanges[subObjectRangeIndex].bindingRangeIndex;
}

SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeSpaceOffset(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt subObjectRangeIndex)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);

    if (subObjectRangeIndex < 0)
        return 0;
    if (subObjectRangeIndex >= extTypeLayout->m_subObjectRanges.getCount())
        return 0;

    return extTypeLayout->m_subObjectRanges[subObjectRangeIndex].spaceOffset;
}

SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_getSubObjectRangeOffset(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt subObjectRangeIndex)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    auto extTypeLayout = Slang::getExtendedTypeLayout(typeLayout);

    if (subObjectRangeIndex < 0)
        return 0;
    if (subObjectRangeIndex >= extTypeLayout->m_subObjectRanges.getCount())
        return 0;

    return convert(extTypeLayout->m_subObjectRanges[subObjectRangeIndex].offsetVarLayout);
}


#if 0
SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeSubObjectRangeIndex(SlangReflectionTypeLayout* inTypeLayout, SlangInt index)
{
    auto typeLayout = convert(inTypeLayout);
    if(!typeLayout) return 0;

    return Slang::_findBindingRange(typeLayout, index).subObjectRangeIndex;
}
#endif


SLANG_API SlangInt spReflectionTypeLayout_getFieldBindingRangeOffset(
    SlangReflectionTypeLayout* inTypeLayout,
    SlangInt fieldIndex)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    if (auto structTypeLayout = as<StructTypeLayout>(typeLayout))
    {
        getExtendedTypeLayout(structTypeLayout);

        return structTypeLayout->fields[fieldIndex]->bindingRangeOffset;
    }

    return 0;
}

SLANG_API SlangInt
spReflectionTypeLayout_getExplicitCounterBindingRangeOffset(SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if (!typeLayout)
        return 0;

    if (const auto structuredBufferTypeLayout = as<StructuredBufferTypeLayout>(typeLayout))
    {
        getExtendedTypeLayout(structuredBufferTypeLayout);
        return structuredBufferTypeLayout->counterVarLayout
                   ? structuredBufferTypeLayout->counterVarLayout->bindingRangeOffset
                   : 0;
    }

    return 0;
}

#if 0
SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeCount(SlangReflectionTypeLayout* inTypeLayout)
{
    auto typeLayout = convert(inTypeLayout);
    if(!typeLayout) return 0;

    return Slang::_calcSubObjectRangeCount(typeLayout);
}

SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeObjectCount(SlangReflectionTypeLayout* inTypeLayout, SlangInt index)
{
    auto typeLayout = convert(inTypeLayout);
    if(!typeLayout) return 0;

    auto count = Slang::_findSubObjectRange(typeLayout, index).count;
    return count.isFinite() ? SlangInt(count.getFiniteValue()) : -1;
}

SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeBindingRangeIndex(SlangReflectionTypeLayout* inTypeLayout, SlangInt index)
{
    auto typeLayout = convert(inTypeLayout);
    if(!typeLayout) return 0;

    return Slang::_findSubObjectRange(typeLayout, index).bindingRangeIndex;
}


SLANG_API SlangReflectionTypeLayout* spReflectionTypeLayout_getSubObjectRangeTypeLayout(SlangReflectionTypeLayout* inTypeLayout, SlangInt index)
{
    auto typeLayout = convert(inTypeLayout);
    if(!typeLayout) return 0;

    return convert(Slang::_findSubObjectRange(typeLayout, index).leafTypeLayout);
}

SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeDescriptorRangeCount(SlangReflectionTypeLayout* inTypeLayout, SlangInt subObjectRangeIndex)
{
    auto typeLayout = convert(inTypeLayout);
    if(!typeLayout) return 0;

    auto subObjectRange = Slang::_findSubObjectRange(typeLayout, subObjectRangeIndex);
    return Slang::_getSubObjectDescriptorRangeCount(subObjectRange);
}

SLANG_API SlangBindingType spReflectionTypeLayout_getSubObjectRangeDescriptorRangeBindingType(SlangReflectionTypeLayout* inTypeLayout, SlangInt subObjectRangeIndex, SlangInt bindingRangeIndexInSubObject)
{
    auto typeLayout = convert(inTypeLayout);
    if(!typeLayout) return 0;

    auto subObjectRange = Slang::_findSubObjectRange(typeLayout, subObjectRangeIndex);
    return Slang::_getSubObjectDescriptorRange(subObjectRange, bindingRangeIndexInSubObject).bindingType;
}

SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeDescriptorRangeBindingCount(SlangReflectionTypeLayout* inTypeLayout, SlangInt subObjectRangeIndex, SlangInt bindingRangeIndexInSubObject)
{
    auto typeLayout = convert(inTypeLayout);
    if(!typeLayout) return 0;

    auto subObjectRange = Slang::_findSubObjectRange(typeLayout, subObjectRangeIndex);
    auto count = Slang::_getSubObjectDescriptorRange(subObjectRange, bindingRangeIndexInSubObject).count;
    return count.isFinite() ? count.getFiniteValue() : -1;
}

SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeDescriptorRangeIndexOffset(SlangReflectionTypeLayout* inTypeLayout, SlangInt subObjectRangeIndex, SlangInt bindingRangeIndexInSubObject)
{
    auto typeLayout = convert(inTypeLayout);
    if(!typeLayout) return 0;

    auto subObjectRange = Slang::_findSubObjectRange(typeLayout, subObjectRangeIndex);
    return Slang::_getSubObjectDescriptorRange(subObjectRange, bindingRangeIndexInSubObject).indexOffset;
}

SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeDescriptorRangeSpaceOffset(SlangReflectionTypeLayout* inTypeLayout, SlangInt subObjectRangeIndex, SlangInt bindingRangeIndexInSubObject)
{
    auto typeLayout = convert(inTypeLayout);
    if(!typeLayout) return 0;

    auto subObjectRange = Slang::_findSubObjectRange(typeLayout, subObjectRangeIndex);
    return Slang::_getSubObjectDescriptorRange(subObjectRange, bindingRangeIndexInSubObject).spaceOffset;
}
#endif

// Variable Reflection

SLANG_API char const* spReflectionVariable_GetName(SlangReflectionVariable* inVar)
{
    auto var = convert(inVar).getDecl();
    if (as<InheritanceDecl>(var))
        return "$base";

    if (!var)
        return nullptr;

    // If the variable is one that has an "external" name that is supposed
    // to be exposed for reflection, then report it here
    if (auto reflectionNameMod = var->findModifier<ParameterGroupReflectionName>())
        return getText(reflectionNameMod->nameAndLoc.name).getBuffer();

    return getText(var->getName()).getBuffer();
}

SLANG_API SlangReflectionType* spReflectionVariable_GetType(SlangReflectionVariable* inVar)
{
    auto var = convert(inVar);

    if (!var)
        return nullptr;

    auto astBuilder = getModule(var.getDecl())->getLinkage()->getASTBuilder();

    if (auto inheritanceDecl = as<InheritanceDecl>(var.getDecl()))
        return convert(inheritanceDecl->base.type);

    if (auto varDecl = as<VarDeclBase>(var.getDecl()))
        return convert(substituteType(SubstitutionSet(var), astBuilder, varDecl->getType()));

    return nullptr;
}

SLANG_API SlangReflectionModifier* spReflectionVariable_FindModifier(
    SlangReflectionVariable* inVar,
    SlangModifierID modifierID)
{
    auto var = convert(inVar).getDecl();

    if (!var)
        return nullptr;

    Modifier* modifier = nullptr;
    switch (modifierID)
    {
    case SLANG_MODIFIER_SHARED:
        modifier = var->findModifier<HLSLEffectSharedModifier>();
        break;
    case SLANG_MODIFIER_CONST:
        modifier = var->findModifier<ConstModifier>();
        break;
    case SLANG_MODIFIER_NO_DIFF:
        modifier = var->findModifier<NoDiffModifier>();
        break;
    case SLANG_MODIFIER_STATIC:
        modifier = var->findModifier<HLSLStaticModifier>();
        break;
    case SLANG_MODIFIER_EXPORT:
        modifier = var->findModifier<HLSLExportModifier>();
        break;
    case SLANG_MODIFIER_EXTERN:
        modifier = var->findModifier<ExternModifier>();
        break;
    case SLANG_MODIFIER_DIFFERENTIABLE:
        modifier = var->findModifier<DifferentiableAttribute>();
        break;
    case SLANG_MODIFIER_MUTATING:
        modifier = var->findModifier<MutatingAttribute>();
        break;
    case SLANG_MODIFIER_IN:
        modifier = var->findModifier<InModifier>();
        break;
    case SLANG_MODIFIER_OUT:
        modifier = var->findModifier<OutModifier>();
        break;
    case SLANG_MODIFIER_INOUT:
        modifier = var->findModifier<InOutModifier>();
        break;
    default:
        return nullptr;
    }

    return (SlangReflectionModifier*)modifier;
}

SLANG_API unsigned int spReflectionVariable_GetUserAttributeCount(SlangReflectionVariable* inVar)
{
    auto varDecl = convert(inVar).getDecl();
    if (!varDecl)
        return 0;
    return getUserAttributeCount(varDecl);
}
SLANG_API SlangReflectionUserAttribute* spReflectionVariable_GetUserAttribute(
    SlangReflectionVariable* inVar,
    unsigned int index)
{
    auto varDecl = convert(inVar).getDecl();
    if (!varDecl)
        return 0;
    return getUserAttributeByIndex(varDecl, index);
}
SLANG_API SlangReflectionUserAttribute* spReflectionVariable_FindUserAttributeByName(
    SlangReflectionVariable* inVar,
    SlangSession* session,
    char const* name)
{
    auto varDecl = convert(inVar).getDecl();
    if (!varDecl)
        return 0;
    return findUserAttributeByName(asInternal(session), varDecl, name);
}

SLANG_API bool spReflectionVariable_HasDefaultValue(SlangReflectionVariable* inVar)
{
    auto decl = convert(inVar).getDecl();
    if (auto varDecl = as<VarDeclBase>(decl))
    {
        return varDecl->initExpr != nullptr;
    }

    return false;
}

SLANG_API SlangResult
spReflectionVariable_GetDefaultValueInt(SlangReflectionVariable* inVar, int64_t* rs)
{
    auto decl = convert(inVar).getDecl();
    if (auto varDecl = as<VarDeclBase>(decl))
    {
        if (auto constantVal = as<ConstantIntVal>(varDecl->val))
        {
            *rs = constantVal->getValue();
            return 0;
        }
    }

    return SLANG_E_INVALID_ARG;
}

SLANG_API SlangReflectionGeneric* spReflectionVariable_GetGenericContainer(
    SlangReflectionVariable* var)
{
    auto declRef = convert(var);
    return convertDeclToGeneric(getInnermostGenericParent(declRef));
}

SLANG_API SlangReflectionVariable* spReflectionVariable_applySpecializations(
    SlangReflectionVariable* var,
    SlangReflectionGeneric* generic)
{
    auto declRef = convert(var);
    auto genericDeclRef = convertGenericToDeclRef(generic);
    if (!declRef || !genericDeclRef)
        return nullptr;

    auto astBuilder = getModule(declRef.getDecl())->getLinkage()->getASTBuilder();

    auto substDeclRef = substituteDeclRef(SubstitutionSet(genericDeclRef), astBuilder, declRef);
    return convert(substDeclRef);
}

// Variable Layout Reflection

SLANG_API SlangReflectionVariable* spReflectionVariableLayout_GetVariable(
    SlangReflectionVariableLayout* inVarLayout)
{
    auto varLayout = convert(inVarLayout);
    if (!varLayout)
        return nullptr;

    return convert(varLayout->varDecl);
}

SLANG_API SlangReflectionTypeLayout* spReflectionVariableLayout_GetTypeLayout(
    SlangReflectionVariableLayout* inVarLayout)
{
    auto varLayout = convert(inVarLayout);
    if (!varLayout)
        return nullptr;

    return convert(varLayout->getTypeLayout());
}

SLANG_API size_t spReflectionVariableLayout_GetOffset(
    SlangReflectionVariableLayout* inVarLayout,
    SlangParameterCategory category)
{
    auto varLayout = convert(inVarLayout);
    if (!varLayout)
        return 0;

    auto info = varLayout->FindResourceInfo(LayoutResourceKind(category));

    if (!info)
    {
        // No match with requested category? Try again with one they might have meant...
        category = maybeRemapParameterCategory(varLayout->getTypeLayout(), category);
        info = varLayout->FindResourceInfo(LayoutResourceKind(category));
    }

    if (!info)
        return 0;

    return info->index;
}

SLANG_API size_t spReflectionVariableLayout_GetSpace(
    SlangReflectionVariableLayout* inVarLayout,
    SlangParameterCategory category)
{
    auto varLayout = convert(inVarLayout);
    if (!varLayout)
        return 0;


    auto info = varLayout->FindResourceInfo(LayoutResourceKind(category));
    if (!info)
    {
        // No match with requested category? Try again with one they might have meant...
        category = maybeRemapParameterCategory(varLayout->getTypeLayout(), category);
        info = varLayout->FindResourceInfo(LayoutResourceKind(category));
    }

    UInt space = 0;

    // First, deal with any offset applied to the specific resource kind specified
    if (info)
    {
        space += info->space;
    }

    if (auto regSpaceInfo = varLayout->FindResourceInfo(LayoutResourceKind::RegisterSpace))
        space += regSpaceInfo->index;

    // Note: this code used to try and take a variable with
    // an offset for `LayoutResourceKind::RegisterSpace` and
    // add it to the space returned, but that isn't going
    // to be right in some cases.
    //
    // Imageine if we have:
    //
    //  struct X { Texture2D y; }
    //  struct S { Texture2D t; ParmaeterBlock<X> x; }
    //
    //  Texture2D gA;
    //  S gS;
    //
    // We expect `gS` to have an offset for `LayoutResourceKind::ShaderResourceView`
    // of one (since its texture must come after `gA`), and an offset for
    // `LayoutResourceKind::RegisterSpace` of one (since the default space will be
    // space zero). It would be incorrect for us to imply that `gS.t` should
    // be `t1, space1`, though, because the space offset of `gS` doesn't actually
    // apply to `t`.
    //
    // For now we are punting on this issue and leaving it in the hands of the
    // application to determine when a space offset from an "outer" variable should
    // apply to the locations of things in an "inner" variable.
    //
    // There is no policy we can apply locally in this function that
    // will Just Work, so the best we can do is try to not lie.

    return space;
}

SLANG_API SlangImageFormat
spReflectionVariableLayout_GetImageFormat(SlangReflectionVariableLayout* inVarLayout)
{
    auto varLayout = convert(inVarLayout);
    if (!varLayout)
        return SLANG_IMAGE_FORMAT_unknown;

    if (auto leafVar = varLayout->getVariable())
    {
        if (auto formatAttrib = leafVar->findModifier<FormatAttribute>())
        {
            return (SlangImageFormat)formatAttrib->format;
        }
    }
    return SLANG_IMAGE_FORMAT_unknown;
}

SLANG_API char const* spReflectionVariableLayout_GetSemanticName(
    SlangReflectionVariableLayout* inVarLayout)
{
    auto varLayout = convert(inVarLayout);
    if (!varLayout)
        return 0;

    if (!(varLayout->flags & Slang::VarLayoutFlag::HasSemantic))
        return 0;

    return varLayout->semanticName.getBuffer();
}

SLANG_API size_t
spReflectionVariableLayout_GetSemanticIndex(SlangReflectionVariableLayout* inVarLayout)
{
    auto varLayout = convert(inVarLayout);
    if (!varLayout)
        return 0;

    if (!(varLayout->flags & Slang::VarLayoutFlag::HasSemantic))
        return 0;

    return varLayout->semanticIndex;
}

SLANG_API SlangStage spReflectionVariableLayout_getStage(SlangReflectionVariableLayout* inVarLayout)
{
    auto varLayout = convert(inVarLayout);
    if (!varLayout)
        return SLANG_STAGE_NONE;

    // A parameter that is not a varying input or output is
    // not considered to belong to a single stage.
    //
    // TODO: We might need to reconsider this for, e.g., entry
    // point parameters, where they might be stage-specific even
    // if they are uniform.
    if (!varLayout->FindResourceInfo(Slang::LayoutResourceKind::VaryingInput) &&
        !varLayout->FindResourceInfo(Slang::LayoutResourceKind::VaryingOutput))
    {
        return SLANG_STAGE_NONE;
    }

    // TODO: We should find the stage for a variable layout by
    // walking up the tree of layout information, until we find
    // something that has a definitive stage attached to it (e.g.,
    // either an entry point or a GLSL translation unit).
    //
    // We don't currently have parent links in the reflection layout
    // information, so doing that walk would be tricky right now, so
    // it is easier to just bloat the representation and store yet another
    // field on every variable layout.
    return (SlangStage)varLayout->stage;
}

// Function Reflection

SLANG_API SlangReflectionDecl* spReflectionFunction_asDecl(SlangReflectionFunction* inFunc)
{
    auto func = convertToFunc(inFunc);
    if (!func)
        return nullptr;

    return (SlangReflectionDecl*)func.getDecl();
}

SLANG_API char const* spReflectionFunction_GetName(SlangReflectionFunction* inFunc)
{
    auto func = convertToFunc(inFunc);
    if (!func)
        return nullptr;

    return getText(func.getDecl()->getName()).getBuffer();
}

SLANG_API SlangReflectionType* spReflectionFunction_GetResultType(SlangReflectionFunction* inFunc)
{
    auto func = convertToFunc(inFunc);
    if (!func)
        return nullptr;

    auto rawType = func.getDecl()->returnType.type;
    auto astBuilder = rawType->getASTBuilderForReflection();

    return convert((Type*)rawType->substitute(astBuilder, SubstitutionSet(func.declRefBase)));
}

SLANG_API SlangReflectionModifier* spReflectionFunction_FindModifier(
    SlangReflectionFunction* inFunc,
    SlangModifierID modifierID)
{
    auto funcDeclRef = convertToFunc(inFunc);
    if (!funcDeclRef)
        return nullptr;

    auto varRefl = convert(funcDeclRef.as<Decl>());
    if (!varRefl)
        return nullptr;

    return spReflectionVariable_FindModifier(varRefl, modifierID);
}

SLANG_API unsigned int spReflectionFunction_GetUserAttributeCount(SlangReflectionFunction* inFunc)
{
    auto func = convertToFunc(inFunc);
    if (!func)
        return 0;

    return getUserAttributeCount(func.getDecl());
}

SLANG_API SlangReflectionUserAttribute* spReflectionFunction_GetUserAttribute(
    SlangReflectionFunction* inFunc,
    unsigned int index)
{
    auto func = convertToFunc(inFunc);
    if (!func)
        return nullptr;
    return getUserAttributeByIndex(func.getDecl(), index);
}

SLANG_API SlangReflectionUserAttribute* spReflectionFunction_FindUserAttributeByName(
    SlangReflectionFunction* inFunc,
    SlangSession* session,
    char const* name)
{
    auto func = convertToFunc(inFunc);
    if (!func)
        return nullptr;

    return findUserAttributeByName(asInternal(session), func.getDecl(), name);
}

SLANG_API unsigned int spReflectionFunction_GetParameterCount(SlangReflectionFunction* inFunc)
{
    auto func = convertToFunc(inFunc);
    if (!func)
        return 0;

    return (unsigned int)func.getDecl()->getParameters().getCount();
}

SLANG_API SlangReflectionVariable* spReflectionFunction_GetParameter(
    SlangReflectionFunction* inFunc,
    unsigned int index)
{
    auto func = convertToFunc(inFunc);
    if (!func)
        return nullptr;

    auto astBuilder = getModule(func.getDecl())->getLinkage()->getASTBuilder();

    return convert(getParameters(astBuilder, func)[index]);
}

SLANG_API SlangReflectionGeneric* spReflectionFunction_GetGenericContainer(
    SlangReflectionFunction* func)
{
    auto declRef = convertToFunc(func);
    if (!declRef)
        return nullptr;

    return convertDeclToGeneric(getInnermostGenericParent(declRef));
}

SLANG_API SlangReflectionFunction* spReflectionFunction_applySpecializations(
    SlangReflectionFunction* func,
    SlangReflectionGeneric* generic)
{
    auto declRef = convertToFunc(func);
    auto genericDeclRef = convertGenericToDeclRef(generic);
    if (!declRef || !genericDeclRef)
        return nullptr;

    auto astBuilder = getModule(declRef.getDecl())->getLinkage()->getASTBuilder();

    auto substDeclRef = substituteDeclRef(SubstitutionSet(genericDeclRef), astBuilder, declRef);
    return convert(substDeclRef.as<FunctionDeclBase>());
}

SLANG_API SlangReflectionFunction* spReflectionFunction_specializeWithArgTypes(
    SlangReflectionFunction* func,
    SlangInt argTypeCount,
    SlangReflectionType* const* argTypes)
{
    Linkage* linkage = nullptr;
    Expr* funcExpr = nullptr;

    if (auto funcDeclRef = convertToFunc(func))
    {
        linkage = getModule(funcDeclRef.getDecl())->getLinkage();
        auto declRefExpr = linkage->getASTBuilder()->create<DeclRefExpr>();
        declRefExpr->declRef = funcDeclRef;
        funcExpr = declRefExpr;
    }
    else if (auto overloadedExpr = convertToOverloadedFunc(func))
    {
        linkage = getModule(overloadedExpr->lookupResult2.items[0].declRef.getDecl())->getLinkage();
        funcExpr = overloadedExpr;
    }
    else
    {
        return nullptr;
    }

    List<Type*> argTypeList;
    for (SlangInt ii = 0; ii < argTypeCount; ++ii)
    {
        auto argType = convert(argTypes[ii]);
        argTypeList.add(argType);
    }

    try
    {
        DiagnosticSink sink(linkage->getSourceManager(), Lexer::sourceLocationLexer);
        auto resultFunc =
            linkage->specializeWithArgTypes(funcExpr, argTypeList, &sink).as<FunctionDeclBase>();

        if (sink.getErrorCount() != 0)
            return nullptr; // Failed coercion.

        return convert(resultFunc);
    }
    catch (...)
    {
        return nullptr;
    }
}

SLANG_API bool spReflectionFunction_isOverloaded(SlangReflectionFunction* func)
{
    return (convertToOverloadedFunc(func) != nullptr);
}

SLANG_API unsigned int spReflectionFunction_getOverloadCount(SlangReflectionFunction* func)
{
    auto overloadedFunc = convertToOverloadedFunc(func);
    if (!overloadedFunc)
        return 1;

    return (unsigned int)overloadedFunc->lookupResult2.items.getCount();
}

SLANG_API SlangReflectionFunction* spReflectionFunction_getOverload(
    SlangReflectionFunction* func,
    unsigned int index)
{
    auto overloadedFunc = convertToOverloadedFunc(func);
    if (!overloadedFunc)
        return nullptr;

    auto declRef = overloadedFunc->lookupResult2.items[index].declRef;
    if (auto funcDeclRef = declRef.as<FunctionDeclBase>())
    {
        return convert(declRef.as<FunctionDeclBase>());
    }
    else if (auto genericDeclRef = declRef.as<GenericDecl>())
    {
        auto astBuilder = getModule(genericDeclRef.getDecl())->getLinkage()->getASTBuilder();
        auto innerDeclRef = substituteDeclRef(
            SubstitutionSet(genericDeclRef),
            astBuilder,
            genericDeclRef.getDecl()->inner);
        return convert(createDefaultSubstitutionsIfNeeded(astBuilder, nullptr, innerDeclRef)
                           .as<FunctionDeclBase>());
    }

    return nullptr;
}

// Abstract decl reflection

SLANG_API unsigned int spReflectionDecl_getChildrenCount(SlangReflectionDecl* parentDecl)
{
    Decl* decl = (Decl*)parentDecl;
    if (as<ContainerDecl>(decl))
    {
        return (unsigned int)as<ContainerDecl>(decl)->members.getCount();
    }

    return 0;
}

SLANG_API SlangReflectionDecl* spReflectionDecl_getChild(
    SlangReflectionDecl* parentDecl,
    unsigned int index)
{
    Decl* decl = (Decl*)parentDecl;
    if (auto containerDecl = as<ContainerDecl>(decl))
    {
        if (containerDecl->members.getCount() > index)
            return (SlangReflectionDecl*)containerDecl->members[index];
    }

    return nullptr;
}

SLANG_API char const* spReflectionDecl_getName(SlangReflectionDecl* decl)
{
    Decl* slangDecl = (Decl*)decl;

    if (auto name = slangDecl->getName())
        return getText(name).getBuffer();

    return nullptr;
}

SLANG_API SlangDeclKind spReflectionDecl_getKind(SlangReflectionDecl* decl)
{
    Decl* slangDecl = (Decl*)decl;
    if (as<StructDecl>(slangDecl))
    {
        return SLANG_DECL_KIND_STRUCT;
    }
    else if (as<VarDeclBase>(slangDecl))
    {
        return SLANG_DECL_KIND_VARIABLE;
    }
    else if (as<GenericDecl>(slangDecl))
    {
        return SLANG_DECL_KIND_GENERIC;
    }
    else if (as<FunctionDeclBase>(slangDecl))
    {
        return SLANG_DECL_KIND_FUNC;
    }
    else if (as<ModuleDecl>(slangDecl))
    {
        return SLANG_DECL_KIND_MODULE;
    }
    else if (as<NamespaceDecl>(slangDecl))
    {
        return SLANG_DECL_KIND_NAMESPACE;
    }
    else
        return SLANG_DECL_KIND_UNSUPPORTED_FOR_REFLECTION;
}

SLANG_API SlangReflectionFunction* spReflectionDecl_castToFunction(SlangReflectionDecl* decl)
{
    Decl* slangDecl = (Decl*)decl;
    if (auto funcDecl = as<FunctionDeclBase>(slangDecl))
    {
        return convert(DeclRef<FunctionDeclBase>(funcDecl->getDefaultDeclRef()));
    }

    // Improper cast
    return nullptr;
}

SLANG_API SlangReflectionVariable* spReflectionDecl_castToVariable(SlangReflectionDecl* decl)
{
    Decl* slangDecl = (Decl*)decl;
    if (auto varDecl = as<VarDeclBase>(slangDecl))
    {
        return convert(DeclRef(varDecl));
    }

    // Improper cast
    return nullptr;
}

SLANG_API SlangReflectionGeneric* spReflectionDecl_castToGeneric(SlangReflectionDecl* decl)
{
    Decl* slangDecl = (Decl*)decl;
    if (auto genericInnerDecl = as<GenericDecl>(slangDecl)->inner)
    {
        return convertDeclToGeneric(genericInnerDecl);
    }

    // Improper cast
    return nullptr;
}

SLANG_API SlangReflectionType* spReflection_getTypeFromDecl(SlangReflectionDecl* decl)
{
    Decl* slangDecl = (Decl*)decl;

    ASTBuilder* builder = getModule(slangDecl)->getLinkage()->getASTBuilder();
    // TODO: create default substitutions
    if (auto type = DeclRefType::create(builder, slangDecl->getDefaultDeclRef()))
    {
        return convert(type);
    }

    // Couldn't create a type from the decl
    return nullptr;
}

SLANG_API SlangReflectionDecl* spReflectionDecl_getParent(SlangReflectionDecl* decl)
{
    Decl* slangDecl = (Decl*)decl;
    if (auto parentDecl = slangDecl->parentDecl)
    {
        return (SlangReflectionDecl*)parentDecl;
    }

    return nullptr;
}

// Generic Reflection

SLANG_API SlangReflectionDecl* spReflectionGeneric_asDecl(SlangReflectionGeneric* generic)
{
    return (SlangReflectionDecl*)convertGenericToDeclRef(generic).getDecl()->parentDecl;
}

SLANG_API char const* spReflectionGeneric_GetName(SlangReflectionGeneric* generic)
{
    auto slangGeneric = convertGenericToDeclRef(generic);
    if (!slangGeneric)
        return nullptr;
    return getText(slangGeneric.getDecl()->getName()).getBuffer();
}

SLANG_API unsigned int spReflectionGeneric_GetTypeParameterCount(SlangReflectionGeneric* generic)
{
    auto slangGeneric = convertGenericToDeclRef(generic);
    if (!slangGeneric)
        return 0;
    auto astBuilder = getModule(slangGeneric.getDecl())->getLinkage()->getASTBuilder();

    return (unsigned int)getMembersOfType<GenericTypeParamDecl>(
               astBuilder,
               slangGeneric.getDecl()->parentDecl)
        .getCount();
}

SLANG_API SlangReflectionVariable* spReflectionGeneric_GetTypeParameter(
    SlangReflectionGeneric* generic,
    unsigned index)
{
    auto slangGeneric = convertGenericToDeclRef(generic);
    if (!slangGeneric)
        return nullptr;
    auto astBuilder = getModule(slangGeneric.getDecl())->getLinkage()->getASTBuilder();

    auto paramDeclRef = getMembersOfType<GenericTypeParamDecl>(
        astBuilder,
        slangGeneric.getDecl()->parentDecl)[index];

    return convert(substituteDeclRef(SubstitutionSet(slangGeneric), astBuilder, paramDeclRef));
}

SLANG_API unsigned int spReflectionGeneric_GetValueParameterCount(SlangReflectionGeneric* generic)
{
    auto slangGeneric = convertGenericToDeclRef(generic);
    if (!slangGeneric)
        return 0;
    auto astBuilder = getModule(slangGeneric.getDecl())->getLinkage()->getASTBuilder();

    return (unsigned int)getMembersOfType<GenericValueParamDecl>(
               astBuilder,
               slangGeneric.getDecl()->parentDecl)
        .getCount();
}

SLANG_API SlangReflectionVariable* spReflectionGeneric_GetValueParameter(
    SlangReflectionGeneric* generic,
    unsigned index)
{
    auto slangGeneric = convertGenericToDeclRef(generic);
    if (!slangGeneric)
        return nullptr;
    auto astBuilder = getModule(slangGeneric.getDecl())->getLinkage()->getASTBuilder();

    auto paramDeclRef = getMembersOfType<GenericValueParamDecl>(
        astBuilder,
        slangGeneric.getDecl()->parentDecl)[index];

    return convert(substituteDeclRef(SubstitutionSet(slangGeneric), astBuilder, paramDeclRef));
}

SLANG_API unsigned int spReflectionGeneric_GetTypeParameterConstraintCount(
    SlangReflectionGeneric* generic,
    SlangReflectionVariable* typeParam)
{
    auto slangGeneric = convertGenericToDeclRef(generic);
    if (!slangGeneric)
        return 0;
    auto astBuilder = getModule(slangGeneric.getDecl())->getLinkage()->getASTBuilder();

    if (auto typeParamDecl = as<GenericTypeParamDecl>(convert(typeParam).getDecl()))
    {
        auto constraints = getCanonicalGenericConstraints(
            astBuilder,
            DeclRef<GenericDecl>(slangGeneric.getDecl()->parentDecl));
        return (unsigned int)(constraints[typeParamDecl]).getValue().getCount();
    }

    return 0;
}

SLANG_API SlangReflectionType* spReflectionGeneric_GetTypeParameterConstraintType(
    SlangReflectionGeneric* generic,
    SlangReflectionVariable* typeParam,
    unsigned index)
{
    auto slangGeneric = convertGenericToDeclRef(generic);
    if (!slangGeneric)
        return nullptr;
    auto astBuilder = getModule(slangGeneric.getDecl())->getLinkage()->getASTBuilder();

    if (auto typeParamDecl = as<GenericTypeParamDecl>(convert(typeParam).getDecl()))
    {
        auto constraints = getCanonicalGenericConstraints(
            astBuilder,
            DeclRef<GenericDecl>(slangGeneric.getDecl()->parentDecl));
        if (auto constraint = (constraints[typeParamDecl]).getValue()[index])
        {
            return convert(substituteType(SubstitutionSet(slangGeneric), astBuilder, constraint));
        }
    }

    return nullptr;
}

SLANG_API SlangDeclKind spReflectionGeneric_GetInnerKind(SlangReflectionGeneric* generic)
{
    auto slangGeneric = convertGenericToDeclRef(generic);
    if (!slangGeneric)
        return SLANG_DECL_KIND_UNSUPPORTED_FOR_REFLECTION;

    return spReflectionDecl_getKind((SlangReflectionDecl*)slangGeneric.getDecl());
}

SLANG_API SlangReflectionDecl* spReflectionGeneric_GetInnerDecl(SlangReflectionGeneric* generic)
{
    auto slangGeneric = convertGenericToDeclRef(generic);
    if (!slangGeneric)
        return nullptr;

    return (SlangReflectionDecl*)slangGeneric.getDecl();
}

SLANG_API SlangReflectionGeneric* spReflectionGeneric_GetOuterGenericContainer(
    SlangReflectionGeneric* generic)
{
    auto declRef = convertGenericToDeclRef(generic);

    auto astBuilder = getModule(declRef.getDecl())->getLinkage()->getASTBuilder();

    return convertDeclToGeneric(getInnermostGenericParent(substituteDeclRef(
        SubstitutionSet(declRef),
        astBuilder,
        createDefaultSubstitutionsIfNeeded(
            astBuilder,
            nullptr,
            DeclRef(declRef.getDecl()->parentDecl)))));
}

SLANG_API SlangReflectionType* spReflectionGeneric_GetConcreteType(
    SlangReflectionGeneric* generic,
    SlangReflectionVariable* typeParam)
{
    auto slangGeneric = convertGenericToDeclRef(generic);
    if (!slangGeneric)
        return nullptr;
    auto astBuilder = getModule(slangGeneric.getDecl())->getLinkage()->getASTBuilder();

    auto genericType = DeclRefType::create(astBuilder, convert(typeParam));

    auto substType = substituteType(SubstitutionSet(slangGeneric), astBuilder, genericType);

    if (genericType != substType)
    {
        return convert(substType);
    }

    return nullptr;
}

SLANG_API int64_t spReflectionGeneric_GetConcreteIntVal(
    SlangReflectionGeneric* generic,
    SlangReflectionVariable* valueParam)
{
    auto slangGeneric = convertGenericToDeclRef(generic);
    if (!slangGeneric)
        return 0;
    auto astBuilder = getModule(slangGeneric.getDecl())->getLinkage()->getASTBuilder();

    auto valueParamDeclRef = convert(valueParam);

    Val* valResult = astBuilder->getOrCreate<GenericParamIntVal>(
        valueParamDeclRef.substitute(
            astBuilder,
            as<GenericValueParamDecl>(valueParamDeclRef.getDecl())->getType()),
        valueParamDeclRef);
    valResult = valResult->substitute(astBuilder, SubstitutionSet(slangGeneric));

    auto intVal = as<ConstantIntVal>(valResult);
    if (intVal)
    {
        return intVal->getValue();
    }

    return 0;
}

SLANG_API SlangReflectionGeneric* spReflectionGeneric_applySpecializations(
    SlangReflectionGeneric* currGeneric,
    SlangReflectionGeneric* generic)
{
    auto declRef = convertGenericToDeclRef(currGeneric);
    auto genericDeclRef = convertGenericToDeclRef(generic);
    if (!declRef || !genericDeclRef)
        return nullptr;

    auto astBuilder = getModule(declRef.getDecl())->getLinkage()->getASTBuilder();

    auto substDeclRef = substituteDeclRef(SubstitutionSet(genericDeclRef), astBuilder, declRef);
    return convertDeclToGeneric(substDeclRef);
}


// Shader Parameter Reflection

SLANG_API unsigned spReflectionParameter_GetBindingIndex(SlangReflectionParameter* inVarLayout)
{
    SlangReflectionVariableLayout* varLayout = (SlangReflectionVariableLayout*)inVarLayout;
    return (unsigned)spReflectionVariableLayout_GetOffset(
        varLayout,
        spReflectionTypeLayout_GetParameterCategory(
            spReflectionVariableLayout_GetTypeLayout(varLayout)));
}

SLANG_API unsigned spReflectionParameter_GetBindingSpace(SlangReflectionParameter* inVarLayout)
{
    SlangReflectionVariableLayout* varLayout = (SlangReflectionVariableLayout*)inVarLayout;
    return (unsigned)spReflectionVariableLayout_GetSpace(
        varLayout,
        spReflectionTypeLayout_GetParameterCategory(
            spReflectionVariableLayout_GetTypeLayout(varLayout)));
}

SLANG_API SlangResult spIsParameterLocationUsed(
    SlangCompileRequest* request,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    SlangParameterCategory category,
    SlangUInt spaceIndex,
    SlangUInt registerIndex,
    bool& outUsed)
{
    if (!request)
        return SLANG_E_INVALID_ARG;

    return request->isParameterLocationUsed(
        entryPointIndex,
        targetIndex,
        category,
        spaceIndex,
        registerIndex,
        outUsed);
}


// Entry Point Reflection

SLANG_API char const* spReflectionEntryPoint_getName(SlangReflectionEntryPoint* inEntryPoint)
{
    auto entryPointLayout = convert(inEntryPoint);
    return entryPointLayout ? getCstr(entryPointLayout->name) : nullptr;
}

SLANG_API char const* spReflectionEntryPoint_getNameOverride(
    SlangReflectionEntryPoint* inEntryPoint)
{
    auto entryPointLayout = convert(inEntryPoint);
    if (entryPointLayout)
    {
        if (entryPointLayout->nameOverride.getLength())
            return entryPointLayout->nameOverride.getBuffer();
        else
            return getCstr(entryPointLayout->name);
    }
    return nullptr;
}

SLANG_API SlangReflectionFunction* spReflectionEntryPoint_getFunction(
    SlangReflectionEntryPoint* inEntryPoint)
{
    auto entryPointLayout = convert(inEntryPoint);
    if (entryPointLayout)
    {
        return convert(entryPointLayout->entryPoint.as<FunctionDeclBase>());
    }
    return nullptr;
}

SLANG_API unsigned spReflectionEntryPoint_getParameterCount(SlangReflectionEntryPoint* inEntryPoint)
{
    auto entryPointLayout = convert(inEntryPoint);
    if (!entryPointLayout)
        return 0;

    return getParameterCount(entryPointLayout->parametersLayout->typeLayout);
}

SLANG_API SlangReflectionVariableLayout* spReflectionEntryPoint_getParameterByIndex(
    SlangReflectionEntryPoint* inEntryPoint,
    unsigned index)
{
    auto entryPointLayout = convert(inEntryPoint);
    if (!entryPointLayout)
        return 0;

    return convert(getParameterByIndex(entryPointLayout->parametersLayout->typeLayout, index));
}

SLANG_API SlangStage spReflectionEntryPoint_getStage(SlangReflectionEntryPoint* inEntryPoint)
{
    auto entryPointLayout = convert(inEntryPoint);

    if (!entryPointLayout)
        return SLANG_STAGE_NONE;

    return SlangStage(entryPointLayout->profile.getStage());
}

SLANG_API void spReflectionEntryPoint_getComputeThreadGroupSize(
    SlangReflectionEntryPoint* inEntryPoint,
    SlangUInt axisCount,
    SlangUInt* outSizeAlongAxis)
{
    auto entryPointLayout = convert(inEntryPoint);

    if (!entryPointLayout)
        return;
    if (!axisCount)
        return;
    if (!outSizeAlongAxis)
        return;

    auto entryPointFunc = entryPointLayout->entryPoint;
    if (!entryPointFunc)
        return;

    SlangUInt sizeAlongAxis[3] = {1, 1, 1};

    // First look for the HLSL case, where we have an attribute attached to the entry point function
    auto numThreadsAttribute = entryPointFunc.getDecl()->findModifier<NumThreadsAttribute>();
    if (numThreadsAttribute)
    {
        for (int i = 0; i < 3; ++i)
        {
            if (auto cint =
                    entryPointLayout->program->tryFoldIntVal(numThreadsAttribute->extents[i]))
                sizeAlongAxis[i] = (SlangUInt)cint->getValue();
            else if (numThreadsAttribute->extents[i])
                sizeAlongAxis[i] = 0;
        }
    }

    //

    if (axisCount > 0)
        outSizeAlongAxis[0] = sizeAlongAxis[0];
    if (axisCount > 1)
        outSizeAlongAxis[1] = sizeAlongAxis[1];
    if (axisCount > 2)
        outSizeAlongAxis[2] = sizeAlongAxis[2];
    for (SlangUInt aa = 3; aa < axisCount; ++aa)
    {
        outSizeAlongAxis[aa] = 1;
    }
}

SLANG_API void spReflectionEntryPoint_getComputeWaveSize(
    SlangReflectionEntryPoint* inEntryPoint,
    SlangUInt* outWaveSize)
{
    auto entryPointLayout = convert(inEntryPoint);

    if (!entryPointLayout)
        return;
    if (!outWaveSize)
        return;

    auto entryPointFunc = entryPointLayout->entryPoint;
    if (!entryPointFunc)
        return;

    // First look for the HLSL case, where we have an attribute attached to the entry point function
    if (auto waveSizeAttribute = entryPointFunc.getDecl()->findModifier<WaveSizeAttribute>())
    {
        if (auto cint = entryPointLayout->program->tryFoldIntVal(waveSizeAttribute->numLanes))
            *outWaveSize = (SlangUInt)cint->getValue();
        else if (waveSizeAttribute->numLanes)
            *outWaveSize = 0;
    }
}

SLANG_API int spReflectionEntryPoint_usesAnySampleRateInput(SlangReflectionEntryPoint* inEntryPoint)
{
    auto entryPointLayout = convert(inEntryPoint);
    if (!entryPointLayout)
        return 0;

    if (entryPointLayout->profile.getStage() != Stage::Fragment)
        return 0;

    return (entryPointLayout->flags & EntryPointLayout::Flag::usesAnySampleRateInput) != 0;
}

SLANG_API SlangReflectionVariableLayout* spReflectionEntryPoint_getVarLayout(
    SlangReflectionEntryPoint* inEntryPoint)
{
    auto entryPointLayout = convert(inEntryPoint);
    if (!entryPointLayout)
        return nullptr;

    return convert(entryPointLayout->parametersLayout);
}

SLANG_API SlangReflectionVariableLayout* spReflectionEntryPoint_getResultVarLayout(
    SlangReflectionEntryPoint* inEntryPoint)
{
    auto entryPointLayout = convert(inEntryPoint);
    if (!entryPointLayout)
        return nullptr;

    return convert(entryPointLayout->resultLayout);
}

SLANG_API int spReflectionEntryPoint_hasDefaultConstantBuffer(
    SlangReflectionEntryPoint* inEntryPoint)
{
    auto entryPointLayout = convert(inEntryPoint);
    if (!entryPointLayout)
        return 0;

    return hasDefaultConstantBuffer(entryPointLayout);
}


// SlangReflectionTypeParameter
SLANG_API char const* spReflectionTypeParameter_GetName(SlangReflectionTypeParameter* inTypeParam)
{
    auto specializationParam = convert(inTypeParam);
    if (auto genericParamLayout = as<GenericSpecializationParamLayout>(specializationParam))
    {
        return genericParamLayout->decl->getName()->text.getBuffer();
    }
    // TODO: Add case for existential type parameter? They don't have as simple of a notion of
    // "name" as the generic case...
    return nullptr;
}

SLANG_API unsigned spReflectionTypeParameter_GetIndex(SlangReflectionTypeParameter* inTypeParam)
{
    auto typeParam = convert(inTypeParam);
    return (unsigned)(typeParam->index);
}

SLANG_API unsigned int spReflectionTypeParameter_GetConstraintCount(
    SlangReflectionTypeParameter* inTypeParam)
{
    auto specializationParam = convert(inTypeParam);
    if (auto genericParamLayout = as<GenericSpecializationParamLayout>(specializationParam))
    {
        if (auto globalGenericParamDecl = as<GlobalGenericParamDecl>(genericParamLayout->decl))
        {
            auto constraints =
                globalGenericParamDecl->getMembersOfType<GenericTypeConstraintDecl>();
            return (unsigned int)constraints.getCount();
        }
        // TODO: Add case for entry-point generic parameters.
    }
    // TODO: Add case for existential type parameters.
    return 0;
}

SLANG_API SlangReflectionType* spReflectionTypeParameter_GetConstraintByIndex(
    SlangReflectionTypeParameter* inTypeParam,
    unsigned index)
{
    auto specializationParam = convert(inTypeParam);
    if (auto genericParamLayout = as<GenericSpecializationParamLayout>(specializationParam))
    {
        if (auto globalGenericParamDecl = as<GlobalGenericParamDecl>(genericParamLayout->decl))
        {
            auto constraints =
                globalGenericParamDecl->getMembersOfType<GenericTypeConstraintDecl>();
            return (SlangReflectionType*)constraints[index]->sup.Ptr();
        }
        // TODO: Add case for entry-point generic parameters.
    }
    // TODO: Add case for existential type parameters.
    return 0;
}

// Shader Reflection

SLANG_API unsigned spReflection_GetParameterCount(SlangReflection* inProgram)
{
    auto program = convert(inProgram);
    if (!program)
        return 0;

    auto globalStructLayout = getGlobalStructLayout(program);
    if (!globalStructLayout)
        return 0;

    return (unsigned)globalStructLayout->fields.getCount();
}

SLANG_API SlangReflectionParameter* spReflection_GetParameterByIndex(
    SlangReflection* inProgram,
    unsigned index)
{
    auto program = convert(inProgram);
    if (!program)
        return nullptr;

    auto globalStructLayout = getGlobalStructLayout(program);
    if (!globalStructLayout)
        return 0;

    return convert(globalStructLayout->fields[index].Ptr());
}

SLANG_API SlangReflectionVariableLayout* spReflection_getGlobalParamsVarLayout(
    SlangReflection* inProgram)
{
    auto program = convert(inProgram);
    if (!program)
        return nullptr;

    return convert(program->parametersLayout);
}

SLANG_API unsigned int spReflection_GetTypeParameterCount(SlangReflection* reflection)
{
    auto program = convert(reflection);
    return (unsigned int)program->specializationParams.getCount();
}

SLANG_API slang::ISession* spReflection_GetSession(SlangReflection* reflection)
{
    auto program = convert(reflection);
    return program->getTargetProgram()->getTargetReq()->getLinkage();
}

SLANG_API SlangReflectionTypeParameter* spReflection_GetTypeParameterByIndex(
    SlangReflection* reflection,
    unsigned int index)
{
    auto program = convert(reflection);
    return (SlangReflectionTypeParameter*)program->specializationParams[index].Ptr();
}

SLANG_API SlangReflectionTypeParameter* spReflection_FindTypeParameter(
    SlangReflection* inProgram,
    char const* name)
{
    auto program = convert(inProgram);
    if (!program)
        return nullptr;
    for (auto& param : program->specializationParams)
    {
        auto genericParamLayout = as<GenericSpecializationParamLayout>(param);
        if (!genericParamLayout)
            continue;

        if (getText(genericParamLayout->decl->getName()) != UnownedTerminatedStringSlice(name))
            continue;

        return (SlangReflectionTypeParameter*)genericParamLayout;
    }

    return 0;
}

SLANG_API SlangUInt spReflection_getEntryPointCount(SlangReflection* inProgram)
{
    auto program = convert(inProgram);
    if (!program)
        return 0;

    return SlangUInt(program->entryPoints.getCount());
}

SLANG_API SlangReflectionEntryPoint* spReflection_getEntryPointByIndex(
    SlangReflection* inProgram,
    SlangUInt index)
{
    auto program = convert(inProgram);
    if (!program)
        return 0;

    return convert(program->entryPoints[(int)index].Ptr());
}

SLANG_API SlangReflectionEntryPoint* spReflection_findEntryPointByName(
    SlangReflection* inProgram,
    char const* name)
{
    auto program = convert(inProgram);
    if (!program)
        return 0;

    // TODO: improve on naive linear search
    for (auto ep : program->entryPoints)
    {
        if (ep->entryPoint.getName()->text == name)
        {
            return convert(ep);
        }
    }

    return nullptr;
}

SLANG_API SlangUInt spReflection_getGlobalConstantBufferBinding(SlangReflection* inProgram)
{
    auto program = convert(inProgram);
    if (!program)
        return 0;
    auto cb = program->parametersLayout->FindResourceInfo(LayoutResourceKind::ConstantBuffer);
    if (!cb)
        return 0;
    return cb->index;
}

SLANG_API size_t spReflection_getGlobalConstantBufferSize(SlangReflection* inProgram)
{
    auto program = convert(inProgram);
    if (!program)
        return 0;
    auto structLayout = getGlobalStructLayout(program);
    auto uniform = structLayout->FindResourceInfo(LayoutResourceKind::Uniform);
    if (!uniform)
        return 0;
    return getReflectionSize(uniform->count);
}

SLANG_API SlangReflectionType* spReflection_specializeType(
    SlangReflection* inProgramLayout,
    SlangReflectionType* inType,
    SlangInt specializationArgCount,
    SlangReflectionType* const* specializationArgs,
    ISlangBlob** outDiagnostics)
{
    auto programLayout = convert(inProgramLayout);
    if (!programLayout)
        return nullptr;

    auto unspecializedType = convert(inType);
    if (!unspecializedType)
        return nullptr;

    auto linkage = programLayout->getProgram()->getLinkage();

    DiagnosticSink sink(linkage->getSourceManager(), Lexer::sourceLocationLexer);

    auto specializedType = linkage->specializeType(
        unspecializedType,
        specializationArgCount,
        (Type* const*)specializationArgs,
        &sink);

    sink.getBlobIfNeeded(outDiagnostics);

    return convert(specializedType);
}


SLANG_API SlangReflectionGeneric* spReflection_specializeGeneric(
    SlangReflection* inProgramLayout,
    SlangReflectionGeneric* generic,
    SlangInt argCount,
    SlangReflectionGenericArgType const* argTypes,
    SlangReflectionGenericArg const* args,
    ISlangBlob** outDiagnostics)
{
    auto programLayout = convert(inProgramLayout);
    auto slangGeneric = convertGenericToDeclRef(generic);
    if (!slangGeneric)
        return nullptr;
    auto astBuilder = getModule(slangGeneric.getDecl())->getLinkage()->getASTBuilder();

    auto linkage = programLayout->getProgram()->getLinkage();

    DiagnosticSink sink(linkage->getSourceManager(), Lexer::sourceLocationLexer);

    List<Expr*> argExprs;
    for (SlangInt i = 0; i < argCount; ++i)
    {
        auto argType = argTypes[i];
        auto arg = args[i];

        switch (argType)
        {
        case SLANG_GENERIC_ARG_TYPE:
            {
                auto type = convert(arg.typeVal);
                auto declRefType = as<DeclRefType>(type);
                auto declRefExpr = astBuilder->create<DeclRefExpr>();
                declRefExpr->declRef = declRefType->getDeclRef();
                declRefExpr->type.type = astBuilder->getOrCreate<TypeType>(type);
                argExprs.add(declRefExpr);
                break;
            }
        case SLANG_GENERIC_ARG_INT:
            {
                auto literalExpr = astBuilder->create<IntegerLiteralExpr>();
                literalExpr->value = args[i].intVal;
                literalExpr->type = astBuilder->getIntType();
                argExprs.add(literalExpr);
                break;
            }
        case SLANG_GENERIC_ARG_BOOL:
            {
                auto literalExpr = astBuilder->create<BoolLiteralExpr>();
                literalExpr->value = args[i].boolVal;
                literalExpr->type = astBuilder->getBoolType();
                argExprs.add(literalExpr);
                break;
            }
        default:
            // abort (TODO: throw a proper error)
            return nullptr;
        }
    }

    auto specialized = linkage->specializeGeneric(slangGeneric, argExprs, &sink);
    sink.getBlobIfNeeded(outDiagnostics);

    return convertDeclToGeneric(specialized);
}


SLANG_API SlangUInt spReflection_getHashedStringCount(SlangReflection* reflection)
{
    auto programLayout = convert(reflection);
    auto slices = programLayout->hashedStringLiteralPool.getAdded();
    return slices.getCount();
}

SLANG_API const char* spReflection_getHashedString(
    SlangReflection* reflection,
    SlangUInt index,
    size_t* outCount)
{
    auto programLayout = convert(reflection);

    auto slices = programLayout->hashedStringLiteralPool.getAdded();
    auto slice = slices[Index(index)];

    *outCount = slice.getLength();
    return slice.begin();
}

SLANG_API SlangUInt32 spComputeStringHash(const char* chars, size_t count)
{
    return SlangUInt32(getStableHashCode32(chars, count));
}

SLANG_API SlangReflectionTypeLayout* spReflection_getGlobalParamsTypeLayout(
    SlangReflection* reflection)
{
    auto programLayout = convert(reflection);
    if (!programLayout)
        return nullptr;

    return convert(programLayout->parametersLayout->typeLayout);
}
