
#include "slang-reflection-json.h"

#include "../core/slang-blob.h"
#include "slang-ast-support-types.h"

template<typename T>
struct Range
{
public:
    Range(T begin, T end)
        : m_begin(begin), m_end(end)
    {
    }

    struct Iterator
    {
    public:
        explicit Iterator(T value)
            : m_value(value)
        {
        }

        T operator*() const { return m_value; }
        void operator++() { m_value++; }

        bool operator!=(Iterator const& other) { return m_value != other.m_value; }

    private:
        T m_value;
    };

    Iterator begin() const { return Iterator(m_begin); }
    Iterator end() const { return Iterator(m_end); }

private:
    T m_begin;
    T m_end;
};

template<typename T>
Range<T> makeRange(T begin, T end)
{
    return Range<T>(begin, end);
}

template<typename T>
Range<T> makeRange(T end)
{
    return Range<T>(T(0), end);
}

namespace Slang
{

static void emitReflectionVarInfoJSON(PrettyWriter& writer, slang::VariableReflection* var);
static void emitReflectionTypeLayoutJSON(PrettyWriter& writer, slang::TypeLayoutReflection* type);
static void emitReflectionTypeJSON(PrettyWriter& writer, slang::TypeReflection* type);

static void emitReflectionVarBindingInfoJSON(
    PrettyWriter& writer,
    SlangParameterCategory category,
    SlangUInt index,
    SlangUInt count,
    SlangUInt space = 0)
{
    if (category == SLANG_PARAMETER_CATEGORY_UNIFORM)
    {
        writer << "\"kind\": \"uniform\"";
        writer << ", ";
        writer << "\"offset\": " << index;
        writer << ", ";
        writer << "\"size\": " << count;
    }
    else
    {
        writer << "\"kind\": \"";
        switch (category)
        {
#define CASE(NAME, KIND)                  \
    case SLANG_PARAMETER_CATEGORY_##NAME: \
        writer.write(toSlice(#KIND));     \
        break
            CASE(CONSTANT_BUFFER, constantBuffer);
            CASE(SHADER_RESOURCE, shaderResource);
            CASE(UNORDERED_ACCESS, unorderedAccess);
            CASE(VARYING_INPUT, varyingInput);
            CASE(VARYING_OUTPUT, varyingOutput);
            CASE(SAMPLER_STATE, samplerState);
            CASE(UNIFORM, uniform);
            CASE(PUSH_CONSTANT_BUFFER, pushConstantBuffer);
            CASE(DESCRIPTOR_TABLE_SLOT, descriptorTableSlot);
            CASE(SPECIALIZATION_CONSTANT, specializationConstant);
            CASE(MIXED, mixed);
            CASE(REGISTER_SPACE, registerSpace);
            CASE(SUB_ELEMENT_REGISTER_SPACE, subElementRegisterSpace);
            CASE(GENERIC, generic);
            CASE(METAL_ARGUMENT_BUFFER_ELEMENT, metalArgumentBufferElement);
#undef CASE

        default:
            writer << "unknown";
            assert(!"unhandled case");
            break;
        }
        writer << "\"";
        if (space && category != SLANG_PARAMETER_CATEGORY_REGISTER_SPACE)
        {
            writer << ", ";
            writer << "\"space\": " << space;
        }
        writer << ", ";
        writer << "\"index\": ";
        writer << index;
        if (count != 1)
        {
            writer << ", ";
            writer << "\"count\": ";
            if (count == SLANG_UNBOUNDED_SIZE)
            {
                writer << "\"unbounded\"";
            }
            else
            {
                writer << count;
            }
        }
    }
}

static void emitReflectionVarBindingInfoJSON(
    PrettyWriter& writer,
    slang::VariableLayoutReflection* var,
    SlangCompileRequest* request = nullptr,
    int entryPointIndex = -1)
{
    auto stage = var->getStage();
    if (stage != SLANG_STAGE_NONE)
    {
        writer.maybeComma();
        char const* stageName = "UNKNOWN";
        switch (stage)
        {
        case SLANG_STAGE_VERTEX:
            stageName = "vertex";
            break;
        case SLANG_STAGE_HULL:
            stageName = "hull";
            break;
        case SLANG_STAGE_DOMAIN:
            stageName = "domain";
            break;
        case SLANG_STAGE_GEOMETRY:
            stageName = "geometry";
            break;
        case SLANG_STAGE_FRAGMENT:
            stageName = "fragment";
            break;
        case SLANG_STAGE_COMPUTE:
            stageName = "compute";
            break;

        default:
            break;
        }

        writer << "\"stage\": \"" << stageName << "\"";
    }

    auto typeLayout = var->getTypeLayout();
    auto categoryCount = var->getCategoryCount();

    if (categoryCount)
    {
        writer.maybeComma();
        if (categoryCount != 1)
        {
            writer << "\"bindings\": [\n";
        }
        else
        {
            writer << "\"binding\": ";
        }
        writer.indent();

        for (uint32_t cc = 0; cc < categoryCount; ++cc)
        {
            auto category = SlangParameterCategory(var->getCategoryByIndex(cc));
            auto index = var->getOffset(category);
            auto space = var->getBindingSpace(category);
            auto count = typeLayout->getSize(category);

            // Query the paramater usage for the specified entry point.
            // Note: both `request` and `entryPointIndex` may be invalid here, but that should just
            // make the function return a failure.
            bool used = false;
            bool usedAvailable = spIsParameterLocationUsed(
                                     request,
                                     entryPointIndex,
                                     0,
                                     category,
                                     space,
                                     index,
                                     used) == SLANG_OK;

            if (cc != 0)
                writer << ",\n";

            writer << "{";

            emitReflectionVarBindingInfoJSON(writer, category, index, count, space);

            if (usedAvailable)
            {
                writer << ", \"used\": ";
                writer << used;
            }

            writer << "}";
        }

        writer.dedent();
        if (categoryCount != 1)
        {
            writer << "\n]";
        }
    }

    if (auto semanticName = var->getSemanticName())
    {
        writer.maybeComma();
        writer << "\"semanticName\": ";
        writer.writeEscapedString(UnownedStringSlice(semanticName));

        if (auto semanticIndex = var->getSemanticIndex())
        {
            writer.maybeComma();
            writer << "\"semanticIndex\": " << int(semanticIndex);
        }
    }

    if (auto format = var->getImageFormat())
    {
        writer.maybeComma();
        auto formatName = getImageFormatInfo((Slang::ImageFormat)format).name;
        writer << "\"format\": \"";
        writer << formatName;
        writer << "\"";
    }
}

static void emitReflectionNameInfoJSON(PrettyWriter& writer, char const* name)
{
    // TODO: deal with escaping special characters if/when needed
    writer << "\"name\": ";
    writer.writeEscapedString(UnownedStringSlice(name));
}

static void emitUserAttributes(PrettyWriter& writer, slang::VariableReflection* var);

static void emitReflectionModifierInfoJSON(PrettyWriter& writer, slang::VariableReflection* var)
{
    if (var->findModifier(slang::Modifier::Shared))
    {
        writer.maybeComma();
        writer << "\"shared\": true";
    }

    emitUserAttributes(writer, var);
}

static void emitUserAttributeJSON(PrettyWriter& writer, slang::UserAttribute* userAttribute)
{
    writer << "{\n";
    writer.indent();
    writer << "\"name\": \"";
    writer.write(userAttribute->getName());
    writer << "\",\n";
    writer << "\"arguments\": [\n";
    writer.indent();
    for (unsigned int i = 0; i < userAttribute->getArgumentCount(); i++)
    {
        int intVal;
        float floatVal;
        size_t bufSize = 0;
        if (i > 0)
            writer << ",\n";
        if (SLANG_SUCCEEDED(userAttribute->getArgumentValueInt(i, &intVal)))
        {
            writer << intVal;
        }
        else if (SLANG_SUCCEEDED(userAttribute->getArgumentValueFloat(i, &floatVal)))
        {
            writer << floatVal;
        }
        else if (auto str = userAttribute->getArgumentValueString(i, &bufSize))
        {
            writer.writeEscapedString(UnownedStringSlice(str, bufSize));
        }
        else
            writer << "\"invalid value\"";
    }
    writer.dedent();
    writer << "\n]\n";
    writer.dedent();
    writer << "}";
}

static void emitUserAttributes(PrettyWriter& writer, slang::TypeReflection* type)
{
    auto attribCount = type->getUserAttributeCount();
    if (attribCount)
    {
        writer << ",\n\"userAttribs\": [\n";
        writer.indent();
        for (unsigned int i = 0; i < attribCount; i++)
        {
            if (i > 0)
                writer << ",\n";
            auto attrib = type->getUserAttributeByIndex(i);
            emitUserAttributeJSON(writer, attrib);
        }
        writer.dedent();
        writer << "\n]";
    }
}
static void emitUserAttributes(PrettyWriter& writer, slang::VariableReflection* var)
{
    auto attribCount = var->getUserAttributeCount();
    if (attribCount)
    {
        writer << ",\n\"userAttribs\": [\n";
        writer.indent();
        for (unsigned int i = 0; i < attribCount; i++)
        {
            if (i > 0)
                writer << ",\n";
            auto attrib = var->getUserAttributeByIndex(i);
            emitUserAttributeJSON(writer, attrib);
        }
        writer.dedent();
        writer << "\n]";
    }
}
static void emitUserAttributes(PrettyWriter& writer, slang::FunctionReflection* func)
{
    auto attribCount = func->getUserAttributeCount();
    if (attribCount)
    {
        writer << ",\n\"userAttribs\": [\n";
        writer.indent();
        for (unsigned int i = 0; i < attribCount; i++)
        {
            if (i > 0)
                writer << ",\n";
            auto attrib = func->getUserAttributeByIndex(i);
            emitUserAttributeJSON(writer, attrib);
        }
        writer.dedent();
        writer << "\n]";
    }
}

static void emitReflectionVarLayoutJSON(PrettyWriter& writer, slang::VariableLayoutReflection* var)
{
    writer << "{\n";
    writer.indent();

    CommaTrackerRAII commaTracker(writer);

    if (auto name = var->getName())
    {
        writer.maybeComma();
        emitReflectionNameInfoJSON(writer, name);
    }

    writer.maybeComma();
    writer << "\"type\": ";
    emitReflectionTypeLayoutJSON(writer, var->getTypeLayout());

    emitReflectionModifierInfoJSON(writer, var->getVariable());

    emitReflectionVarBindingInfoJSON(writer, var);

    emitUserAttributes(writer, var->getVariable());
    writer.dedent();
    writer << "\n}";
}

static void emitReflectionScalarTypeInfoJSON(PrettyWriter& writer, SlangScalarType scalarType)
{
    writer << "\"scalarType\": \"";
    switch (scalarType)
    {
    default:
        writer << "unknown";
        assert(!"unhandled case");
        break;
#define CASE(TAG, ID)                                                          \
    case static_cast<SlangScalarType>(slang::TypeReflection::ScalarType::TAG): \
        writer.write(toSlice(#ID));                                            \
        break
        CASE(Void, void);
        CASE(Bool, bool);

        CASE(Int8, int8);
        CASE(UInt8, uint8);
        CASE(Int16, int16);
        CASE(UInt16, uint16);
        CASE(Int32, int32);
        CASE(UInt32, uint32);
        CASE(Int64, int64);
        CASE(UInt64, uint64);

        CASE(Float16, float16);
        CASE(Float32, float32);
        CASE(Float64, float64);
#undef CASE
    }
    writer << "\"";
}

static void emitReflectionResourceTypeBaseInfoJSON(
    PrettyWriter& writer,
    slang::TypeReflection* type)
{
    auto shape = type->getResourceShape();
    auto access = type->getResourceAccess();
    writer.maybeComma();
    writer << "\"kind\": \"resource\"";
    writer.maybeComma();
    writer << "\"baseShape\": \"";
    switch (shape & SLANG_RESOURCE_BASE_SHAPE_MASK)
    {
    default:
        writer << "unknown";
        assert(!"unhandled case");
        break;

#define CASE(SHAPE, NAME)             \
    case SLANG_##SHAPE:               \
        writer.write(toSlice(#NAME)); \
        break
        CASE(TEXTURE_1D, texture1D);
        CASE(TEXTURE_2D, texture2D);
        CASE(TEXTURE_3D, texture3D);
        CASE(TEXTURE_CUBE, textureCube);
        CASE(TEXTURE_BUFFER, textureBuffer);
        CASE(STRUCTURED_BUFFER, structuredBuffer);
        CASE(BYTE_ADDRESS_BUFFER, byteAddressBuffer);
        CASE(ACCELERATION_STRUCTURE, accelerationStructure);
#undef CASE
    }
    writer << "\"";
    if (shape & SLANG_TEXTURE_ARRAY_FLAG)
    {
        writer.maybeComma();
        writer << "\"array\": true";
    }
    if (shape & SLANG_TEXTURE_MULTISAMPLE_FLAG)
    {
        writer.maybeComma();
        writer << "\"multisample\": true";
    }
    if (shape & SLANG_TEXTURE_FEEDBACK_FLAG)
    {
        writer.maybeComma();
        writer << "\"feedback\": true";
    }

    if (access != SLANG_RESOURCE_ACCESS_READ)
    {
        writer.maybeComma();
        writer << "\"access\": \"";
        switch (access)
        {
        default:
            writer << "unknown";
            assert(!"unhandled case");
            break;

        case SLANG_RESOURCE_ACCESS_READ:
            break;
        case SLANG_RESOURCE_ACCESS_WRITE:
            writer << "write";
            break;
        case SLANG_RESOURCE_ACCESS_READ_WRITE:
            writer << "readWrite";
            break;
        case SLANG_RESOURCE_ACCESS_RASTER_ORDERED:
            writer << "rasterOrdered";
            break;
        case SLANG_RESOURCE_ACCESS_APPEND:
            writer << "append";
            break;
        case SLANG_RESOURCE_ACCESS_CONSUME:
            writer << "consume";
            break;
        case SLANG_RESOURCE_ACCESS_FEEDBACK:
            writer << "feedback";
            break;
        }
        writer << "\"";
    }
}


static void emitReflectionTypeInfoJSON(PrettyWriter& writer, slang::TypeReflection* type)
{
    auto kind = type->getKind();
    switch (kind)
    {
    case slang::TypeReflection::Kind::SamplerState:
        writer.maybeComma();
        writer << "\"kind\": \"samplerState\"";
        break;

    case slang::TypeReflection::Kind::Resource:
        {
            emitReflectionResourceTypeBaseInfoJSON(writer, type);

            // TODO: We should really print the result type for all resource
            // types, but current test output depends on the old behavior, so
            // we only add result type output for structured buffers at first.
            //
            auto shape = type->getResourceShape();
            switch (shape & SLANG_RESOURCE_BASE_SHAPE_MASK)
            {
            default:
                break;

            case SLANG_STRUCTURED_BUFFER:
            case SLANG_TEXTURE_1D:
            case SLANG_TEXTURE_2D:
            case SLANG_TEXTURE_3D:
            case SLANG_TEXTURE_CUBE:
                if (auto resultType = type->getResourceResultType())
                {
                    writer.maybeComma();
                    writer << "\"resultType\": ";
                    emitReflectionTypeJSON(writer, resultType);
                }
                break;
            }
        }
        break;

    case slang::TypeReflection::Kind::ConstantBuffer:
        writer.maybeComma();
        writer << "\"kind\": \"constantBuffer\"";
        writer.maybeComma();
        writer << "\"elementType\": ";
        emitReflectionTypeJSON(writer, type->getElementType());
        break;

    case slang::TypeReflection::Kind::ParameterBlock:
        writer.maybeComma();
        writer << "\"kind\": \"parameterBlock\"";
        writer.maybeComma();
        writer << "\"elementType\": ";
        emitReflectionTypeJSON(writer, type->getElementType());
        break;

    case slang::TypeReflection::Kind::TextureBuffer:
        writer.maybeComma();
        writer << "\"kind\": \"textureBuffer\"";
        writer.maybeComma();
        writer << "\"elementType\": ";
        emitReflectionTypeJSON(writer, type->getElementType());
        break;

    case slang::TypeReflection::Kind::ShaderStorageBuffer:
        writer.maybeComma();
        writer << "\"kind\": \"shaderStorageBuffer\"";
        writer.maybeComma();
        writer << "\"elementType\": ";
        emitReflectionTypeJSON(writer, type->getElementType());
        break;

    case slang::TypeReflection::Kind::Scalar:
        writer.maybeComma();
        writer << "\"kind\": \"scalar\"";
        writer.maybeComma();
        emitReflectionScalarTypeInfoJSON(writer, SlangScalarType(type->getScalarType()));
        break;

    case slang::TypeReflection::Kind::Vector:
        writer.maybeComma();
        writer << "\"kind\": \"vector\"";
        writer.maybeComma();
        writer << "\"elementCount\": ";
        writer << int(type->getElementCount());
        writer.maybeComma();
        writer << "\"elementType\": ";
        emitReflectionTypeJSON(writer, type->getElementType());
        break;

    case slang::TypeReflection::Kind::Matrix:
        writer.maybeComma();
        writer << "\"kind\": \"matrix\"";
        writer.maybeComma();
        writer << "\"rowCount\": ";
        writer << type->getRowCount();
        writer.maybeComma();
        writer << "\"columnCount\": ";
        writer << type->getColumnCount();
        writer.maybeComma();
        writer << "\"elementType\": ";
        emitReflectionTypeJSON(writer, type->getElementType());
        break;

    case slang::TypeReflection::Kind::Array:
        {
            auto arrayType = type;
            writer.maybeComma();
            writer << "\"kind\": \"array\"";
            writer.maybeComma();
            writer << "\"elementCount\": ";
            writer << int(arrayType->getElementCount());
            writer.maybeComma();
            writer << "\"elementType\": ";
            emitReflectionTypeJSON(writer, arrayType->getElementType());
        }
        break;
    case slang::TypeReflection::Kind::Pointer:
        {
            auto pointerType = type;
            writer.maybeComma();
            writer << "\"kind\": \"pointer\"";
            writer.maybeComma();
            writer << "\"targetType\": ";
            emitReflectionTypeJSON(writer, pointerType->getElementType());
        }
        break;

    case slang::TypeReflection::Kind::Struct:
        {
            writer.maybeComma();
            writer << "\"kind\": \"struct\"";
            writer.maybeComma();
            writer << "\"fields\": [\n";
            writer.indent();

            auto structType = type;
            auto fieldCount = structType->getFieldCount();
            for (uint32_t ff = 0; ff < fieldCount; ++ff)
            {
                if (ff != 0)
                    writer << ",\n";
                emitReflectionVarInfoJSON(writer, structType->getFieldByIndex(ff));
            }
            writer.dedent();
            writer << "\n]";
        }
        break;

    case slang::TypeReflection::Kind::GenericTypeParameter:
        writer.maybeComma();
        writer << "\"kind\": \"GenericTypeParameter\"";
        writer.maybeComma();
        emitReflectionNameInfoJSON(writer, type->getName());
        break;
    case slang::TypeReflection::Kind::Interface:
        writer.maybeComma();
        writer << "\"kind\": \"Interface\"";
        writer.maybeComma();
        emitReflectionNameInfoJSON(writer, type->getName());
        break;
    case slang::TypeReflection::Kind::Feedback:
        writer.maybeComma();
        writer << "\"kind\": \"Feedback\"";
        writer.maybeComma();
        emitReflectionNameInfoJSON(writer, type->getName());
        break;
    case slang::TypeReflection::Kind::DynamicResource:
        writer.maybeComma();
        writer << "\"kind\": \"DynamicResource\"";
        break;
    default:
        assert(!"unhandled case");
        break;
    }
    emitUserAttributes(writer, type);
}

static void emitReflectionParameterGroupTypeLayoutInfoJSON(
    PrettyWriter& writer,
    slang::TypeLayoutReflection* typeLayout,
    const char* kind)
{
    writer << "\"kind\": \"";
    writer.write(kind);
    writer << "\"";

    writer << ",\n\"elementType\": ";
    emitReflectionTypeLayoutJSON(writer, typeLayout->getElementTypeLayout());

    // Note: There is a subtle detail below when it comes to the
    // container/element variable layouts that get nested inside
    // a parameter group type layout.
    //
    // A top-level parameter group type layout like `ConstantBuffer<Foo>`
    // needs to store both information about the `ConstantBuffer` part of
    // things (e.g., it might consume 1 `binding`), as well as the `Foo`
    // part (e.g., it might consume 4 bytes plus 1 `binding`), and there
    // is offset information for each.
    //
    // The "element" part is easy: it is a variable layout for a variable
    // of type `Foo`. The actual variable will be null, but everything else
    // will be filled in as a client would expect.
    //
    // The "container" part is thornier: what should the type and type
    // layout of the "container" variable be? The obvious answer (which
    // the Slang reflection implementation uses today) is that the type
    // is the type of the parameter group itself (e.g., `ConstantBuffer<Foo>`),
    // and the layout is a dummy `TypeLayout` that just reflects the
    // resource usage of the "container" part of things.
    //
    // That means that at runtime the "container var layout" will have
    // a parameter group type (e.g., `TYPE_KIND_CONSTANT_BUFFER`)
    // but its type layotu will be a base `TypeLayout` and not a
    // `ParameterGroupLayout` (since that would introduce infinite regress).
    //
    // We thus have to guard here against the recursive path where
    // we are emitting reflection info for the "container" part of things.
    //
    // TODO: We should probably

    {
        CommaTrackerRAII commaTracker(writer);

        writer << ",\n\"containerVarLayout\": {\n";
        writer.indent();
        emitReflectionVarBindingInfoJSON(writer, typeLayout->getContainerVarLayout());
        writer.dedent();
        writer << "\n}";
    }

    writer << ",\n\"elementVarLayout\": ";
    emitReflectionVarLayoutJSON(writer, typeLayout->getElementVarLayout());
}

static void emitReflectionTypeLayoutInfoJSON(
    PrettyWriter& writer,
    slang::TypeLayoutReflection* typeLayout)
{
    switch (typeLayout->getKind())
    {
    default:
        emitReflectionTypeInfoJSON(writer, typeLayout->getType());
        break;

    case slang::TypeReflection::Kind::Pointer:
        {
            auto valueTypeLayout = typeLayout->getElementTypeLayout();
            SLANG_ASSERT(valueTypeLayout);

            writer.maybeComma();
            writer << "\"kind\": \"pointer\"";

            writer.maybeComma();
            writer << "\"valueType\": ";

            auto typeName = valueTypeLayout->getType()->getName();

            if (typeName && typeName[0])
            {
                // TODO(JS):
                // We can't emit the type layout, because the type could contain
                // a pointer and we end up in a recursive loop. For now we output the typename.
                writer.writeEscapedString(UnownedStringSlice(typeName));
            }
            else
            {
                // TODO(JS): We will need to generate name that we will associate with this type
                // as it doesn't seem to have one
                writer.writeEscapedString(toSlice("unknown name!"));
                SLANG_ASSERT(!"Doesn't have an associated name");
            }

            /*
            emitReflectionTypeLayoutJSON(
                writer,
                valueTypeLayout); */
        }
        break;
    case slang::TypeReflection::Kind::Array:
        {
            auto arrayTypeLayout = typeLayout;
            auto elementTypeLayout = arrayTypeLayout->getElementTypeLayout();
            writer.maybeComma();
            writer << "\"kind\": \"array\"";

            writer.maybeComma();
            writer << "\"elementCount\": ";
            writer << int(arrayTypeLayout->getElementCount());

            writer.maybeComma();
            writer << "\"elementType\": ";
            emitReflectionTypeLayoutJSON(writer, elementTypeLayout);

            if (arrayTypeLayout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM) != 0)
            {
                writer.maybeComma();
                writer << "\"uniformStride\": ";
                writer << int(arrayTypeLayout->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM));
            }
        }
        break;

    case slang::TypeReflection::Kind::Struct:
        {
            auto structTypeLayout = typeLayout;

            writer.maybeComma();
            writer << "\"kind\": \"struct\"";
            if (auto name = structTypeLayout->getName())
            {
                writer.maybeComma();
                emitReflectionNameInfoJSON(writer, name);
            }
            writer.maybeComma();
            writer << "\"fields\": [\n";
            writer.indent();

            auto fieldCount = structTypeLayout->getFieldCount();
            for (uint32_t ff = 0; ff < fieldCount; ++ff)
            {
                if (ff != 0)
                    writer << ",\n";
                emitReflectionVarLayoutJSON(writer, structTypeLayout->getFieldByIndex(ff));
            }
            writer.dedent();
            writer << "\n]";
            emitUserAttributes(writer, structTypeLayout->getType());
        }
        break;

    case slang::TypeReflection::Kind::ConstantBuffer:
        emitReflectionParameterGroupTypeLayoutInfoJSON(writer, typeLayout, "constantBuffer");
        break;

    case slang::TypeReflection::Kind::ParameterBlock:
        emitReflectionParameterGroupTypeLayoutInfoJSON(writer, typeLayout, "parameterBlock");
        break;

    case slang::TypeReflection::Kind::TextureBuffer:
        emitReflectionParameterGroupTypeLayoutInfoJSON(writer, typeLayout, "textureBuffer");
        break;

    case slang::TypeReflection::Kind::ShaderStorageBuffer:
        writer.maybeComma();
        writer << "\"kind\": \"shaderStorageBuffer\"";

        writer.maybeComma();
        writer << "\"elementType\": ";
        emitReflectionTypeLayoutJSON(writer, typeLayout->getElementTypeLayout());
        break;
    case slang::TypeReflection::Kind::GenericTypeParameter:
        writer.maybeComma();
        writer << "\"kind\": \"GenericTypeParameter\"";

        writer.maybeComma();
        emitReflectionNameInfoJSON(writer, typeLayout->getName());
        break;
    case slang::TypeReflection::Kind::Interface:
        writer.maybeComma();
        writer << "\"kind\": \"Interface\"";

        writer.maybeComma();
        emitReflectionNameInfoJSON(writer, typeLayout->getName());
        break;

    case slang::TypeReflection::Kind::Resource:
        {
            // Some resource types (notably structured buffers)
            // encode layout information for their result/element
            // type, but others don't. We need to check for
            // the relevant cases here.
            //
            auto type = typeLayout->getType();
            auto shape = type->getResourceShape();

            const auto baseType = shape & SLANG_RESOURCE_BASE_SHAPE_MASK;

            if (baseType == SLANG_STRUCTURED_BUFFER)
            {
                emitReflectionResourceTypeBaseInfoJSON(writer, type);

                if (auto resultTypeLayout = typeLayout->getElementTypeLayout())
                {
                    writer.maybeComma();
                    writer << "\"resultType\": ";
                    emitReflectionTypeLayoutJSON(writer, resultTypeLayout);
                }
            }
            else if (shape & SLANG_TEXTURE_FEEDBACK_FLAG)
            {
                emitReflectionResourceTypeBaseInfoJSON(writer, type);

                if (auto resultType = typeLayout->getResourceResultType())
                {
                    writer.maybeComma();
                    writer << "\"resultType\": ";
                    emitReflectionTypeJSON(writer, resultType);
                }
            }
            else
            {
                emitReflectionTypeInfoJSON(writer, type);
            }
        }
        break;
    }
}

static void emitReflectionTypeLayoutJSON(
    PrettyWriter& writer,
    slang::TypeLayoutReflection* typeLayout)
{
    CommaTrackerRAII commaTracker(writer);
    writer << "{\n";
    writer.indent();
    emitReflectionTypeLayoutInfoJSON(writer, typeLayout);
    writer.dedent();
    writer << "\n}";
}

static void emitReflectionTypeJSON(PrettyWriter& writer, slang::TypeReflection* type)
{
    CommaTrackerRAII commaTracker(writer);
    writer << "{\n";
    writer.indent();
    emitReflectionTypeInfoJSON(writer, type);
    writer.dedent();
    writer << "\n}";
}

static void emitReflectionVarInfoJSON(PrettyWriter& writer, slang::VariableReflection* var)
{
    emitReflectionNameInfoJSON(writer, var->getName());

    emitReflectionModifierInfoJSON(writer, var);

    writer << ",\n";
    writer << "\"type\": ";
    emitReflectionTypeJSON(writer, var->getType());
}

static void emitReflectionParamJSON(PrettyWriter& writer, slang::VariableLayoutReflection* param)
{
    // TODO: This function is likely redundant with `emitReflectionVarLayoutJSON`
    // and we should try to collapse them into one.

    writer << "{\n";
    writer.indent();

    CommaTrackerRAII commaTracker(writer);

    if (auto name = param->getName())
    {
        writer.maybeComma();
        emitReflectionNameInfoJSON(writer, name);
    }

    emitReflectionModifierInfoJSON(writer, param->getVariable());

    emitReflectionVarBindingInfoJSON(writer, param);

    writer.maybeComma();
    writer << "\"type\": ";
    emitReflectionTypeLayoutJSON(writer, param->getTypeLayout());

    writer.dedent();
    writer << "\n}";
}


static void emitEntryPointParamJSON(
    PrettyWriter& writer,
    slang::VariableLayoutReflection* param,
    SlangCompileRequest* request,
    int entryPointIndex)
{
    writer << "{\n";
    writer.indent();

    if (auto name = param->getName())
    {
        emitReflectionNameInfoJSON(writer, name);
    }

    emitReflectionVarBindingInfoJSON(writer, param, request, entryPointIndex);

    writer.dedent();
    writer << "\n}";
}


static void emitReflectionTypeParamJSON(
    PrettyWriter& writer,
    slang::TypeParameterReflection* typeParam)
{
    writer << "{\n";
    writer.indent();
    emitReflectionNameInfoJSON(writer, typeParam->getName());
    writer << ",\n";
    writer << "\"constraints\": \n";
    writer << "[\n";
    writer.indent();
    auto constraintCount = typeParam->getConstraintCount();
    for (auto ee : makeRange(constraintCount))
    {
        if (ee != 0)
            writer << ",\n";
        writer << "{\n";
        writer.indent();
        CommaTrackerRAII commaTracker(writer);
        emitReflectionTypeInfoJSON(writer, typeParam->getConstraintByIndex(ee));
        writer.dedent();
        writer << "\n}";
    }
    writer.dedent();
    writer << "\n]";
    writer.dedent();
    writer << "\n}";
}

static void emitReflectionEntryPointJSON(
    PrettyWriter& writer,
    SlangCompileRequest* request,
    slang::ShaderReflection* programReflection,
    int entryPointIndex)
{
    slang::EntryPointReflection* entryPoint =
        programReflection->getEntryPointByIndex(entryPointIndex);

    writer << "{\n";
    writer.indent();

    emitReflectionNameInfoJSON(writer, entryPoint->getName());

    switch (entryPoint->getStage())
    {
    case SLANG_STAGE_VERTEX:
        writer << ",\n\"stage\": \"vertex\"";
        break;
    case SLANG_STAGE_HULL:
        writer << ",\n\"stage\": \"hull\"";
        break;
    case SLANG_STAGE_DOMAIN:
        writer << ",\n\"stage\": \"domain\"";
        break;
    case SLANG_STAGE_GEOMETRY:
        writer << ",\n\"stage\": \"geometry\"";
        break;
    case SLANG_STAGE_FRAGMENT:
        writer << ",\n\"stage\": \"fragment\"";
        break;
    case SLANG_STAGE_COMPUTE:
        writer << ",\n\"stage\": \"compute\"";
        break;
    default:
        break;
    }

    auto entryPointParameterCount = entryPoint->getParameterCount();
    if (entryPointParameterCount)
    {
        writer << ",\n\"parameters\": [\n";
        writer.indent();

        for (auto pp : makeRange(entryPointParameterCount))
        {
            if (pp != 0)
                writer << ",\n";

            auto parameter = entryPoint->getParameterByIndex(pp);
            emitReflectionParamJSON(writer, parameter);
        }

        writer.dedent();
        writer << "\n]";
    }
    if (entryPoint->usesAnySampleRateInput())
    {
        writer << ",\n\"usesAnySampleRateInput\": true";
    }
    if (auto resultVarLayout = entryPoint->getResultVarLayout())
    {
        writer << ",\n\"result\": ";
        emitReflectionParamJSON(writer, resultVarLayout);
    }

    if (entryPoint->getStage() == SLANG_STAGE_COMPUTE)
    {
        SlangUInt threadGroupSize[3];
        entryPoint->getComputeThreadGroupSize(3, threadGroupSize);

        writer << ",\n\"threadGroupSize\": [";
        for (int ii = 0; ii < 3; ++ii)
        {
            if (ii != 0)
                writer << ", ";
            writer << threadGroupSize[ii];
        }
        writer << "]";
    }

    // If code generation has been performed, print out the parameter usage by this entry point.
    if (request && (request->getCompileFlags() & SLANG_COMPILE_FLAG_NO_CODEGEN) == 0)
    {
        writer << ",\n\"bindings\": [\n";
        writer.indent();

        auto programParameterCount = programReflection->getParameterCount();
        for (auto pp : makeRange(programParameterCount))
        {
            if (pp != 0)
                writer << ",\n";

            auto parameter = programReflection->getParameterByIndex(pp);
            emitEntryPointParamJSON(writer, parameter, request, entryPointIndex);
        }

        writer.dedent();
        writer << "\n]";
    }

    emitUserAttributes(writer, entryPoint->getFunction());

    writer.dedent();
    writer << "\n}";
}

static void emitReflectionJSON(
    PrettyWriter& writer,
    SlangCompileRequest* request,
    slang::ShaderReflection* programReflection)
{
    writer << "{\n";
    writer.indent();
    writer << "\"parameters\": [\n";
    writer.indent();

    auto parameterCount = programReflection->getParameterCount();
    for (auto pp : makeRange(parameterCount))
    {
        if (pp != 0)
            writer << ",\n";

        auto parameter = programReflection->getParameterByIndex(pp);
        emitReflectionParamJSON(writer, parameter);
    }

    writer.dedent();
    writer << "\n]";

    auto entryPointCount = programReflection->getEntryPointCount();
    if (entryPointCount)
    {
        writer << ",\n\"entryPoints\": [\n";
        writer.indent();

        for (auto ee : makeRange(entryPointCount))
        {
            if (ee != 0)
                writer << ",\n";

            emitReflectionEntryPointJSON(writer, request, programReflection, (int)ee);
        }

        writer.dedent();
        writer << "\n]";
    }

    auto genParamCount = programReflection->getTypeParameterCount();
    if (genParamCount)
    {
        writer << ",\n\"typeParams\":\n";
        writer << "[\n";
        writer.indent();
        for (auto ee : makeRange(genParamCount))
        {
            if (ee != 0)
                writer << ",\n";

            auto typeParam = programReflection->getTypeParameterByIndex(ee);
            emitReflectionTypeParamJSON(writer, typeParam);
        }
        writer.dedent();
        writer << "\n]";
    }

    {
        SlangUInt count = programReflection->getHashedStringCount();
        if (count)
        {
            writer << ",\n\"hashedStrings\": {\n";
            writer.indent();

            for (SlangUInt i = 0; i < count; ++i)
            {
                if (i)
                {
                    writer << ",\n";
                }

                size_t charsCount;
                const char* chars = programReflection->getHashedString(i, &charsCount);
                const int hash = spComputeStringHash(chars, charsCount);

                writer.writeEscapedString(UnownedStringSlice(chars, charsCount));
                writer << ": ";

                writer << hash;
            }

            writer.dedent();
            writer << "\n}\n";
        }
    }

    writer.dedent();
    writer << "\n}\n";
}

void emitReflectionJSON(
    SlangCompileRequest* request,
    SlangReflection* reflection,
    PrettyWriter& writer)
{
    auto programReflection = (slang::ShaderReflection*)reflection;
    emitReflectionJSON(writer, request, programReflection);
}

} // namespace Slang


extern "C"
{
    SLANG_API SlangResult spReflection_ToJson(
        SlangReflection* reflection,
        SlangCompileRequest* request,
        ISlangBlob** outBlob)
    {
        using namespace Slang;
        PrettyWriter writer;
        emitReflectionJSON(request, reflection, writer);
        *outBlob = StringBlob::moveCreate(writer.getBuilder()).detach();
        return SLANG_OK;
    }
}
