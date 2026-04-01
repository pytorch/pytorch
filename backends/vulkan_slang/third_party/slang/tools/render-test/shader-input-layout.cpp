// Stop warnings from Visual Studio
#define _CRT_SECURE_NO_WARNINGS 1

#include "shader-input-layout.h"

#include "core/slang-token-reader.h"
#include "core/slang-type-text-util.h"

#include <slang-rhi.h>

namespace renderer_test
{
using namespace Slang;

// clang-format off
#define SLANG_SCALAR_TYPES(x) \
    x("int", INT32) \
    x("uint", UINT32) \
    x("float", FLOAT32)
// clang-format on


Format _getFormatFromName(const UnownedStringSlice& slice)
{
    for (int i = 0; i < int(Format::_Count); ++i)
    {
        const FormatInfo& info = getFormatInfo(Format(i));
        if (slice == info.name)
        {
            return Format(i);
        }
    }
    return Format::Undefined;
}

struct TypeInfo
{
    UnownedStringSlice name;
    SlangScalarType type;
};

#define SLANG_SCALAR_TYPE_INFO(name, value) \
    {UnownedStringSlice::fromLiteral(name), SLANG_SCALAR_TYPE_##value},
static const TypeInfo g_scalarTypeInfos[] = {SLANG_SCALAR_TYPES(SLANG_SCALAR_TYPE_INFO)};
#undef SLANG_SCALAR_TYPES
#undef SLANG_SCALAR_TYPE_INFO

static SlangScalarType _getScalarType(const UnownedStringSlice& slice)
{
    for (const auto& info : g_scalarTypeInfos)
    {
        if (info.name == slice)
        {
            return info.type;
        }
    }
    return SLANG_SCALAR_TYPE_NONE;
}

void ShaderInputLayout::AggVal::addField(ShaderInputLayout::Field const& field)
{
    fields.add(field);
}

void ShaderInputLayout::ArrayVal::addField(ShaderInputLayout::Field const& field)
{
    vals.add(field.val);
}

class ShaderInputLayoutFormatException : public Exception
{
public:
    ShaderInputLayoutFormatException(String message)
        : Exception(message)
    {
    }
};

struct ShaderInputLayoutParser
{
    ShaderInputLayout* layout;
    RandomGenerator* rand;

    ShaderInputLayoutParser(ShaderInputLayout* layout, RandomGenerator* rand)
        : layout(layout), rand(rand)
    {
    }

    RefPtr<ShaderInputLayout::ParentVal> parentVal;
    List<RefPtr<ShaderInputLayout::ParentVal>> parentValStack;

    SlangResult parseOption(
        Misc::TokenReader& parser,
        String const& word,
        ShaderInputLayout::TextureVal* val)
    {
        if (word == "depth")
        {
            val->textureDesc.isDepthTexture = true;
        }
        else if (word == "arrayLength")
        {
            parser.Read("=");
            val->textureDesc.arrayLength = parser.ReadInt();
        }
        else if (word == "size")
        {
            parser.Read("=");
            auto size = parser.ReadInt();
            val->textureDesc.size = size;
        }
        else if (word == "content")
        {
            parser.Read("=");
            auto contentWord = parser.ReadWord();
            if (contentWord == "zero")
                val->textureDesc.content = InputTextureContent::Zero;
            else if (contentWord == "one")
                val->textureDesc.content = InputTextureContent::One;
            else if (contentWord == "chessboard")
                val->textureDesc.content = InputTextureContent::ChessBoard;
            else
                val->textureDesc.content = InputTextureContent::Gradient;
        }
        else if (word == "sampleCount")
        {
            parser.Read("=");
            auto contentWord = parser.ReadWord();
            if (contentWord == "one")
                val->textureDesc.sampleCount = InputTextureSampleCount::One;
            else if (contentWord == "two")
                val->textureDesc.sampleCount = InputTextureSampleCount::Two;
            else if (contentWord == "four")
                val->textureDesc.sampleCount = InputTextureSampleCount::Four;
            else if (contentWord == "eight")
                val->textureDesc.sampleCount = InputTextureSampleCount::Eight;
            else if (contentWord == "sixteen")
                val->textureDesc.sampleCount = InputTextureSampleCount::Sixteen;
            else if (contentWord == "thirtyTwo")
                val->textureDesc.sampleCount = InputTextureSampleCount::ThirtyTwo;
            else if (contentWord == "sixtyFour")
                val->textureDesc.sampleCount = InputTextureSampleCount::SixtyFour;
        }
        else if (word == "mipMaps")
        {
            parser.Read("=");
            val->textureDesc.mipMapCount = int(parser.ReadInt());
        }
        else if (word == "format")
        {
            val->textureDesc.format = parseFormatOption(parser);

            if (val->textureDesc.format == Format::Undefined)
            {
                return SLANG_FAIL;
            }
        }
        else
        {
            return SLANG_FAIL;
        }
        return SLANG_OK;
    }

    SlangResult parseOption(
        Misc::TokenReader& parser,
        String const& word,
        ShaderInputLayout::SamplerVal* val)
    {
        if (word == "depthCompare")
        {
            val->samplerDesc.isCompareSampler = true;
        }
        else
        {
            return SLANG_FAIL;
        }
        return SLANG_OK;
    }


    SlangResult parseOption(
        Misc::TokenReader& parser,
        String const& word,
        ShaderInputLayout::CombinedTextureSamplerVal* val)
    {
        auto result = parseOption(parser, word, val->textureVal);
        if (SLANG_SUCCEEDED(result))
            return result;

        result = parseOption(parser, word, val->samplerVal);
        return result;
    }

    SlangResult parseOption(
        Misc::TokenReader& parser,
        String const& word,
        ShaderInputLayout::DataValBase* val)
    {
        if (word == "data")
        {
            parser.Read("=");

            parser.Read("[");
            uint32_t offset = 0;
            while (!parser.IsEnd() && !parser.LookAhead("]"))
            {
                bool negate = false;
                if (parser.NextToken().Type == Misc::TokenType::OpSub)
                {
                    parser.ReadToken();
                    negate = true;
                }

                if (parser.NextToken().Type == Misc::TokenType::IntLiteral)
                {
                    uint32_t value = parser.ReadUInt();
                    if (negate)
                        value = uint32_t(-int32_t(value));
                    val->bufferData.add(value);
                }
                else
                {
                    auto floatNum = parser.ReadFloat();
                    if (negate)
                        floatNum = -floatNum;
                    val->bufferData.add(*(unsigned int*)&floatNum);
                }
                offset += 4;
            }
            parser.Read("]");
        }
        else
        {
            return SLANG_FAIL;
        }
        return SLANG_OK;
    }

    SlangResult parseOption(
        Misc::TokenReader& parser,
        String const& word,
        ShaderInputLayout::BufferVal* val)
    {
        if (word == "stride")
        {
            parser.Read("=");
            val->bufferDesc.stride = parser.ReadInt();
        }
        else if (word == "count")
        {
            parser.Read("=");
            val->bufferDesc.elementCount = parser.ReadInt();
        }
        else if (word == "counter")
        {
            parser.Read("=");
            val->bufferDesc.counter = parser.ReadInt();
        }
        else if (word == "random")
        {
            parser.Read("(");
            // Read the type
            String type = parser.ReadWord();
            SlangScalarType scalarType = _getScalarType(type.getUnownedSlice());
            if (scalarType == SLANG_SCALAR_TYPE_NONE)
            {
                StringBuilder scalarTypeNames;
                for (const auto& info : g_scalarTypeInfos)
                {
                    if (scalarTypeNames.getLength() != 0)
                    {
                        scalarTypeNames << ", ";
                    }
                    scalarTypeNames << info.name;
                }

                throw ShaderInputLayoutFormatException(
                    StringBuilder()
                    << "Expecting " << scalarTypeNames << " " << parser.NextToken().Position.Line);
            }

            parser.Read(",");
            const int size = int(parser.ReadUInt());

            switch (scalarType)
            {
            case SLANG_SCALAR_TYPE_INT32:
                {
                    bool hasRange = false;

                    int32_t minValue = -0x7fffffff - 1;
                    int32_t maxValue = 0x7fffffff;

                    if (parser.LookAhead(","))
                    {
                        hasRange = true;
                        parser.ReadToken();
                        minValue = parser.ReadInt();

                        if (parser.LookAhead(","))
                        {
                            parser.ReadToken();
                            maxValue = parser.ReadInt();
                        }
                    }
                    SLANG_ASSERT(minValue <= maxValue);
                    maxValue = (maxValue >= minValue) ? maxValue : minValue;

                    // Generate the data
                    val->bufferData.setCount(size);

                    int32_t* dst = (int32_t*)val->bufferData.getBuffer();
                    for (int i = 0; i < size; ++i)
                    {
                        dst[i] = hasRange ? rand->nextInt32InRange(minValue, maxValue)
                                          : rand->nextInt32();
                    }
                    break;
                }
            case SLANG_SCALAR_TYPE_UINT32:
                {
                    bool hasRange = false;
                    uint32_t minValue = 0;
                    uint32_t maxValue = 0xffffffff;

                    if (parser.LookAhead(","))
                    {
                        parser.ReadToken();
                        minValue = parser.ReadUInt();

                        hasRange = true;

                        if (parser.LookAhead(","))
                        {
                            parser.ReadToken();
                            maxValue = parser.ReadUInt();
                        }
                    }

                    SLANG_ASSERT(minValue <= maxValue);
                    maxValue = (maxValue >= minValue) ? maxValue : minValue;

                    // Generate the data
                    val->bufferData.setCount(size);

                    uint32_t* dst = (uint32_t*)val->bufferData.getBuffer();
                    for (int i = 0; i < size; ++i)
                    {
                        dst[i] = hasRange ? rand->nextUInt32InRange(minValue, maxValue)
                                          : rand->nextUInt32();
                    }

                    break;
                }
            case SLANG_SCALAR_TYPE_FLOAT32:
                {
                    float minValue = -1.0f;
                    float maxValue = 1.0f;

                    if (parser.LookAhead(","))
                    {
                        parser.ReadToken();
                        minValue = parser.ReadFloat();

                        if (parser.LookAhead(","))
                        {
                            parser.ReadToken();
                            maxValue = parser.ReadFloat();
                        }
                    }

                    SLANG_ASSERT(minValue <= maxValue);
                    maxValue = (maxValue >= minValue) ? maxValue : minValue;

                    // Generate the data
                    val->bufferData.setCount(size);

                    float* dst = (float*)val->bufferData.getBuffer();
                    for (int i = 0; i < size; ++i)
                    {
                        dst[i] = (rand->nextUnitFloat32() * (maxValue - minValue)) + minValue;
                    }
                    break;
                }
            }

            // Read the range

            parser.Read(")");
        }
        else if (word == "format")
        {
            val->bufferDesc.format = parseFormatOption(parser);
        }
        else
        {
            return parseOption(parser, word, static_cast<ShaderInputLayout::DataValBase*>(val));
        }
        return SLANG_OK;
    }

    SlangResult parseOption(
        Misc::TokenReader& parser,
        String const& word,
        ShaderInputLayout::ObjectVal* val)
    {
        if (word == "type")
        {
            parser.Read("=");
            val->typeName = parser.ReadWord();
        }
        else
        {
            return SLANG_FAIL;
        }
        return SLANG_OK;
    }

    Format parseFormatOption(Misc::TokenReader& parser)
    {
        parser.Read("=");
        auto formatWord = parser.ReadWord();

        return _getFormatFromName(formatWord.getUnownedSlice());
    }

    template<typename T>
    void maybeParseOptions(Misc::TokenReader& parser, T* val)
    {
        // parse options
        if (parser.LookAhead("("))
        {
            parser.Read("(");
            while (!parser.IsEnd() && !parser.LookAhead(")"))
            {
                auto word = parser.ReadWord();
                if (SLANG_FAILED(parseOption(parser, word, val)))
                {
                    throw ShaderInputLayoutFormatException(
                        String("Unsupported option '") + word + String("' at line ") +
                        String(parser.NextToken().Position.Line));
                }

                if (parser.LookAhead(","))
                    parser.Read(",");
                else
                    break;
            }
            parser.Read(")");
        }
    }

    RefPtr<ShaderInputLayout::Val> parseNumericValExpr(
        Misc::TokenReader& parser,
        bool negate = false)
    {
        switch (parser.NextToken().Type)
        {
        case Misc::TokenType::IntLiteral:
            {
                RefPtr<ShaderInputLayout::DataVal> val = new ShaderInputLayout::DataVal;

                uint32_t value = parser.ReadUInt();
                if (negate)
                    value = uint32_t(-int32_t(value));
                val->bufferData.add(value);

                return val;
            }
            break;

        case Misc::TokenType::DoubleLiteral:
            {
                RefPtr<ShaderInputLayout::DataVal> val = new ShaderInputLayout::DataVal;

                float floatValue = parser.ReadFloat();
                if (negate)
                    floatValue = -floatValue;

                uint32_t value = 0;
                memcpy(&value, &floatValue, sizeof(floatValue));
                val->bufferData.add(value);

                return val;
            }
            break;

        default:
            throw ShaderInputLayoutFormatException(
                String("Expected a numeric literal but found '") + parser.NextToken().Content +
                String("' at line") + String(parser.NextToken().Position.Line));
        }
    }

    String parseTypeName(Misc::TokenReader& parser)
    {
        String typeName = parser.ReadWord();
        if (parser.AdvanceIf("<"))
        {
            StringBuilder sb;
            sb << typeName << "<";
            for (;;)
            {
                if (parser.LookAhead(Misc::TokenType::IntLiteral))
                    sb << parser.ReadInt();
                else
                    sb << parseTypeName(parser);
                if (!parser.AdvanceIf(","))
                    break;
                sb << ",";
            }
            sb << ">";
            parser.Read(">");
            return sb.produceString();
        }
        return typeName;
    }

    RefPtr<ShaderInputLayout::Val> parseValExpr(Misc::TokenReader& parser)
    {
        typedef Misc::TokenType TokenType;

        switch (parser.NextToken().Type)
        {
        case TokenType::OpSub:
            {
                parser.ReadToken();
                return parseNumericValExpr(parser, true);
            }
            break;

        case TokenType::IntLiteral:
        case TokenType::DoubleLiteral:
            return parseNumericValExpr(parser);

        case TokenType::LBrace:
            {
                // aggregate
                parser.ReadToken();
                RefPtr<ShaderInputLayout::AggVal> val = new ShaderInputLayout::AggVal;

                while (!parser.IsEnd() && !parser.LookAhead(TokenType::RBrace))
                {
                    ShaderInputLayout::Field field;

                    if (parser.LookAhead(TokenType::Identifier) &&
                        parser.NextToken(1).Type == TokenType::Colon)
                    {
                        field.name = parser.ReadWord();
                        parser.Read(TokenType::Colon);
                    }

                    field.val = parseValExpr(parser);

                    val->fields.add(field);

                    if (parser.LookAhead(TokenType::RBrace))
                        break;

                    parser.Read(TokenType::Comma);
                }
                parser.Read(TokenType::RBrace);


                return val;
            }
            break;

        case TokenType::LBracket:
            {
                // array
                parser.ReadToken();
                RefPtr<ShaderInputLayout::ArrayVal> val = new ShaderInputLayout::ArrayVal;

                while (!parser.IsEnd() && !parser.LookAhead(TokenType::RBracket))
                {
                    val->vals.add(parseValExpr(parser));

                    if (parser.LookAhead(TokenType::RBracket))
                        break;

                    parser.Read(TokenType::Comma);
                }
                parser.Read(TokenType::RBracket);

                return val;
            }
            break;

        case TokenType::Identifier:
            {
                if (parser.AdvanceIf("new"))
                {
                    RefPtr<ShaderInputLayout::ObjectVal> val = new ShaderInputLayout::ObjectVal;

                    if (parser.NextToken().Type == TokenType::Identifier)
                    {
                        val->typeName = parseTypeName(parser);
                    }

                    val->contentVal = parseValExpr(parser);
                    return val;
                }
                else if (parser.AdvanceIf("out"))
                {
                    auto val = parseValExpr(parser);
                    val->isOutput = true;
                    return val;
                }
                else if (parser.AdvanceIf("specialize"))
                {
                    RefPtr<ShaderInputLayout::SpecializeVal> val =
                        new ShaderInputLayout::SpecializeVal();

                    parser.Read(Misc::TokenType::LParent);
                    while (!parser.IsEnd() && parser.NextToken().Type != Misc::TokenType::RParent)
                    {
                        val->typeArgs.add(parseTypeName(parser));
                        if (!parser.AdvanceIf(","))
                            break;
                    }
                    parser.Read(Misc::TokenType::RParent);
                    val->contentVal = parseValExpr(parser);
                    return val;
                }
                else if (parser.AdvanceIf("dynamic"))
                {
                    RefPtr<ShaderInputLayout::SpecializeVal> val =
                        new ShaderInputLayout::SpecializeVal();
                    val->typeArgs.add("__Dynamic");
                    val->contentVal = parseValExpr(parser);
                    return val;
                }
                else
                {
                    // We assume that any other word is introducing one of the other
                    // cases for a parse-able value.
                    return parseVal(parser);
                }
            }
            break;

        default:
            throw ShaderInputLayoutFormatException(
                String("Unexpected '") + parser.NextToken().Content + String("' at line") +
                String(parser.NextToken().Position.Line));
        }
    }

    RefPtr<ShaderInputLayout::Val> parseVal(Misc::TokenReader& parser)
    {
        auto word = parser.NextToken().Content;
        if (parser.AdvanceIf("begin_array"))
        {
            RefPtr<ShaderInputLayout::ArrayVal> val = new ShaderInputLayout::ArrayVal;
            pushParentVal(val);
            return val;
        }
        else if (parser.AdvanceIf("begin_object"))
        {
            RefPtr<ShaderInputLayout::ObjectVal> val = new ShaderInputLayout::ObjectVal;
            maybeParseOptions(parser, val.Ptr());

            RefPtr<ShaderInputLayout::AggVal> contentVal = new ShaderInputLayout::AggVal;
            val->contentVal = contentVal;
            pushParentVal(contentVal);

            return val;
        }
        else if (parser.AdvanceIf("uniform"))
        {
            RefPtr<ShaderInputLayout::DataVal> val = new ShaderInputLayout::DataVal;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("cbuffer"))
        {
            // A `cbuffer` is basically just an object where the content of
            // the object is being provided by `uniform` data instead.

            RefPtr<ShaderInputLayout::ObjectVal> objVal = new ShaderInputLayout::ObjectVal;

            RefPtr<ShaderInputLayout::DataVal> dataVal = new ShaderInputLayout::DataVal;
            maybeParseOptions(parser, dataVal.Ptr());

            objVal->contentVal = dataVal;

            return objVal;
        }
        else if (parser.AdvanceIf("ubuffer"))
        {
            RefPtr<ShaderInputLayout::BufferVal> val = new ShaderInputLayout::BufferVal;
            val->bufferDesc.type = InputBufferType::StorageBuffer;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("Texture1D"))
        {
            RefPtr<ShaderInputLayout::TextureVal> val = new ShaderInputLayout::TextureVal;
            val->textureDesc.dimension = 1;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("RWTextureBuffer"))
        {
            RefPtr<ShaderInputLayout::BufferVal> val = new ShaderInputLayout::BufferVal;
            val->bufferDesc.type = InputBufferType::StorageBuffer;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("Texture2D"))
        {
            RefPtr<ShaderInputLayout::TextureVal> val = new ShaderInputLayout::TextureVal;
            val->textureDesc.dimension = 2;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("Texture3D"))
        {
            RefPtr<ShaderInputLayout::TextureVal> val = new ShaderInputLayout::TextureVal;
            val->textureDesc.dimension = 3;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("TextureCube"))
        {
            RefPtr<ShaderInputLayout::TextureVal> val = new ShaderInputLayout::TextureVal;
            val->textureDesc.dimension = 2;
            val->textureDesc.isCube = true;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("RWTexture1D"))
        {
            RefPtr<ShaderInputLayout::TextureVal> val = new ShaderInputLayout::TextureVal;
            val->textureDesc.dimension = 1;
            val->textureDesc.isRWTexture = true;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("RWTexture2D"))
        {
            RefPtr<ShaderInputLayout::TextureVal> val = new ShaderInputLayout::TextureVal;
            val->textureDesc.dimension = 2;
            val->textureDesc.isRWTexture = true;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("RWTexture3D"))
        {
            RefPtr<ShaderInputLayout::TextureVal> val = new ShaderInputLayout::TextureVal;
            val->textureDesc.dimension = 3;
            val->textureDesc.isRWTexture = true;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("RWTextureCube"))
        {
            RefPtr<ShaderInputLayout::TextureVal> val = new ShaderInputLayout::TextureVal;
            val->textureDesc.dimension = 2;
            val->textureDesc.isCube = true;
            val->textureDesc.isRWTexture = true;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("Sampler"))
        {
            RefPtr<ShaderInputLayout::SamplerVal> val = new ShaderInputLayout::SamplerVal;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("TextureSampler1D"))
        {
            RefPtr<ShaderInputLayout::CombinedTextureSamplerVal> val =
                new ShaderInputLayout::CombinedTextureSamplerVal;
            val->textureVal = new ShaderInputLayout::TextureVal;
            val->samplerVal = new ShaderInputLayout::SamplerVal;
            val->textureVal->textureDesc.dimension = 1;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("TextureSampler2D"))
        {
            RefPtr<ShaderInputLayout::CombinedTextureSamplerVal> val =
                new ShaderInputLayout::CombinedTextureSamplerVal;
            val->textureVal = new ShaderInputLayout::TextureVal;
            val->samplerVal = new ShaderInputLayout::SamplerVal;
            val->textureVal->textureDesc.dimension = 2;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("TextureSampler3D"))
        {
            RefPtr<ShaderInputLayout::CombinedTextureSamplerVal> val =
                new ShaderInputLayout::CombinedTextureSamplerVal;
            val->textureVal = new ShaderInputLayout::TextureVal;
            val->samplerVal = new ShaderInputLayout::SamplerVal;
            val->textureVal->textureDesc.dimension = 3;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("TextureSamplerCube"))
        {
            RefPtr<ShaderInputLayout::CombinedTextureSamplerVal> val =
                new ShaderInputLayout::CombinedTextureSamplerVal;
            val->textureVal = new ShaderInputLayout::TextureVal;
            val->samplerVal = new ShaderInputLayout::SamplerVal;
            val->textureVal->textureDesc.dimension = 2;
            val->textureVal->textureDesc.isCube = true;
            maybeParseOptions(parser, val.Ptr());
            return val;
        }
        else if (parser.AdvanceIf("AccelerationStructure"))
        {
            RefPtr<ShaderInputLayout::AccelerationStructureVal> val =
                new ShaderInputLayout::AccelerationStructureVal();
            return val;
        }
        else
        {
            throw ShaderInputLayoutFormatException(
                String("Unknown shader input type '") + word + String("' at line") +
                String(parser.NextToken().Position.Line));
        }
        parser.ReadToken();
        return nullptr;
    }

    String parseName(Misc::TokenReader& parser)
    {
        typedef Misc::Token Token;
        typedef Misc::TokenType TokenType;

        StringBuilder builder;

        Token nameToken = parser.ReadToken();
        if (nameToken.Type != TokenType::Identifier)
        {
            throw ShaderInputLayoutFormatException(
                StringBuilder() << "Invalid input syntax at line "
                                << parser.NextToken().Position.Line);
        }
        builder << nameToken.Content;

        for (;;)
        {
            Token token = parser.NextToken(0);

            if (token.Type == TokenType::LBracket)
            {
                parser.ReadToken();
                int index = parser.ReadInt();
                SLANG_ASSERT(index >= 0);
                parser.ReadMatchingToken(TokenType::RBracket);

                builder << "[" << index << "]";
            }
            else if (token.Type == TokenType::Dot)
            {
                parser.ReadToken();
                Token identifierToken = parser.ReadMatchingToken(TokenType::Identifier);

                builder << "." << identifierToken.Content;
            }
            else
            {
                return builder;
            }
        }
    }

    void parseFieldBindings(Misc::TokenReader& parser, ShaderInputLayout::Field& ioField)
    {
        // parse bindings
        if (parser.LookAhead(":"))
        {
            parser.Read(":");
            while (!parser.IsEnd())
            {
                if (parser.AdvanceIf("out"))
                {
                    ioField.val->isOutput = true;
                }
                else if (parser.AdvanceIf("name"))
                {
                    // Optionally consume '='
                    if (parser.NextToken().Type == Misc::TokenType::OpAssign)
                    {
                        parser.ReadToken();
                    }

                    ioField.name = parseName(parser);
                }
                else
                {
                    fprintf(
                        stderr,
                        "Invalid TEST_INPUT syntax '%s'\n",
                        parser.NextToken().Content.getBuffer());
                    break;
                }

                if (parser.LookAhead(","))
                    parser.Read(",");
            }
        }
    }

    void pushParentVal(ShaderInputLayout::ParentVal* val)
    {
        parentValStack.add(parentVal);
        parentVal = val;
    }

    void parseValEntry(Misc::TokenReader& parser)
    {
        auto parentForNewVal = parentVal;

        ShaderInputLayout::Field field;
        field.val = parseVal(parser);
        parseFieldBindings(parser, field);

        parentForNewVal->addField(field);
    }

    void parseSetEntry(Misc::TokenReader& parser)
    {
        auto parentForNewVal = parentVal;

        ShaderInputLayout::Field field;
        field.name = parseName(parser);
        parser.Read(Misc::TokenType::OpAssign);
        field.val = parseValExpr(parser);

        parentForNewVal->addField(field);
    }

    void parseTypeConformance(Misc::TokenReader& parser)
    {
        ShaderInputLayout::TypeConformanceVal conformance;
        conformance.derivedTypeName = parseTypeName(parser);
        parser.Read(":");
        conformance.baseTypeName = parseTypeName(parser);
        if (parser.AdvanceIf("="))
            conformance.idOverride = parser.ReadInt();
        layout->typeConformances.add(conformance);
    }

    void parseLine(Misc::TokenReader& parser)
    {
        if (parser.LookAhead("entryPointSpecializationArg") || parser.LookAhead("type") ||
            parser.LookAhead("entryPointExistentialType"))
        {
            parser.ReadToken();
            StringBuilder typeExp;
            while (!parser.IsEnd())
                typeExp << parser.ReadToken().Content;
            layout->entryPointSpecializationArgs.add(typeExp);
        }
        else if (
            parser.LookAhead("globalSpecializationArg") || parser.LookAhead("global_type") ||
            parser.LookAhead("globalExistentialType"))
        {
            parser.ReadToken();
            StringBuilder typeExp;
            while (!parser.IsEnd())
                typeExp << parser.ReadToken().Content;
            layout->globalSpecializationArgs.add(typeExp);
        }
        else if (parser.AdvanceIf("render_targets"))
        {
            layout->numRenderTargets = parser.ReadInt();
        }
        else if (parser.AdvanceIf("end"))
        {
            parentVal = parentValStack.getLast();
            parentValStack.removeLast();
        }
        else if (parser.AdvanceIf("set"))
        {
            parseSetEntry(parser);
        }
        else if (parser.AdvanceIf("type_conformance"))
        {
            parseTypeConformance(parser);
        }
        else
        {
            parseValEntry(parser);
        }
    }

    RefPtr<ShaderInputLayout::AggVal> parse(const char* source)
    {
        RefPtr<ShaderInputLayout::AggVal> rootVal = new ShaderInputLayout::AggVal;
        parentVal = rootVal;

        auto lines = Misc::Split(source, '\n');
        for (auto& line : lines)
        {
            if (line.startsWith("//TEST_INPUT:"))
            {
                auto lineContent = line.subString(13, line.getLength() - 13);
                Misc::TokenReader parser(lineContent);
                try
                {
                    parseLine(parser);
                }
                catch (const Misc::TextFormatException&)
                {
                    StringBuilder msg;
                    msg << "Invalid input syntax at line " << parser.NextToken().Position.Line;
                    throw ShaderInputLayoutFormatException(msg);
                }
            }
        }

        // TODO: check that stack has been maintained correctly...

        return rootVal;
    }
};

void ShaderInputLayout::parse(RandomGenerator* rand, const char* source)
{
    rootVal = nullptr;
    globalSpecializationArgs.clear();
    entryPointSpecializationArgs.clear();

    ShaderInputLayoutParser parser(this, rand);
    rootVal = parser.parse(source);
}

/* static */ SlangResult ShaderInputLayout::writeBinding(
    slang::TypeLayoutReflection* typeLayout,
    const void* data,
    size_t sizeInBytes,
    WriterHelper writer)
{
    typedef slang::TypeReflection::ScalarType ScalarType;

    slang::TypeReflection::ScalarType scalarType = slang::TypeReflection::ScalarType::None;

    slang::TypeLayoutReflection* elementTypeLayout = nullptr;

    if (typeLayout)
    {
        switch (typeLayout->getKind())
        {

        // case slang::TypeReflection::Kind::Struct:
        case slang::TypeReflection::Kind::Array:
        case slang::TypeReflection::Kind::Matrix:
        case slang::TypeReflection::Kind::Vector:
            {
                elementTypeLayout = typeLayout->getElementTypeLayout();
                break;
            }
        case slang::TypeReflection::Kind::Scalar:
            {
                elementTypeLayout = typeLayout;
                break;
            }
        case slang::TypeReflection::Kind::Resource:
            {
                elementTypeLayout = typeLayout->getElementTypeLayout();
                break;
            }
        case slang::TypeReflection::Kind::TextureBuffer:
        case slang::TypeReflection::Kind::ShaderStorageBuffer:
            {
                elementTypeLayout = typeLayout->getElementTypeLayout();
                break;
            }
        }
    }

    if (elementTypeLayout)
    {
        scalarType = elementTypeLayout->getScalarType();
    }

    if (scalarType != ScalarType::None && scalarType != ScalarType::Void)
    {
        UnownedStringSlice text = TypeTextUtil::getScalarTypeName(scalarType);
        // Write out the type
        writer.put("type: ");
        writer.put(text);
        writer.put("\n");
    }

    switch (scalarType)
    {
    // TODO(JS):
    // Bool is here, because it's not clear across APIs how bool is laid out in memory
    default:
    case ScalarType::None:
    case ScalarType::Void:
    case ScalarType::Bool:
        {
            auto ptr = (const uint32_t*)data;
            const size_t size = sizeInBytes / sizeof(ptr[0]);
            for (size_t i = 0; i < size; ++i)
            {
                uint32_t v = ptr[i];
                writer.print("%X\n", v);
            }
            break;
        }
    case ScalarType::Float16:
        {
            auto ptr = (const uint16_t*)data;
            const size_t size = sizeInBytes / sizeof(ptr[0]);
            for (size_t i = 0; i < size; ++i)
            {
                const float v = HalfToFloat(ptr[i]);
                writer.print("%f\n", v);
            }
            break;
        }
    case ScalarType::UInt32:
        {
            auto ptr = (const uint32_t*)data;
            const size_t size = sizeInBytes / sizeof(ptr[0]);
            for (size_t i = 0; i < size; ++i)
            {
                uint32_t v = ptr[i];
                writer.print("%u\n", v);
            }
            break;
        }
    case ScalarType::Int32:
        {
            auto ptr = (const int32_t*)data;
            const size_t size = sizeInBytes / sizeof(ptr[0]);
            for (size_t i = 0; i < size; ++i)
            {
                int32_t v = ptr[i];
                writer.print("%i\n", v);
            }
            break;
        }
    case ScalarType::Int64:
        {
            auto ptr = (const int64_t*)data;
            const size_t size = sizeInBytes / sizeof(ptr[0]);
            for (size_t i = 0; i < size; ++i)
            {
                int64_t v = ptr[i];
                writer.print("%" PRId64 "\n", v);
            }
            break;
        }
    case ScalarType::UInt64:
        {
            auto ptr = (const uint64_t*)data;
            const size_t size = sizeInBytes / sizeof(ptr[0]);
            for (size_t i = 0; i < size; ++i)
            {
                uint64_t v = ptr[i];
                writer.print("%" PRIu64 "\n", v);
            }
            break;
        }
    case ScalarType::Float32:
        {
            auto ptr = (const float*)data;
            const size_t size = sizeInBytes / sizeof(ptr[0]);
            for (size_t i = 0; i < size; ++i)
            {
                const float v = ptr[i];
                writer.print("%f\n", v);
            }
            break;
        }
    case ScalarType::Float64:
        {
            auto ptr = (const double*)data;
            const size_t size = sizeInBytes / sizeof(ptr[0]);
            for (size_t i = 0; i < size; ++i)
            {
                const double v = ptr[i];
                writer.print("%f\n", v);
            }
            break;
        }
    }

    return SLANG_OK;
}

void loadDataIntoHalf(uint16_t& out, const uint8_t& in)
{
    out = FloatToHalf((float(in) / 255.0f));
}
void loadDataIntoFloat(float& out, const uint8_t& in)
{
    out = (float(in) / 255.0f);
}
template<typename T>
void loadDataIntoUint(T& out, const uint8_t& in)
{
    out = T(in);
}
template<typename T>
void loadDataIntoInt(T& out, const uint8_t& in)
{
    out = T(in);
}

// T for type to return, F for function pointer to operate on uint8->T
template<typename T, typename F>
void generateTextureDataWithTargetTStorage(
    TextureData& output,
    const InputTextureDesc& desc,
    const rhi::FormatInfo& formatInfo,
    F loadUint8ToT)
{
    // the following function assumes input of 0 or 1 since our testing framework only tests with 0
    // or 1
    TextureData work;
    generateTextureDataRGB8(work, desc);

    output.init(desc.format);

    output.m_textureSize = work.m_textureSize;
    output.m_mipLevels = work.m_mipLevels;
    output.m_arraySize = work.m_arraySize;

    List<TextureData::Slice>& dstSlices = output.m_slices;

    Index numSlices = work.m_slices.getCount();
    dstSlices.setCount(numSlices);

    for (int i = 0; i < numSlices; ++i)
    {
        const TextureData::Slice& srcSlice = work.m_slices[i];

        const Index pixelCount = srcSlice.valuesCount;
        const uint8_t* srcPixels = (const uint8_t*)srcSlice.values;

        T* dstPixels = (T*)output.setSliceCount(i, pixelCount);
        switch (formatInfo.channelCount)
        {
        case 1:
            {
                for (Index j = 0; j < pixelCount; ++j, srcPixels += 4, dstPixels += 1)
                {
                    // Copy out r
                    loadUint8ToT(dstPixels[0], srcPixels[0]);
                }
                break;
            }
        case 2:
            {
                for (Index j = 0; j < pixelCount; ++j, srcPixels += 4, dstPixels += 2)
                {
                    // Copy out rg
                    loadUint8ToT(dstPixels[0], srcPixels[0]);
                    loadUint8ToT(dstPixels[1], srcPixels[1]);
                }
                break;
            }
        case 3:
            {
                for (Index j = 0; j < pixelCount; ++j, srcPixels += 4, dstPixels += 3)
                {
                    // Copy out rgb
                    loadUint8ToT(dstPixels[0], srcPixels[0]);
                    loadUint8ToT(dstPixels[1], srcPixels[1]);
                    loadUint8ToT(dstPixels[2], srcPixels[2]);
                }
                break;
            }
        case 4:
            {
                for (Index j = 0; j < pixelCount; ++j, srcPixels += 4, dstPixels += 4)
                {
                    // Copy out rgba
                    loadUint8ToT(dstPixels[0], srcPixels[0]);
                    loadUint8ToT(dstPixels[1], srcPixels[1]);
                    loadUint8ToT(dstPixels[2], srcPixels[2]);
                    loadUint8ToT(dstPixels[3], srcPixels[3]);
                }
                break;
            }
        }
    }
}
void generateTextureData(TextureData& output, const InputTextureDesc& desc)
{
    const FormatInfo& formatInfo = getFormatInfo(desc.format);

    switch (desc.format)
    {
    case Format::RGBA8Unorm:
        {
            generateTextureDataRGB8(output, desc);
            break;
        }
    case Format::R16Float:
    case Format::RG16Float:
    case Format::RGBA16Float:
        {
            generateTextureDataWithTargetTStorage<uint16_t>(
                output,
                desc,
                formatInfo,
                loadDataIntoHalf);
            break;
        }
    case Format::R64Uint:
        {
            generateTextureDataWithTargetTStorage<uint64_t>(
                output,
                desc,
                formatInfo,
                loadDataIntoUint<uint64_t>);
            break;
        }
    case Format::R32Float:
    case Format::RG32Float:
    case Format::RGB32Float:
    case Format::RGBA32Float:
    case Format::D32Float:
        {
            generateTextureDataWithTargetTStorage<float>(
                output,
                desc,
                formatInfo,
                loadDataIntoFloat);
            break;
        }
    case Format::R32Uint:
    case Format::RG32Uint:
    case Format::RGB32Uint:
    case Format::RGBA32Uint:
        {
            generateTextureDataWithTargetTStorage<uint32_t>(
                output,
                desc,
                formatInfo,
                loadDataIntoUint<uint32_t>);
            break;
        }
    case Format::R16Uint:
    case Format::RG16Uint:
    case Format::RGBA16Uint:
        {
            generateTextureDataWithTargetTStorage<uint16_t>(
                output,
                desc,
                formatInfo,
                loadDataIntoUint<uint16_t>);
            break;
        }
    case Format::R8Uint:
    case Format::RG8Uint:
    case Format::RGBA8Uint:
        {
            generateTextureDataWithTargetTStorage<uint8_t>(
                output,
                desc,
                formatInfo,
                loadDataIntoUint<uint8_t>);
            break;
        }
    case Format::R64Sint:
        {
            generateTextureDataWithTargetTStorage<int64_t>(
                output,
                desc,
                formatInfo,
                loadDataIntoInt<int64_t>);
            break;
        }
    case Format::R32Sint:
    case Format::RG32Sint:
    case Format::RGB32Sint:
    case Format::RGBA32Sint:
        {
            generateTextureDataWithTargetTStorage<int32_t>(
                output,
                desc,
                formatInfo,
                loadDataIntoInt<int32_t>);
            break;
        }
    case Format::R16Sint:
    case Format::RG16Sint:
    case Format::RGBA16Sint:
        {
            generateTextureDataWithTargetTStorage<int16_t>(
                output,
                desc,
                formatInfo,
                loadDataIntoInt<int16_t>);
            break;
        }
    case Format::R8Sint:
    case Format::RG8Sint:
    case Format::RGBA8Sint:
        {
            generateTextureDataWithTargetTStorage<int8_t>(
                output,
                desc,
                formatInfo,
                loadDataIntoInt<int8_t>);
            break;
        }
    default:
        {
            SLANG_ASSERT(!"Unhandled format");
            break;
        }
    }
}

template<typename F>
void _iteratePixels(int dimension, int size, unsigned int* buffer, F f)
{
    if (dimension == 1)
        for (int i = 0; i < size; i++)
            buffer[i] = f(i, 0, 0);
    else if (dimension == 2)
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                buffer[i * size + j] = f(j, i, 0);
    else if (dimension == 3)
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                for (int k = 0; k < size; k++)
                    buffer[i * size * size + j * size + k] = f(k, j, i);
};

void generateTextureDataRGB8(TextureData& output, const InputTextureDesc& inputDesc)
{
    int arrLen = inputDesc.arrayLength;
    if (arrLen == 0)
        arrLen = 1;

    output.init(Format::RGBA8Unorm);

    enum class SimpleScalarType
    {
        kUint,
        kInt,
        kFloat,
    };
    SimpleScalarType type;
    const rhi::FormatInfo& formatInfo = getFormatInfo(inputDesc.format);
    switch (formatInfo.channelType)
    {
    case SLANG_SCALAR_TYPE_UINT64:
    case SLANG_SCALAR_TYPE_UINT32:
    case SLANG_SCALAR_TYPE_UINT16:
    case SLANG_SCALAR_TYPE_UINT8:
        type = SimpleScalarType::kUint;
        break;
    case SLANG_SCALAR_TYPE_INT64:
    case SLANG_SCALAR_TYPE_INT32:
    case SLANG_SCALAR_TYPE_INT16:
    case SLANG_SCALAR_TYPE_INT8:
        type = SimpleScalarType::kInt;
        break;
    case SLANG_SCALAR_TYPE_FLOAT64:
    case SLANG_SCALAR_TYPE_FLOAT32:
    case SLANG_SCALAR_TYPE_FLOAT16:
        type = SimpleScalarType::kFloat;
        break;
    default:
        type = SimpleScalarType::kUint;
        break;
    }
    // List<List<unsigned int>>& dataBuffer = output.dataBuffer;
    int arraySize = arrLen;
    if (inputDesc.isCube)
        arraySize *= 6;
    output.m_arraySize = arraySize;
    output.m_textureSize = inputDesc.size;

    const Index maxMipLevels = Math::Log2Floor(output.m_textureSize) + 1;
    Index mipLevels = (inputDesc.mipMapCount <= 0) ? maxMipLevels : inputDesc.mipMapCount;
    mipLevels = (mipLevels > maxMipLevels) ? maxMipLevels : mipLevels;

    output.m_mipLevels = int(mipLevels);
    output.m_slices.setCount(output.m_mipLevels * output.m_arraySize);

    int slice = 0;
    for (int i = 0; i < arraySize; i++)
    {
        for (int j = 0; j < output.m_mipLevels; j++)
        {
            int size = output.m_textureSize >> j;
            int bufferLen = size;
            if (inputDesc.dimension == 2)
                bufferLen *= size;
            else if (inputDesc.dimension == 3)
                bufferLen *= size * size;

            uint32_t* dst = (uint32_t*)output.setSliceCount(slice, bufferLen);

            if (type == SimpleScalarType::kFloat)
                _iteratePixels(
                    inputDesc.dimension,
                    size,
                    dst,
                    [&](int x, int y, int z) -> unsigned int
                    {
                        if (inputDesc.content == InputTextureContent::Zero)
                        {
                            return 0x0;
                        }
                        else if (inputDesc.content == InputTextureContent::One)
                        {
                            return 0xFFFFFFFF;
                        }
                        else if (inputDesc.content == InputTextureContent::Gradient)
                        {
                            unsigned char r = (unsigned char)(x / (float)(size - 1) * 255.0f);
                            unsigned char g = (unsigned char)(y / (float)(size - 1) * 255.0f);
                            unsigned char b = (unsigned char)(z / (float)(size - 1) * 255.0f);
                            return 0xFF000000 + r + (g << 8) + (b << 16);
                        }
                        else if (inputDesc.content == InputTextureContent::ChessBoard)
                        {
                            unsigned int xSig = x < (size >> 1) ? 1 : 0;
                            unsigned int ySig = y < (size >> 1) ? 1 : 0;
                            unsigned int zSig = z < (size >> 1) ? 1 : 0;
                            auto sig = xSig ^ ySig ^ zSig;
                            if (sig)
                                return 0xFFFFFFFF;
                            else
                                return 0xFF808080;
                        }
                        return 0x0;
                    });
            else if (type == SimpleScalarType::kUint || type == SimpleScalarType::kInt)
                _iteratePixels(
                    inputDesc.dimension,
                    size,
                    dst,
                    [&](int x, int y, int z) -> unsigned int
                    {
                        if (inputDesc.content == InputTextureContent::Zero)
                        {
                            return 0x0;
                        }
                        else if (inputDesc.content == InputTextureContent::One)
                        {
                            return 0x01010101;
                        }
                        else if (inputDesc.content == InputTextureContent::Gradient)
                        {
                            unsigned char r = (unsigned char)(x / (float)(size - 1));
                            unsigned char g = (unsigned char)(y / (float)(size - 1));
                            unsigned char b = (unsigned char)(z / (float)(size - 1));
                            return 0x01000000 + r + (g << 8) + (b << 16);
                        }
                        else if (inputDesc.content == InputTextureContent::ChessBoard)
                        {
                            unsigned int xSig = x < (size >> 1) ? 1 : 0;
                            unsigned int ySig = y < (size >> 1) ? 1 : 0;
                            unsigned int zSig = z < (size >> 1) ? 1 : 0;
                            auto sig = xSig ^ ySig ^ zSig;
                            if (sig)
                                return 0x01010101;
                            else
                                return 0x0;
                        }
                        return 0x0;
                    });
            slice++;
        }
    }
}
} // namespace renderer_test
