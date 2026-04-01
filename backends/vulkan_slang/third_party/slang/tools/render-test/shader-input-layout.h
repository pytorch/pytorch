#ifndef SLANG_TEST_SHADER_INPUT_LAYOUT_H
#define SLANG_TEST_SHADER_INPUT_LAYOUT_H

#include "core/slang-basic.h"
#include "core/slang-random-generator.h"
#include "core/slang-writer.h"

#include <slang-rhi.h>

namespace renderer_test
{

using namespace rhi;

enum class ShaderInputType
{
    Buffer,
    Texture,
    Sampler,
    CombinedTextureSampler,
    Array,
    UniformData,
    Object,
    Aggregate,
    Specialize,
    AccelerationStructure,
};

enum class InputTextureContent
{
    Zero,
    One,
    ChessBoard,
    Gradient
};

enum InputTextureSampleCount
{
    One = 1,
    Two = 2,
    Four = 4,
    Eight = 8,
    Sixteen = 16,
    ThirtyTwo = 32,
    SixtyFour = 64,
};
struct InputTextureDesc
{
    int dimension = 2;
    int arrayLength = 0;
    bool isCube = false;
    bool isDepthTexture = false;
    bool isRWTexture = false;
    int size = 4;
    int mipMapCount = 0; ///< 0 means the maximum number of mips will be bound

    InputTextureSampleCount sampleCount = InputTextureSampleCount::One;
    Format format = Format::RGBA8Unorm;

    InputTextureContent content = InputTextureContent::One;
};

enum class InputBufferType
{
    //    ConstantBuffer,
    StorageBuffer,
    //    RootConstantBuffer,
};

struct InputBufferDesc
{
    InputBufferType type = InputBufferType::StorageBuffer;
    int stride = 0; // stride == 0 indicates an unstructured buffer.
    int elementCount = 1;
    Format format = Format::Undefined;
    // For RWStructuredBuffer, AppendStructuredBuffer, ConsumeStructuredBuffer
    // the default value of 0xffffffff indicates that a counter buffer should
    // not be assigned
    uint32_t counter = ~0u;
};

struct InputSamplerDesc
{
    bool isCompareSampler = false;
};

struct TextureData
{
    struct Slice
    {
        static Slice make(void* values, size_t size)
        {
            Slice slice;
            slice.values = values;
            slice.valuesCount = size;
            return slice;
        }

        void* values = nullptr; ///< Values of the type format
        size_t valuesCount = 0;
    };

    void addSlice(const void* data, size_t elemCount)
    {
        const size_t totalSize = m_formatSize * elemCount;
        void* dst = ::malloc(totalSize);
        ::memcpy(dst, data, totalSize);
        m_slices.add(Slice::make(dst, elemCount));
    }
    void* addSlice(size_t elemCount)
    {
        void* dst = ::malloc(m_formatSize * elemCount);
        m_slices.add(Slice::make(dst, elemCount));
        return dst;
    }

    /// Set the size of the slice in count of format sized elements
    void* setSliceCount(Slang::Index sliceIndex, size_t count)
    {
        auto& slice = m_slices[sliceIndex];
        if (count != slice.valuesCount)
        {
            slice.values = ::realloc(slice.values, count * m_formatSize);
            slice.valuesCount = count;
        }
        return slice.values;
    }

    void init(Format format)
    {
        clearSlices();

        const FormatInfo& formatInfo = getFormatInfo(format);
        m_formatSize = uint8_t(formatInfo.blockSizeInBytes / formatInfo.pixelsPerBlock);
        m_format = format;
    }

    ~TextureData() { clearSlices(); }

    void clearSlices()
    {
        for (auto& slice : m_slices)
        {
            if (slice.values)
            {
                ::free(slice.values);
            }
        }
        m_slices.clear();
    }

    rhi::Format m_format = rhi::Format::Undefined;
    uint8_t m_formatSize = 0;

    Slang::List<Slice> m_slices;
    int m_textureSize;
    int m_mipLevels;
    int m_arraySize;
};

class ShaderInputLayout
{
public:
    class Val : public Slang::RefObject
    {
    public:
        Val(ShaderInputType kind)
            : kind(kind)
        {
        }

        ShaderInputType kind;
        bool isOutput = false;
    };
    typedef Slang::RefPtr<Val> ValPtr;

    class TextureVal : public Val
    {
    public:
        TextureVal()
            : Val(ShaderInputType::Texture)
        {
        }

        InputTextureDesc textureDesc;
    };

    class DataValBase : public Val
    {
    public:
        DataValBase(ShaderInputType kind)
            : Val(kind)
        {
        }

        Slang::List<unsigned int> bufferData;
    };

    class BufferVal : public DataValBase
    {
    public:
        BufferVal()
            : DataValBase(ShaderInputType::Buffer)
        {
        }

        InputBufferDesc bufferDesc;
    };

    class DataVal : public DataValBase
    {
    public:
        DataVal()
            : DataValBase(ShaderInputType::UniformData)
        {
        }
    };

    class SamplerVal : public Val
    {
    public:
        SamplerVal()
            : Val(ShaderInputType::Sampler)
        {
        }

        InputSamplerDesc samplerDesc;
    };

    class CombinedTextureSamplerVal : public Val
    {
    public:
        CombinedTextureSamplerVal()
            : Val(ShaderInputType::CombinedTextureSampler)
        {
        }

        Slang::RefPtr<TextureVal> textureVal;
        Slang::RefPtr<SamplerVal> samplerVal;
    };

    class AccelerationStructureVal : public Val
    {
    public:
        AccelerationStructureVal()
            : Val(ShaderInputType::AccelerationStructure)
        {
        }
    };

    struct Field
    {
        Slang::String name;
        ValPtr val;
    };
    typedef Field Entry;

    class ParentVal : public Val
    {
    public:
        ParentVal(ShaderInputType kind)
            : Val(kind)
        {
        }

        virtual void addField(Field const& field) = 0;
    };

    class AggVal : public ParentVal
    {
    public:
        AggVal(ShaderInputType kind = ShaderInputType::Aggregate)
            : ParentVal(kind)
        {
        }

        Slang::List<Field> fields;

        virtual void addField(Field const& field) override;
    };

    class ObjectVal : public Val
    {
    public:
        ObjectVal()
            : Val(ShaderInputType::Object)
        {
        }

        Slang::String typeName;
        ValPtr contentVal;
    };

    class SpecializeVal : public Val
    {
    public:
        ValPtr contentVal;
        Slang::List<Slang::String> typeArgs;
        SpecializeVal()
            : Val(ShaderInputType::Specialize)
        {
        }
    };

    class ArrayVal : public ParentVal
    {
    public:
        ArrayVal()
            : ParentVal(ShaderInputType::Array)
        {
        }

        Slang::List<ValPtr> vals;

        virtual void addField(Field const& field) override;
    };

    Slang::RefPtr<AggVal> rootVal;
    Slang::List<Slang::String> globalSpecializationArgs;
    Slang::List<Slang::String> entryPointSpecializationArgs;

    class TypeConformanceVal
    {
    public:
        Slang::String derivedTypeName;
        Slang::String baseTypeName;
        Slang::Int idOverride = -1;
    };
    Slang::List<TypeConformanceVal> typeConformances;

    int numRenderTargets = 1;

    Slang::Index findEntryIndexByName(const Slang::String& name) const;

    void parse(Slang::RandomGenerator* rand, const char* source);

    /// Writes a binding, if bindRoot is set, will try to honor the underlying type when outputting.
    /// If not will dump as uint32_t hex.
    static SlangResult writeBinding(
        slang::TypeLayoutReflection* typeLayout,
        const void* data,
        size_t sizeInBytes,
        Slang::WriterHelper writer);
};

void generateTextureDataRGB8(TextureData& output, const InputTextureDesc& desc);
void generateTextureData(TextureData& output, const InputTextureDesc& desc);


} // namespace renderer_test

#endif
