#ifndef SLANG_PRELUDE_CPP_TYPES_H
#define SLANG_PRELUDE_CPP_TYPES_H

#ifdef SLANG_PRELUDE_NAMESPACE
namespace SLANG_PRELUDE_NAMESPACE
{
#endif

#ifndef SLANG_FORCE_INLINE
#define SLANG_FORCE_INLINE inline
#endif

#include "slang-cpp-types-core.h"

typedef Vector<float, 2> float2;
typedef Vector<float, 3> float3;
typedef Vector<float, 4> float4;

typedef Vector<int32_t, 2> int2;
typedef Vector<int32_t, 3> int3;
typedef Vector<int32_t, 4> int4;

typedef Vector<uint32_t, 2> uint2;
typedef Vector<uint32_t, 3> uint3;
typedef Vector<uint32_t, 4> uint4;

// We can just map `NonUniformResourceIndex` type directly to the index type on CPU, as CPU does not
// require any special handling around such accesses.
typedef size_t NonUniformResourceIndex;

// ----------------------------- ResourceType -----------------------------------------

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-structuredbuffer-getdimensions
// Missing  Load(_In_  int  Location, _Out_ uint Status);

template<typename T>
struct RWStructuredBuffer
{
    SLANG_FORCE_INLINE T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    const T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride)
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }

    T* data;
    size_t count;
};

template<typename T>
struct StructuredBuffer
{
    SLANG_FORCE_INLINE const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    const T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride)
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }

    T* data;
    size_t count;
};


template<typename T>
struct RWBuffer
{
    SLANG_FORCE_INLINE T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    const T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outCount) { *outCount = uint32_t(count); }

    T* data;
    size_t count;
};

template<typename T>
struct Buffer
{
    SLANG_FORCE_INLINE const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    const T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outCount) { *outCount = uint32_t(count); }

    T* data;
    size_t count;
};

// Missing  Load(_In_  int  Location, _Out_ uint Status);
struct ByteAddressBuffer
{
    void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }
    uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return *(const T*)(((const char*)data) + index);
    }

    const uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer
// Missing support for Atomic operations
// Missing support for Load with status
struct RWByteAddressBuffer
{
    void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }

    uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return *(const T*)(((const char*)data) + index);
    }

    void Store(size_t index, uint32_t v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        data[index >> 2] = v;
    }
    void Store2(size_t index, uint2 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
    }
    void Store3(size_t index, uint3 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
    }
    void Store4(size_t index, uint4 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
        data[dataIdx + 3] = v.w;
    }
    template<typename T>
    void Store(size_t index, T const& value) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        *(T*)(((char*)data) + index) = value;
    }

    uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

struct ISamplerState;
struct ISamplerComparisonState;

struct SamplerState
{
    ISamplerState* state;
};

struct SamplerComparisonState
{
    ISamplerComparisonState* state;
};

#ifndef SLANG_RESOURCE_SHAPE
#define SLANG_RESOURCE_SHAPE
typedef unsigned int SlangResourceShape;
enum
{
    SLANG_RESOURCE_BASE_SHAPE_MASK = 0x0F,

    SLANG_RESOURCE_NONE = 0x00,

    SLANG_TEXTURE_1D = 0x01,
    SLANG_TEXTURE_2D = 0x02,
    SLANG_TEXTURE_3D = 0x03,
    SLANG_TEXTURE_CUBE = 0x04,
    SLANG_TEXTURE_BUFFER = 0x05,

    SLANG_STRUCTURED_BUFFER = 0x06,
    SLANG_BYTE_ADDRESS_BUFFER = 0x07,
    SLANG_RESOURCE_UNKNOWN = 0x08,
    SLANG_ACCELERATION_STRUCTURE = 0x09,
    SLANG_TEXTURE_SUBPASS = 0x0A,

    SLANG_RESOURCE_EXT_SHAPE_MASK = 0xF0,

    SLANG_TEXTURE_FEEDBACK_FLAG = 0x10,
    SLANG_TEXTURE_ARRAY_FLAG = 0x40,
    SLANG_TEXTURE_MULTISAMPLE_FLAG = 0x80,

    SLANG_TEXTURE_1D_ARRAY = SLANG_TEXTURE_1D | SLANG_TEXTURE_ARRAY_FLAG,
    SLANG_TEXTURE_2D_ARRAY = SLANG_TEXTURE_2D | SLANG_TEXTURE_ARRAY_FLAG,
    SLANG_TEXTURE_CUBE_ARRAY = SLANG_TEXTURE_CUBE | SLANG_TEXTURE_ARRAY_FLAG,

    SLANG_TEXTURE_2D_MULTISAMPLE = SLANG_TEXTURE_2D | SLANG_TEXTURE_MULTISAMPLE_FLAG,
    SLANG_TEXTURE_2D_MULTISAMPLE_ARRAY =
        SLANG_TEXTURE_2D | SLANG_TEXTURE_MULTISAMPLE_FLAG | SLANG_TEXTURE_ARRAY_FLAG,
    SLANG_TEXTURE_SUBPASS_MULTISAMPLE = SLANG_TEXTURE_SUBPASS | SLANG_TEXTURE_MULTISAMPLE_FLAG,
};
#endif

//
struct TextureDimensions
{
    void reset()
    {
        shape = 0;
        width = height = depth = 0;
        numberOfLevels = 0;
        arrayElementCount = 0;
    }
    int getDimSizes(uint32_t outDims[4]) const
    {
        const auto baseShape = (shape & SLANG_RESOURCE_BASE_SHAPE_MASK);
        int count = 0;
        switch (baseShape)
        {
        case SLANG_TEXTURE_1D:
            {
                outDims[count++] = width;
                break;
            }
        case SLANG_TEXTURE_2D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                break;
            }
        case SLANG_TEXTURE_3D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                outDims[count++] = depth;
                break;
            }
        case SLANG_TEXTURE_CUBE:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                outDims[count++] = 6;
                break;
            }
        }

        if (shape & SLANG_TEXTURE_ARRAY_FLAG)
        {
            outDims[count++] = arrayElementCount;
        }
        return count;
    }
    int getMIPDims(int outDims[3]) const
    {
        const auto baseShape = (shape & SLANG_RESOURCE_BASE_SHAPE_MASK);
        int count = 0;
        switch (baseShape)
        {
        case SLANG_TEXTURE_1D:
            {
                outDims[count++] = width;
                break;
            }
        case SLANG_TEXTURE_CUBE:
        case SLANG_TEXTURE_2D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                break;
            }
        case SLANG_TEXTURE_3D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                outDims[count++] = depth;
                break;
            }
        }
        return count;
    }
    int calcMaxMIPLevels() const
    {
        int dims[3];
        const int dimCount = getMIPDims(dims);
        for (int count = 1; true; count++)
        {
            bool allOne = true;
            for (int i = 0; i < dimCount; ++i)
            {
                if (dims[i] > 1)
                {
                    allOne = false;
                    dims[i] >>= 1;
                }
            }
            if (allOne)
            {
                return count;
            }
        }
    }

    uint32_t shape;
    uint32_t width, height, depth;
    uint32_t numberOfLevels;
    uint32_t arrayElementCount; ///< For array types, 0 otherwise
};


// Texture

struct ITexture
{
    virtual TextureDimensions GetDimensions(int mipLevel = -1) = 0;
    virtual void Load(const int32_t* v, void* outData, size_t dataSize) = 0;
    virtual void Sample(
        SamplerState samplerState,
        const float* loc,
        void* outData,
        size_t dataSize) = 0;
    virtual void SampleLevel(
        SamplerState samplerState,
        const float* loc,
        float level,
        void* outData,
        size_t dataSize) = 0;
};

template<typename T>
struct Texture1D
{
    void GetDimensions(uint32_t* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, uint32_t* outWidth, uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(float* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, float* outWidth, float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int2& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, float loc) const
    {
        T out;
        texture->Sample(samplerState, &loc, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, float loc, float level)
    {
        T out;
        texture->SampleLevel(samplerState, &loc, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture2D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float2& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float2& loc, float level)
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture3D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outDepth,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outDepth,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int4& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float3& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float3& loc, float level)
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct TextureCube
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Sample(SamplerState samplerState, const float3& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float3& loc, float level)
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture1DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(float* outWidth, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outElements,
        float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
        *outElements = dims.arrayElementCount;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float2& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float2& loc, float level)
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture2DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(uint32_t* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int4& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float3& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float3& loc, float level)
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct TextureCubeArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(uint32_t* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Sample(SamplerState samplerState, const float4& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float4& loc, float level)
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

/* !!!!!!!!!!!!!!!!!!!!!!!!!!! RWTexture !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

struct IRWTexture : ITexture
{
    /// Get the reference to the element at loc.
    virtual void* refAt(const uint32_t* loc) = 0;
};

template<typename T>
struct RWTexture1D
{
    void GetDimensions(uint32_t* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, uint32_t* outWidth, uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(float* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, float* outWidth, float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(int32_t loc) const
    {
        T out;
        texture->Load(&loc, &out, sizeof(out));
        return out;
    }
    T& operator[](uint32_t loc) { return *(T*)texture->refAt(&loc); }
    IRWTexture* texture;
};

template<typename T>
struct RWTexture2D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int2& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](const uint2& loc) { return *(T*)texture->refAt(&loc.x); }
    IRWTexture* texture;
};

template<typename T>
struct RWTexture3D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outDepth,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outDepth,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](const uint3& loc) { return *(T*)texture->refAt(&loc.x); }
    IRWTexture* texture;
};


template<typename T>
struct RWTexture1DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outElements,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(int2 loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](uint2 loc) { return *(T*)texture->refAt(&loc.x); }

    IRWTexture* texture;
};

template<typename T>
struct RWTexture2DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](const uint3& loc) { return *(T*)texture->refAt(&loc.x); }

    IRWTexture* texture;
};

// FeedbackTexture

struct FeedbackType
{
};
struct SAMPLER_FEEDBACK_MIN_MIP : FeedbackType
{
};
struct SAMPLER_FEEDBACK_MIP_REGION_USED : FeedbackType
{
};

struct IFeedbackTexture
{
    virtual TextureDimensions GetDimensions(int mipLevel = -1) = 0;

    // Note here we pass the optional clamp parameter as a pointer. Passing nullptr means no clamp.
    // This was preferred over having two function definitions, and having to differentiate their
    // names
    virtual void WriteSamplerFeedback(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        const float* clamp = nullptr) = 0;
    virtual void WriteSamplerFeedbackBias(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        float bias,
        const float* clamp = nullptr) = 0;
    virtual void WriteSamplerFeedbackGrad(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        const float* ddx,
        const float* ddy,
        const float* clamp = nullptr) = 0;

    virtual void WriteSamplerFeedbackLevel(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        float lod) = 0;
};

template<typename T>
struct FeedbackTexture2D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    template<typename S>
    void WriteSamplerFeedback(Texture2D<S> tex, SamplerState samp, float2 location, float clamp)
    {
        texture->WriteSamplerFeedback(tex.texture, samp, &location.x, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(
        Texture2D<S> tex,
        SamplerState samp,
        float2 location,
        float bias,
        float clamp)
    {
        texture->WriteSamplerFeedbackBias(tex.texture, samp, &location.x, bias, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2D<S> tex,
        SamplerState samp,
        float2 location,
        float2 ddx,
        float2 ddy,
        float clamp)
    {
        texture->WriteSamplerFeedbackGrad(tex.texture, samp, &location.x, &ddx.x, &ddy.x, &clamp);
    }

    // Level

    template<typename S>
    void WriteSamplerFeedbackLevel(Texture2D<S> tex, SamplerState samp, float2 location, float lod)
    {
        texture->WriteSamplerFeedbackLevel(tex.texture, samp, &location.x, lod);
    }

    // Without Clamp
    template<typename S>
    void WriteSamplerFeedback(Texture2D<S> tex, SamplerState samp, float2 location)
    {
        texture->WriteSamplerFeedback(tex.texture, samp, &location.x);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(Texture2D<S> tex, SamplerState samp, float2 location, float bias)
    {
        texture->WriteSamplerFeedbackBias(tex.texture, samp, &location.x, bias);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2D<S> tex,
        SamplerState samp,
        float2 location,
        float2 ddx,
        float2 ddy)
    {
        texture->WriteSamplerFeedbackGrad(tex.texture, samp, &location.x, &ddx.x, &ddy.x);
    }

    IFeedbackTexture* texture;
};

template<typename T>
struct FeedbackTexture2DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    template<typename S>
    void WriteSamplerFeedback(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float clamp)
    {
        texture->WriteSamplerFeedback(texArray.texture, samp, &location.x, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float bias,
        float clamp)
    {
        texture->WriteSamplerFeedbackBias(texArray.texture, samp, &location.x, bias, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float3 ddx,
        float3 ddy,
        float clamp)
    {
        texture
            ->WriteSamplerFeedbackGrad(texArray.texture, samp, &location.x, &ddx.x, &ddy.x, &clamp);
    }

    // Level
    template<typename S>
    void WriteSamplerFeedbackLevel(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float lod)
    {
        texture->WriteSamplerFeedbackLevel(texArray.texture, samp, &location.x, lod);
    }

    // Without Clamp

    template<typename S>
    void WriteSamplerFeedback(Texture2DArray<S> texArray, SamplerState samp, float3 location)
    {
        texture->WriteSamplerFeedback(texArray.texture, samp, &location.x);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float bias)
    {
        texture->WriteSamplerFeedbackBias(texArray.texture, samp, &location.x, bias);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float3 ddx,
        float3 ddy)
    {
        texture->WriteSamplerFeedbackGrad(texArray.texture, samp, &location.x, &ddx.x, &ddy.x);
    }

    IFeedbackTexture* texture;
};

/* Varying input for Compute */

/* Used when running a single thread */
struct ComputeThreadVaryingInput
{
    uint3 groupID;
    uint3 groupThreadID;
};

struct ComputeVaryingInput
{
    uint3 startGroupID; ///< start groupID
    uint3 endGroupID;   ///< Non inclusive end groupID
};

// The uniformEntryPointParams and uniformState must be set to structures that match layout that the
// kernel expects. This can be determined via reflection for example.

typedef void (*ComputeThreadFunc)(
    ComputeThreadVaryingInput* varyingInput,
    void* uniformEntryPointParams,
    void* uniformState);
typedef void (*ComputeFunc)(
    ComputeVaryingInput* varyingInput,
    void* uniformEntryPointParams,
    void* uniformState);

#ifdef SLANG_PRELUDE_NAMESPACE
}
#endif

#endif
