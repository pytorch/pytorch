// render.h
#pragma once

#include "slang-com-ptr.h"
#include "slang.h"

#include <assert.h>
#include <float.h>


#if defined(SLANG_GFX_DYNAMIC)
    #if defined(_MSC_VER)
        #ifdef SLANG_GFX_DYNAMIC_EXPORT
            #define SLANG_GFX_API SLANG_DLL_EXPORT
        #else
            #define SLANG_GFX_API __declspec(dllimport)
        #endif
    #else
        // TODO: need to consider compiler capabilities
        // #     ifdef SLANG_DYNAMIC_EXPORT
        #define SLANG_GFX_API SLANG_DLL_EXPORT
    // #     endif
    #endif
#endif

#ifndef SLANG_GFX_API
    #define SLANG_GFX_API
#endif

// Needed for building on cygwin with gcc
#undef Always
#undef None

// GLOBAL TODO: doc comments
// GLOBAL TODO: Rationalize integer types (not a smush of uint/int/Uint/Int/etc)
//    - need typedefs in gfx namespace for Count, Index, Size, Offset (ex. DeviceAddress)
//    - Index and Count are for arrays, and indexing into array - like things(XY coordinates of
//    pixels, etc.)
//         - Count is also for anything where we need to measure how many of something there are.
//         This includes things like extents.
//    - Offset and Size are almost always for bytes and things measured in bytes.
namespace gfx
{

using Slang::ComPtr;

typedef SlangResult Result;

// Had to move here, because Options needs types defined here
typedef SlangInt Int;
typedef SlangUInt UInt;
typedef uint64_t DeviceAddress;
typedef int GfxIndex;
typedef int GfxCount;
typedef size_t Size;
typedef size_t Offset;

const uint64_t kTimeoutInfinite = 0xFFFFFFFFFFFFFFFF;

enum class StructType
{
    D3D12DeviceExtendedDesc,
    D3D12ExperimentalFeaturesDesc,
    SlangSessionExtendedDesc,
    RayTracingValidationDesc
};

// TODO: Rename to Stage
enum class StageType
{
    Unknown,
    Vertex,
    Hull,
    Domain,
    Geometry,
    Fragment,
    Compute,
    RayGeneration,
    Intersection,
    AnyHit,
    ClosestHit,
    Miss,
    Callable,
    Amplification,
    Mesh,
    CountOf,
};

// TODO: Implementation or backend or something else?
enum class DeviceType
{
    Unknown,
    Default,
    DirectX11,
    DirectX12,
    OpenGl,
    Vulkan,
    Metal,
    CPU,
    CUDA,
    WebGPU,
    CountOf,
};

// TODO: Why does this exist it should go poof
enum class ProjectionStyle
{
    Unknown,
    OpenGl,
    DirectX,
    Vulkan,
    Metal,
    CountOf,
};

// TODO: This should also go poof
/// The style of the binding
enum class BindingStyle
{
    Unknown,
    DirectX,
    OpenGl,
    Vulkan,
    Metal,
    CPU,
    CUDA,
    CountOf,
};

// TODO: Is this actually a flag when there are no bit fields?
enum class AccessFlag
{
    None,
    Read,
    Write,
};

// TODO: Needed? Shouldn't be hard-coded if so
const GfxCount kMaxRenderTargetCount = 8;

class ITransientResourceHeap;

enum class ShaderModuleSourceType
{
    SlangSource,           // a slang source string in memory.
    SlangModuleBinary,     // a slang module binary code in memory.
    SlangSourceFile,       // a slang source from file.
    SlangModuleBinaryFile, // a slang module binary code from file.
};

class IShaderProgram : public ISlangUnknown
{
public:
    // Defines how linking should be performed for a shader program.
    enum class LinkingStyle
    {
        // Compose all entry-points in a single program, then compile all entry-points together with
        // the same set of root shader arguments.
        SingleProgram,

        // Link and compile each entry-point individually, potentially with different
        // specializations.
        SeparateEntryPointCompilation
    };

    enum class DownstreamLinkMode
    {
        None,
        Deferred,
    };

    struct Desc
    {
        // TODO: Tess doesn't like this but doesn't know what to do about it
        // The linking style of this program.
        LinkingStyle linkingStyle = LinkingStyle::SingleProgram;

        // The global scope or a Slang composite component that represents the entire program.
        slang::IComponentType* slangGlobalScope;

        // Number of separate entry point components in the `slangEntryPoints` array to link in.
        // If set to 0, then `slangGlobalScope` must contain Slang EntryPoint components.
        // If not 0, then `slangGlobalScope` must not contain any EntryPoint components.
        GfxCount entryPointCount = 0;

        // An array of Slang entry points. The size of the array must be `entryPointCount`.
        // Each element must define only 1 Slang EntryPoint.
        slang::IComponentType** slangEntryPoints = nullptr;

        // Indicates whether the app is responsible for final downstream linking.
        DownstreamLinkMode downstreamLinkMode = DownstreamLinkMode::None;
    };

    struct CreateDesc2
    {
        ShaderModuleSourceType sourceType;
        void* sourceData;
        Size sourceDataSize;

        // Number of entry points to include in the shader program. 0 means include all entry points
        // defined in the module.
        GfxCount entryPointCount = 0;
        // Names of entry points to include in the shader program. The size of the array must be
        // `entryPointCount`.
        const char** entryPointNames = nullptr;
    };

    virtual SLANG_NO_THROW slang::TypeReflection* SLANG_MCALL findTypeByName(const char* name) = 0;
};
#define SLANG_UUID_IShaderProgram                          \
    {                                                      \
        0x9d32d0ad, 0x915c, 0x4ffd,                        \
        {                                                  \
            0x91, 0xe2, 0x50, 0x85, 0x54, 0xa0, 0x4a, 0x76 \
        }                                                  \
    }

// TODO: Confirm with Yong that we really want this naming convention
// TODO: Rename to what?
// Dont' change without keeping in sync with Format
// clang-format off
#define GFX_FORMAT(x) \
    x( Unknown, 0, 0) \
    \
    x(R32G32B32A32_TYPELESS, 16, 1) \
    x(R32G32B32_TYPELESS, 12, 1) \
    x(R32G32_TYPELESS, 8, 1) \
    x(R32_TYPELESS, 4, 1) \
    \
    x(R16G16B16A16_TYPELESS, 8, 1) \
    x(R16G16_TYPELESS, 4, 1) \
    x(R16_TYPELESS, 2, 1) \
    \
    x(R8G8B8A8_TYPELESS, 4, 1) \
    x(R8G8_TYPELESS, 2, 1) \
    x(R8_TYPELESS, 1, 1) \
    x(B8G8R8A8_TYPELESS, 4, 1) \
    \
    x(R32G32B32A32_FLOAT, 16, 1) \
    x(R32G32B32_FLOAT, 12, 1) \
    x(R32G32_FLOAT, 8, 1) \
    x(R32_FLOAT, 4, 1) \
    \
    x(R16G16B16A16_FLOAT, 8, 1) \
    x(R16G16_FLOAT, 4, 1) \
    x(R16_FLOAT, 2, 1) \
    \
    x(R32G32B32A32_UINT, 16, 1) \
    x(R32G32B32_UINT, 12, 1) \
    x(R32G32_UINT, 8, 1) \
    x(R32_UINT, 4, 1) \
    \
    x(R16G16B16A16_UINT, 8, 1) \
    x(R16G16_UINT, 4, 1) \
    x(R16_UINT, 2, 1) \
    \
    x(R8G8B8A8_UINT, 4, 1) \
    x(R8G8_UINT, 2, 1) \
    x(R8_UINT, 1, 1) \
    \
    x(R32G32B32A32_SINT, 16, 1) \
    x(R32G32B32_SINT, 12, 1) \
    x(R32G32_SINT, 8, 1) \
    x(R32_SINT, 4, 1) \
    \
    x(R16G16B16A16_SINT, 8, 1) \
    x(R16G16_SINT, 4, 1) \
    x(R16_SINT, 2, 1) \
    \
    x(R8G8B8A8_SINT, 4, 1) \
    x(R8G8_SINT, 2, 1) \
    x(R8_SINT, 1, 1) \
    \
    x(R16G16B16A16_UNORM, 8, 1) \
    x(R16G16_UNORM, 4, 1) \
    x(R16_UNORM, 2, 1) \
    \
    x(R8G8B8A8_UNORM, 4, 1) \
    x(R8G8B8A8_UNORM_SRGB, 4, 1) \
    x(R8G8_UNORM, 2, 1) \
    x(R8_UNORM, 1, 1) \
    x(B8G8R8A8_UNORM, 4, 1) \
    x(B8G8R8A8_UNORM_SRGB, 4, 1) \
    x(B8G8R8X8_UNORM, 4, 1) \
    x(B8G8R8X8_UNORM_SRGB, 4, 1) \
    \
    x(R16G16B16A16_SNORM, 8, 1) \
    x(R16G16_SNORM, 4, 1) \
    x(R16_SNORM, 2, 1) \
    \
    x(R8G8B8A8_SNORM, 4, 1) \
    x(R8G8_SNORM, 2, 1) \
    x(R8_SNORM, 1, 1) \
    \
    x(D32_FLOAT, 4, 1) \
    x(D16_UNORM, 2, 1) \
    x(D32_FLOAT_S8_UINT, 8, 1) \
    x(R32_FLOAT_X32_TYPELESS, 8, 1) \
    \
    x(B4G4R4A4_UNORM, 2, 1) \
    x(B5G6R5_UNORM, 2, 1) \
    x(B5G5R5A1_UNORM, 2, 1) \
    \
    x(R9G9B9E5_SHAREDEXP, 4, 1) \
    x(R10G10B10A2_TYPELESS, 4, 1) \
    x(R10G10B10A2_UNORM, 4, 1) \
    x(R10G10B10A2_UINT, 4, 1) \
    x(R11G11B10_FLOAT, 4, 1) \
    \
    x(BC1_UNORM, 8, 16) \
    x(BC1_UNORM_SRGB, 8, 16) \
    x(BC2_UNORM, 16, 16) \
    x(BC2_UNORM_SRGB, 16, 16) \
    x(BC3_UNORM, 16, 16) \
    x(BC3_UNORM_SRGB, 16, 16) \
    x(BC4_UNORM, 8, 16) \
    x(BC4_SNORM, 8, 16) \
    x(BC5_UNORM, 16, 16) \
    x(BC5_SNORM, 16, 16) \
    x(BC6H_UF16, 16, 16) \
    x(BC6H_SF16, 16, 16) \
    x(BC7_UNORM, 16, 16) \
    x(BC7_UNORM_SRGB, 16, 16) \
    \
    x(R64_UINT, 8, 1) \
    \
    x(R64_SINT, 8, 1)
// clang-format on

// TODO: This should be generated from above
// TODO: enum class should be explicitly uint32_t or whatever's appropriate
/// Different formats of things like pixels or elements of vertices
/// NOTE! Any change to this type (adding, removing, changing order) - must also be reflected in
/// changes GFX_FORMAT
enum class Format
{
    // D3D formats omitted: 19-22, 44-47, 65-66, 68-70, 73, 76, 79, 82, 88-89, 92-94, 97, 100-114
    // These formats are omitted due to lack of a corresponding Vulkan format. D24_UNORM_S8_UINT
    // (DXGI_FORMAT 45) has a matching Vulkan format but is also omitted as it is only supported by
    // Nvidia.
    Unknown,

    R32G32B32A32_TYPELESS,
    R32G32B32_TYPELESS,
    R32G32_TYPELESS,
    R32_TYPELESS,

    R16G16B16A16_TYPELESS,
    R16G16_TYPELESS,
    R16_TYPELESS,

    R8G8B8A8_TYPELESS,
    R8G8_TYPELESS,
    R8_TYPELESS,
    B8G8R8A8_TYPELESS,

    R32G32B32A32_FLOAT,
    R32G32B32_FLOAT,
    R32G32_FLOAT,
    R32_FLOAT,

    R16G16B16A16_FLOAT,
    R16G16_FLOAT,
    R16_FLOAT,

    R32G32B32A32_UINT,
    R32G32B32_UINT,
    R32G32_UINT,
    R32_UINT,

    R16G16B16A16_UINT,
    R16G16_UINT,
    R16_UINT,

    R8G8B8A8_UINT,
    R8G8_UINT,
    R8_UINT,

    R32G32B32A32_SINT,
    R32G32B32_SINT,
    R32G32_SINT,
    R32_SINT,

    R16G16B16A16_SINT,
    R16G16_SINT,
    R16_SINT,

    R8G8B8A8_SINT,
    R8G8_SINT,
    R8_SINT,

    R16G16B16A16_UNORM,
    R16G16_UNORM,
    R16_UNORM,

    R8G8B8A8_UNORM,
    R8G8B8A8_UNORM_SRGB,
    R8G8_UNORM,
    R8_UNORM,
    B8G8R8A8_UNORM,
    B8G8R8A8_UNORM_SRGB,
    B8G8R8X8_UNORM,
    B8G8R8X8_UNORM_SRGB,

    R16G16B16A16_SNORM,
    R16G16_SNORM,
    R16_SNORM,

    R8G8B8A8_SNORM,
    R8G8_SNORM,
    R8_SNORM,

    D32_FLOAT,
    D16_UNORM,
    D32_FLOAT_S8_UINT,
    R32_FLOAT_X32_TYPELESS,

    B4G4R4A4_UNORM,
    B5G6R5_UNORM,
    B5G5R5A1_UNORM,

    R9G9B9E5_SHAREDEXP,
    R10G10B10A2_TYPELESS,
    R10G10B10A2_UNORM,
    R10G10B10A2_UINT,
    R11G11B10_FLOAT,

    BC1_UNORM,
    BC1_UNORM_SRGB,
    BC2_UNORM,
    BC2_UNORM_SRGB,
    BC3_UNORM,
    BC3_UNORM_SRGB,
    BC4_UNORM,
    BC4_SNORM,
    BC5_UNORM,
    BC5_SNORM,
    BC6H_UF16,
    BC6H_SF16,
    BC7_UNORM,
    BC7_UNORM_SRGB,

    R64_UINT,

    R64_SINT,

    _Count,
};

// TODO: Aspect = Color, Depth, Stencil, etc.
// TODO: Channel = R, G, B, A, D, S, etc.
// TODO: Pick : pixel or texel
// TODO: Block is a good term for what it is
// TODO: Width/Height/Depth/whatever should not be used. We should use extentX, extentY, etc.
struct FormatInfo
{
    GfxCount
        channelCount; ///< The amount of channels in the format. Only set if the channelType is set
    uint8_t channelType; ///< One of SlangScalarType None if type isn't made up of elements of type.
                         ///< TODO: Change to uint32_t?

    Size blockSizeInBytes;   ///< The size of a block in bytes.
    GfxCount pixelsPerBlock; ///< The number of pixels contained in a block.
    GfxCount blockWidth;     ///< The width of a block in pixels.
    GfxCount blockHeight;    ///< The height of a block in pixels.
};

enum class InputSlotClass
{
    PerVertex,
    PerInstance
};

struct InputElementDesc
{
    char const* semanticName; ///< The name of the corresponding parameter in shader code.
    GfxIndex semanticIndex;   ///< The index of the corresponding parameter in shader code. Only
                              ///< needed if multiple parameters share a semantic name.
    Format format;            ///< The format of the data being fetched for this element.
    Offset offset; ///< The offset in bytes of this element from the start of the corresponding
                   ///< chunk of vertex stream data.
    GfxIndex bufferSlotIndex; ///< The index of the vertex stream to fetch this element's data from.
};

struct VertexStreamDesc
{
    Size stride;                   ///< The stride in bytes for this vertex stream.
    InputSlotClass slotClass;      ///< Whether the stream contains per-vertex or per-instance data.
    GfxCount instanceDataStepRate; ///< How many instances to draw per chunk of data.
};

enum class PrimitiveType
{
    Point,
    Line,
    Triangle,
    Patch
};

enum class PrimitiveTopology
{
    TriangleList,
    TriangleStrip,
    PointList,
    LineList,
    LineStrip
};

enum class ResourceState
{
    Undefined,
    General,
    PreInitialized,
    VertexBuffer,
    IndexBuffer,
    ConstantBuffer,
    StreamOutput,
    ShaderResource,
    UnorderedAccess,
    RenderTarget,
    DepthRead,
    DepthWrite,
    Present,
    IndirectArgument,
    CopySource,
    CopyDestination,
    ResolveSource,
    ResolveDestination,
    AccelerationStructure,
    AccelerationStructureBuildInput,
    PixelShaderResource,
    NonPixelShaderResource,
    _Count
};

struct ResourceStateSet
{
public:
    void add(ResourceState state) { m_bitFields |= (1LL << (uint32_t)state); }
    template<typename... TResourceState>
    void add(ResourceState s, TResourceState... states)
    {
        add(s);
        add(states...);
    }
    bool contains(ResourceState state) const
    {
        return (m_bitFields & (1LL << (uint32_t)state)) != 0;
    }
    ResourceStateSet()
        : m_bitFields(0)
    {
    }
    ResourceStateSet(const ResourceStateSet& other) = default;
    ResourceStateSet(ResourceState state) { add(state); }
    template<typename... TResourceState>
    ResourceStateSet(TResourceState... states)
    {
        add(states...);
    }

    ResourceStateSet operator&(const ResourceStateSet& that) const
    {
        ResourceStateSet result;
        result.m_bitFields = this->m_bitFields & that.m_bitFields;
        return result;
    }

private:
    uint64_t m_bitFields = 0;
    void add() {}
};


/// Describes how memory for the resource should be allocated for CPU access.
enum class MemoryType
{
    DeviceLocal,
    Upload,
    ReadBack,
};

enum class InteropHandleAPI
{
    Unknown,
    D3D12,                    // A D3D12 object pointer.
    Vulkan,                   // A general Vulkan object handle.
    CUDA,                     // A general CUDA object handle.
    Win32,                    // A general Win32 HANDLE.
    FileDescriptor,           // A file descriptor.
    DeviceAddress,            // A device address.
    D3D12CpuDescriptorHandle, // A D3D12_CPU_DESCRIPTOR_HANDLE value.
    Metal,                    // A general Metal object handle.
};

struct InteropHandle
{
    InteropHandleAPI api = InteropHandleAPI::Unknown;
    uint64_t handleValue = 0;
};

// Declare opaque type
class IInputLayout : public ISlangUnknown
{
public:
    struct Desc
    {
        InputElementDesc const* inputElements = nullptr;
        GfxCount inputElementCount = 0;
        VertexStreamDesc const* vertexStreams = nullptr;
        GfxCount vertexStreamCount = 0;
    };
};
#define SLANG_UUID_IInputLayout                            \
    {                                                      \
        0x45223711, 0xa84b, 0x455c,                        \
        {                                                  \
            0xbe, 0xfa, 0x49, 0x37, 0x42, 0x1e, 0x8e, 0x2e \
        }                                                  \
    }

class IResource : public ISlangUnknown
{
public:
    /// The type of resource.
    /// NOTE! The order needs to be such that all texture types are at or after Texture1D (otherwise
    /// isTexture won't work correctly)
    enum class Type
    {
        Unknown,     ///< Unknown
        Buffer,      ///< A buffer (like a constant/index/vertex buffer)
        Texture1D,   ///< A 1d texture
        Texture2D,   ///< A 2d texture
        Texture3D,   ///< A 3d texture
        TextureCube, ///< A cubemap consists of 6 Texture2D like faces
        _Count,
    };

    /// Base class for Descs
    struct DescBase
    {
        Type type = Type::Unknown;
        ResourceState defaultState = ResourceState::Undefined;
        ResourceStateSet allowedStates = ResourceStateSet();
        MemoryType memoryType = MemoryType::DeviceLocal;
        InteropHandle existingHandle = {};
        bool isShared = false;
    };

    virtual SLANG_NO_THROW Type SLANG_MCALL getType() = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeResourceHandle(InteropHandle* outHandle) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL getSharedHandle(InteropHandle* outHandle) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL setDebugName(const char* name) = 0;
    virtual SLANG_NO_THROW const char* SLANG_MCALL getDebugName() = 0;
};
#define SLANG_UUID_IResource                               \
    {                                                      \
        0xa0e39f34, 0x8398, 0x4522,                        \
        {                                                  \
            0x95, 0xc2, 0xeb, 0xc0, 0xf9, 0x84, 0xef, 0x3f \
        }                                                  \
    }

struct MemoryRange
{
    // TODO: Change to Offset/Size?
    uint64_t offset;
    uint64_t size;
};

class IBufferResource : public IResource
{
public:
    struct Desc : public DescBase
    {
        Size sizeInBytes = 0; ///< Total size in bytes
        Size elementSize = 0; ///< Get the element stride. If > 0, this is a structured buffer
        Format format = Format::Unknown;
    };

    virtual SLANG_NO_THROW Desc* SLANG_MCALL getDesc() = 0;
    virtual SLANG_NO_THROW DeviceAddress SLANG_MCALL getDeviceAddress() = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL map(MemoryRange* rangeToRead, void** outPointer) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL unmap(MemoryRange* writtenRange) = 0;
};
#define SLANG_UUID_IBufferResource                         \
    {                                                      \
        0x1b274efe, 0x5e37, 0x492b,                        \
        {                                                  \
            0x82, 0x6e, 0x7e, 0xe7, 0xe8, 0xf5, 0xa4, 0x9b \
        }                                                  \
    }

struct DepthStencilClearValue
{
    float depth = 1.0f;
    uint32_t stencil = 0;
};
union ColorClearValue
{
    float floatValues[4];
    uint32_t uintValues[4];
};
struct ClearValue
{
    ColorClearValue color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    DepthStencilClearValue depthStencil;
};

struct BufferRange
{
    Offset offset; ///< Offset in bytes.
    Size size;     ///< Size in bytes.
};

enum class TextureAspect : uint32_t
{
    Default = 0,
    Color = 0x00000001,
    Depth = 0x00000002,
    Stencil = 0x00000004,
    MetaData = 0x00000008,
    Plane0 = 0x00000010,
    Plane1 = 0x00000020,
    Plane2 = 0x00000040,

    DepthStencil = Depth | Stencil,
};

struct SubresourceRange
{
    TextureAspect aspectMask;
    GfxIndex mipLevel;
    GfxCount mipLevelCount;
    GfxIndex baseArrayLayer; // For Texture3D, this is WSlice.
    GfxCount layerCount;     // For cube maps, this is a multiple of 6.
};

class ITextureResource : public IResource
{
public:
    static const GfxCount kRemainingTextureSize = 0xffffffff;
    struct Offset3D
    {
        GfxIndex x = 0;
        GfxIndex y = 0;
        GfxIndex z = 0;
        Offset3D() = default;
        Offset3D(GfxIndex _x, GfxIndex _y, GfxIndex _z)
            : x(_x), y(_y), z(_z)
        {
        }
    };

    struct SampleDesc
    {
        GfxCount numSamples = 1; ///< Number of samples per pixel
        int quality = 0;         ///< The quality measure for the samples
    };

    struct Extents
    {
        GfxCount width = 0;  ///< Width in pixels
        GfxCount height = 0; ///< Height in pixels (if 2d or 3d)
        GfxCount depth = 0;  ///< Depth (if 3d)
    };

    struct Desc : public DescBase
    {
        Extents size;

        GfxCount arraySize = 0; ///< Array size

        GfxCount numMipLevels = 0; ///< Number of mip levels - if 0 will create all mip levels
        Format format;             ///< The resources format
        SampleDesc sampleDesc;     ///< How the resource is sampled
        ClearValue* optimalClearValue = nullptr;
    };

    /// Data for a single subresource of a texture.
    ///
    /// Each subresource is a tensor with `1 <= rank <= 3`,
    /// where the rank is deterined by the base shape of the
    /// texture (Buffer, 1D, 2D, 3D, or Cube). For the common
    /// case of a 2D texture, `rank == 2` and each subresource
    /// is a 2D image.
    ///
    /// Subresource tensors must be stored in a row-major layout,
    /// so that the X axis strides over texels, the Y axis strides
    /// over 1D rows of texels, and the Z axis strides over 2D
    /// "layers" of texels.
    ///
    /// For a texture with multiple mip levels or array elements,
    /// each mip level and array element is stores as a distinct
    /// subresource. When indexing into an array of subresources,
    /// the index of a subresoruce for mip level `m` and array
    /// index `a` is `m + a*mipLevelCount`.
    ///
    struct SubresourceData
    {
        /// Pointer to texel data for the subresource tensor.
        void const* data;

        /// Stride in bytes between rows of the subresource tensor.
        ///
        /// This is the number of bytes to add to a pointer to a texel
        /// at (X,Y,Z) to get to a texel at (X,Y+1,Z).
        ///
        /// Devices may not support all possible values for `strideY`.
        /// In particular, they may only support strictly positive strides.
        ///
        gfx::Size strideY;

        /// Stride in bytes between layers of the subresource tensor.
        ///
        /// This is the number of bytes to add to a pointer to a texel
        /// at (X,Y,Z) to get to a texel at (X,Y,Z+1).
        ///
        /// Devices may not support all possible values for `strideZ`.
        /// In particular, they may only support strictly positive strides.
        ///
        gfx::Size strideZ;
    };

    virtual SLANG_NO_THROW Desc* SLANG_MCALL getDesc() = 0;
};
#define SLANG_UUID_ITextureResource                        \
    {                                                      \
        0xcf88a31c, 0x6187, 0x46c5,                        \
        {                                                  \
            0xa4, 0xb7, 0xeb, 0x58, 0xc7, 0x33, 0x40, 0x17 \
        }                                                  \
    }


enum class ComparisonFunc : uint8_t
{
    Never = 0x0,
    Less = 0x1,
    Equal = 0x2,
    LessEqual = 0x3,
    Greater = 0x4,
    NotEqual = 0x5,
    GreaterEqual = 0x6,
    Always = 0x7,
};

enum class TextureFilteringMode
{
    Point,
    Linear,
};

enum class TextureAddressingMode
{
    Wrap,
    ClampToEdge,
    ClampToBorder,
    MirrorRepeat,
    MirrorOnce,
};

enum class TextureReductionOp
{
    Average,
    Comparison,
    Minimum,
    Maximum,
};

class ISamplerState : public ISlangUnknown
{
public:
    struct Desc
    {
        TextureFilteringMode minFilter = TextureFilteringMode::Linear;
        TextureFilteringMode magFilter = TextureFilteringMode::Linear;
        TextureFilteringMode mipFilter = TextureFilteringMode::Linear;
        TextureReductionOp reductionOp = TextureReductionOp::Average;
        TextureAddressingMode addressU = TextureAddressingMode::Wrap;
        TextureAddressingMode addressV = TextureAddressingMode::Wrap;
        TextureAddressingMode addressW = TextureAddressingMode::Wrap;
        float mipLODBias = 0.0f;
        uint32_t maxAnisotropy = 1;
        ComparisonFunc comparisonFunc = ComparisonFunc::Never;
        float borderColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        float minLOD = -FLT_MAX;
        float maxLOD = FLT_MAX;
    };

    /// Returns a native API handle representing this sampler state object.
    /// When using D3D12, this will be a D3D12_CPU_DESCRIPTOR_HANDLE.
    /// When using Vulkan, this will be a VkSampler.
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outNativeHandle) = 0;
};
#define SLANG_UUID_ISamplerState                           \
    {                                                      \
        0x8b8055df, 0x9377, 0x401d,                        \
        {                                                  \
            0x91, 0xff, 0x3f, 0xa3, 0xbf, 0x66, 0x64, 0xf4 \
        }                                                  \
    }

class IResourceView : public ISlangUnknown
{
public:
    enum class Type
    {
        Unknown,

        RenderTarget,
        DepthStencil,
        ShaderResource,
        UnorderedAccess,
        AccelerationStructure,

        CountOf_,
    };

    struct RenderTargetDesc
    {
        // The resource shape of this render target view.
        IResource::Type shape;
    };

    struct Desc
    {
        Type type;
        Format format;

        // Required fields for `RenderTarget` and `DepthStencil` views.
        RenderTargetDesc renderTarget;
        // Specifies the range of a texture resource for a
        // ShaderRsource/UnorderedAccess/RenderTarget/DepthStencil view.
        SubresourceRange subresourceRange;
        // Specifies the range of a buffer resource for a ShaderResource/UnorderedAccess view.
        BufferRange bufferRange;
    };
    virtual SLANG_NO_THROW Desc* SLANG_MCALL getViewDesc() = 0;

    /// Returns a native API handle representing this resource view object.
    /// When using D3D12, this will be a D3D12_CPU_DESCRIPTOR_HANDLE or a buffer device address
    /// depending on the type of the resource view. When using Vulkan, this will be a VkImageView,
    /// VkBufferView, VkAccelerationStructure or a VkBuffer depending on the type of the resource
    /// view.
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outNativeHandle) = 0;
};
#define SLANG_UUID_IResourceView                           \
    {                                                      \
        0x7b6c4926, 0x884, 0x408c,                         \
        {                                                  \
            0xad, 0x8a, 0x50, 0x3a, 0x8e, 0x23, 0x98, 0xa4 \
        }                                                  \
    }

class IAccelerationStructure : public IResourceView
{
public:
    enum class Kind
    {
        TopLevel,
        BottomLevel
    };

    struct BuildFlags
    {
        // The enum values are intentionally consistent with
        // D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS.
        enum Enum
        {
            None,
            AllowUpdate = 1,
            AllowCompaction = 2,
            PreferFastTrace = 4,
            PreferFastBuild = 8,
            MinimizeMemory = 16,
            PerformUpdate = 32
        };
    };

    enum class GeometryType
    {
        Triangles,
        ProcedurePrimitives
    };

    struct GeometryFlags
    {
        // The enum values are intentionally consistent with
        // D3D12_RAYTRACING_GEOMETRY_FLAGS.
        enum Enum
        {
            None,
            Opaque = 1,
            NoDuplicateAnyHitInvocation = 2
        };
    };

    struct TriangleDesc
    {
        DeviceAddress transform3x4;
        Format indexFormat;
        Format vertexFormat;
        GfxCount indexCount;
        GfxCount vertexCount;
        DeviceAddress indexData;
        DeviceAddress vertexData;
        Size vertexStride;
    };

    struct ProceduralAABB
    {
        float minX;
        float minY;
        float minZ;
        float maxX;
        float maxY;
        float maxZ;
    };

    struct ProceduralAABBDesc
    {
        /// Number of AABBs.
        GfxCount count;

        /// Pointer to an array of `ProceduralAABB` values in device memory.
        DeviceAddress data;

        /// Stride in bytes of the AABB values array.
        Size stride;
    };

    struct GeometryDesc
    {
        GeometryType type;
        GeometryFlags::Enum flags;
        union
        {
            TriangleDesc triangles;
            ProceduralAABBDesc proceduralAABBs;
        } content;
    };

    struct GeometryInstanceFlags
    {
        // The enum values are kept consistent with D3D12_RAYTRACING_INSTANCE_FLAGS
        // and VkGeometryInstanceFlagBitsKHR.
        enum Enum : uint32_t
        {
            None = 0,
            TriangleFacingCullDisable = 0x00000001,
            TriangleFrontCounterClockwise = 0x00000002,
            ForceOpaque = 0x00000004,
            NoOpaque = 0x00000008
        };
    };

    // TODO: Should any of these be changed?
    // The layout of this struct is intentionally consistent with D3D12_RAYTRACING_INSTANCE_DESC
    // and VkAccelerationStructureInstanceKHR.
    struct InstanceDesc
    {
        float transform[3][4];
        uint32_t instanceID : 24;
        uint32_t instanceMask : 8;
        uint32_t instanceContributionToHitGroupIndex : 24;
        uint32_t flags : 8; // Combination of GeometryInstanceFlags::Enum values.
        DeviceAddress accelerationStructure;
    };

    struct PrebuildInfo
    {
        Size resultDataMaxSize;
        Size scratchDataSize;
        Size updateScratchDataSize;
    };

    struct BuildInputs
    {
        Kind kind;

        BuildFlags::Enum flags;

        GfxCount descCount;

        /// Array of `InstanceDesc` values in device memory.
        /// Used when `kind` is `TopLevel`.
        DeviceAddress instanceDescs;

        /// Array of `GeometryDesc` values.
        /// Used when `kind` is `BottomLevel`.
        const GeometryDesc* geometryDescs;
    };

    struct CreateDesc
    {
        Kind kind;
        IBufferResource* buffer;
        Offset offset;
        Size size;
    };

    struct BuildDesc
    {
        BuildInputs inputs;
        IAccelerationStructure* source;
        IAccelerationStructure* dest;
        DeviceAddress scratchData;
    };

    virtual SLANG_NO_THROW DeviceAddress SLANG_MCALL getDeviceAddress() = 0;
};
#define SLANG_UUID_IAccelerationStructure                 \
    {                                                     \
        0xa5cdda3c, 0x1d4e, 0x4df7,                       \
        {                                                 \
            0x8e, 0xf2, 0xb7, 0x3f, 0xce, 0x4, 0xde, 0x3b \
        }                                                 \
    }

class IFence : public ISlangUnknown
{
public:
    struct Desc
    {
        uint64_t initialValue = 0;
        bool isShared = false;
    };

    /// Returns the currently signaled value on the device.
    virtual SLANG_NO_THROW Result SLANG_MCALL getCurrentValue(uint64_t* outValue) = 0;

    /// Signals the fence from the host with the specified value.
    virtual SLANG_NO_THROW Result SLANG_MCALL setCurrentValue(uint64_t value) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL getSharedHandle(InteropHandle* outHandle) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outNativeHandle) = 0;
};
#define SLANG_UUID_IFence                                 \
    {                                                     \
        0x7fe1c283, 0xd3f4, 0x48ed,                       \
        {                                                 \
            0xaa, 0xf3, 0x1, 0x51, 0x96, 0x4e, 0x7c, 0xb5 \
        }                                                 \
    }

struct ShaderOffset
{
    SlangInt uniformOffset = 0; // TODO: Change to Offset?
    GfxIndex bindingRangeIndex = 0;
    GfxIndex bindingArrayIndex = 0;
    uint32_t getHashCode() const
    {
        return (uint32_t)(((bindingRangeIndex << 20) + bindingArrayIndex) ^ uniformOffset);
    }
    bool operator==(const ShaderOffset& other) const
    {
        return uniformOffset == other.uniformOffset &&
               bindingRangeIndex == other.bindingRangeIndex &&
               bindingArrayIndex == other.bindingArrayIndex;
    }
    bool operator!=(const ShaderOffset& other) const { return !this->operator==(other); }
    bool operator<(const ShaderOffset& other) const
    {
        if (bindingRangeIndex < other.bindingRangeIndex)
            return true;
        if (bindingRangeIndex > other.bindingRangeIndex)
            return false;
        if (bindingArrayIndex < other.bindingArrayIndex)
            return true;
        if (bindingArrayIndex > other.bindingArrayIndex)
            return false;
        return uniformOffset < other.uniformOffset;
    }
    bool operator<=(const ShaderOffset& other) const { return (*this == other) || (*this) < other; }
    bool operator>(const ShaderOffset& other) const { return other < *this; }
    bool operator>=(const ShaderOffset& other) const { return other <= *this; }
};

enum class ShaderObjectContainerType
{
    None,
    Array,
    StructuredBuffer
};

class IShaderObject : public ISlangUnknown
{
public:
    virtual SLANG_NO_THROW slang::TypeLayoutReflection* SLANG_MCALL getElementTypeLayout() = 0;
    virtual SLANG_NO_THROW ShaderObjectContainerType SLANG_MCALL getContainerType() = 0;
    virtual SLANG_NO_THROW GfxCount SLANG_MCALL getEntryPointCount() = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getEntryPoint(GfxIndex index, IShaderObject** entryPoint) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    setData(ShaderOffset const& offset, void const* data, Size size) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getObject(ShaderOffset const& offset, IShaderObject** object) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    setObject(ShaderOffset const& offset, IShaderObject* object) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    setResource(ShaderOffset const& offset, IResourceView* resourceView) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    setSampler(ShaderOffset const& offset, ISamplerState* sampler) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL setCombinedTextureSampler(
        ShaderOffset const& offset,
        IResourceView* textureView,
        ISamplerState* sampler) = 0;

    /// Manually overrides the specialization argument for the sub-object binding at `offset`.
    /// Specialization arguments are passed to the shader compiler to specialize the type
    /// of interface-typed shader parameters.
    virtual SLANG_NO_THROW Result SLANG_MCALL setSpecializationArgs(
        ShaderOffset const& offset,
        const slang::SpecializationArg* args,
        GfxCount count) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getCurrentVersion(ITransientResourceHeap* transientHeap, IShaderObject** outObject) = 0;

    virtual SLANG_NO_THROW const void* SLANG_MCALL getRawData() = 0;

    virtual SLANG_NO_THROW Size SLANG_MCALL getSize() = 0;

    /// Use the provided constant buffer instead of the internally created one.
    virtual SLANG_NO_THROW Result SLANG_MCALL
    setConstantBufferOverride(IBufferResource* constantBuffer) = 0;


    inline ComPtr<IShaderObject> getObject(ShaderOffset const& offset)
    {
        ComPtr<IShaderObject> object = nullptr;
        SLANG_RETURN_NULL_ON_FAIL(getObject(offset, object.writeRef()));
        return object;
    }
    inline ComPtr<IShaderObject> getEntryPoint(GfxIndex index)
    {
        ComPtr<IShaderObject> entryPoint = nullptr;
        SLANG_RETURN_NULL_ON_FAIL(getEntryPoint(index, entryPoint.writeRef()));
        return entryPoint;
    }
};
#define SLANG_UUID_IShaderObject                          \
    {                                                     \
        0xc1fa997e, 0x5ca2, 0x45ae,                       \
        {                                                 \
            0x9b, 0xcb, 0xc4, 0x35, 0x9e, 0x85, 0x5, 0x85 \
        }                                                 \
    }

enum class StencilOp : uint8_t
{
    Keep,
    Zero,
    Replace,
    IncrementSaturate,
    DecrementSaturate,
    Invert,
    IncrementWrap,
    DecrementWrap,
};

enum class FillMode : uint8_t
{
    Solid,
    Wireframe,
};

enum class CullMode : uint8_t
{
    None,
    Front,
    Back,
};

enum class FrontFaceMode : uint8_t
{
    CounterClockwise,
    Clockwise,
};

struct DepthStencilOpDesc
{
    StencilOp stencilFailOp = StencilOp::Keep;
    StencilOp stencilDepthFailOp = StencilOp::Keep;
    StencilOp stencilPassOp = StencilOp::Keep;
    ComparisonFunc stencilFunc = ComparisonFunc::Always;
};

struct DepthStencilDesc
{
    bool depthTestEnable = false;
    bool depthWriteEnable = true;
    ComparisonFunc depthFunc = ComparisonFunc::Less;

    bool stencilEnable = false;
    uint32_t stencilReadMask = 0xFFFFFFFF;
    uint32_t stencilWriteMask = 0xFFFFFFFF;
    DepthStencilOpDesc frontFace;
    DepthStencilOpDesc backFace;

    uint32_t stencilRef = 0; // TODO: this should be removed
};

struct RasterizerDesc
{
    FillMode fillMode = FillMode::Solid;
    CullMode cullMode = CullMode::None;
    FrontFaceMode frontFace = FrontFaceMode::CounterClockwise;
    int32_t depthBias = 0;
    float depthBiasClamp = 0.0f;
    float slopeScaledDepthBias = 0.0f;
    bool depthClipEnable = true;
    bool scissorEnable = false;
    bool multisampleEnable = false;
    bool antialiasedLineEnable = false;
    bool enableConservativeRasterization = false;
    uint32_t forcedSampleCount = 0;
};

enum class LogicOp
{
    NoOp,
};

enum class BlendOp
{
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
};

enum class BlendFactor
{
    Zero,
    One,
    SrcColor,
    InvSrcColor,
    SrcAlpha,
    InvSrcAlpha,
    DestAlpha,
    InvDestAlpha,
    DestColor,
    InvDestColor,
    SrcAlphaSaturate,
    BlendColor,
    InvBlendColor,
    SecondarySrcColor,
    InvSecondarySrcColor,
    SecondarySrcAlpha,
    InvSecondarySrcAlpha,
};

namespace RenderTargetWriteMask
{
typedef uint8_t Type;
enum
{
    EnableNone = 0,
    EnableRed = 0x01,
    EnableGreen = 0x02,
    EnableBlue = 0x04,
    EnableAlpha = 0x08,
    EnableAll = 0x0F,
};
}; // namespace RenderTargetWriteMask
typedef RenderTargetWriteMask::Type RenderTargetWriteMaskT;

struct AspectBlendDesc
{
    BlendFactor srcFactor = BlendFactor::One;
    BlendFactor dstFactor = BlendFactor::Zero;
    BlendOp op = BlendOp::Add;
};

struct TargetBlendDesc
{
    AspectBlendDesc color;
    AspectBlendDesc alpha;
    bool enableBlend = false;
    LogicOp logicOp = LogicOp::NoOp;
    RenderTargetWriteMaskT writeMask = RenderTargetWriteMask::EnableAll;
};

struct BlendDesc
{
    TargetBlendDesc targets[kMaxRenderTargetCount];
    GfxCount targetCount = 0;

    bool alphaToCoverageEnable = false;
};

class IFramebufferLayout : public ISlangUnknown
{
public:
    struct TargetLayout
    {
        Format format;
        GfxCount sampleCount;
    };
    struct Desc
    {
        GfxCount renderTargetCount;
        TargetLayout* renderTargets = nullptr;
        TargetLayout* depthStencil = nullptr;
    };
};
#define SLANG_UUID_IFramebufferLayout                     \
    {                                                     \
        0xa838785, 0xc13a, 0x4832,                        \
        {                                                 \
            0xad, 0x88, 0x64, 0x6, 0xb5, 0x4b, 0x5e, 0xba \
        }                                                 \
    }

struct GraphicsPipelineStateDesc
{
    IShaderProgram* program = nullptr;

    IInputLayout* inputLayout = nullptr;
    IFramebufferLayout* framebufferLayout = nullptr;
    PrimitiveType primitiveType = PrimitiveType::Triangle;
    DepthStencilDesc depthStencil;
    RasterizerDesc rasterizer;
    BlendDesc blend;
};

struct ComputePipelineStateDesc
{
    IShaderProgram* program = nullptr;
    void* d3d12RootSignatureOverride = nullptr;
};

struct RayTracingPipelineFlags
{
    enum Enum : uint32_t
    {
        None = 0,
        SkipTriangles = 1,
        SkipProcedurals = 2,
    };
};

struct HitGroupDesc
{
    const char* hitGroupName = nullptr;
    const char* closestHitEntryPoint = nullptr;
    const char* anyHitEntryPoint = nullptr;
    const char* intersectionEntryPoint = nullptr;
};

struct RayTracingPipelineStateDesc
{
    IShaderProgram* program = nullptr;
    GfxCount hitGroupCount = 0;
    const HitGroupDesc* hitGroups = nullptr;
    int maxRecursion = 0;
    Size maxRayPayloadSize = 0;
    Size maxAttributeSizeInBytes = 8;
    RayTracingPipelineFlags::Enum flags = RayTracingPipelineFlags::None;
};

class IShaderTable : public ISlangUnknown
{
public:
    // Specifies the bytes to overwrite into a record in the shader table.
    struct ShaderRecordOverwrite
    {
        Offset offset;   // Offset within the shader record.
        Size size;       // Number of bytes to overwrite.
        uint8_t data[8]; // Content to overwrite.
    };

    struct Desc
    {
        GfxCount rayGenShaderCount;
        const char** rayGenShaderEntryPointNames;
        const ShaderRecordOverwrite* rayGenShaderRecordOverwrites;

        GfxCount missShaderCount;
        const char** missShaderEntryPointNames;
        const ShaderRecordOverwrite* missShaderRecordOverwrites;

        GfxCount hitGroupCount;
        const char** hitGroupNames;
        const ShaderRecordOverwrite* hitGroupRecordOverwrites;

        GfxCount callableShaderCount;
        const char** callableShaderEntryPointNames;
        const ShaderRecordOverwrite* callableShaderRecordOverwrites;

        IShaderProgram* program;
    };
};
#define SLANG_UUID_IShaderTable                            \
    {                                                      \
        0xa721522c, 0xdf31, 0x4c2f,                        \
        {                                                  \
            0xa5, 0xe7, 0x3b, 0xe0, 0x12, 0x4b, 0x31, 0x78 \
        }                                                  \
    }

class IPipelineState : public ISlangUnknown
{
public:
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) = 0;
};
#define SLANG_UUID_IPipelineState                          \
    {                                                      \
        0xca7e57d, 0x8a90, 0x44f3,                         \
        {                                                  \
            0xbd, 0xb1, 0xfe, 0x9b, 0x35, 0x3f, 0x5a, 0x72 \
        }                                                  \
    }


struct ScissorRect
{
    int32_t minX;
    int32_t minY;
    int32_t maxX;
    int32_t maxY;
};

struct Viewport
{
    float originX = 0.0f;
    float originY = 0.0f;
    float extentX = 0.0f;
    float extentY = 0.0f;
    float minZ = 0.0f;
    float maxZ = 1.0f;
};

class IFramebuffer : public ISlangUnknown
{
public:
    struct Desc
    {
        GfxCount renderTargetCount;
        IResourceView* const* renderTargetViews;
        IResourceView* depthStencilView;
        IFramebufferLayout* layout;
    };
};
#define SLANG_UUID_IFrameBuffer                            \
    {                                                      \
        0xf0c0d9a, 0x4ef3, 0x4e18,                         \
        {                                                  \
            0x9b, 0xa9, 0x34, 0x60, 0xea, 0x69, 0x87, 0x95 \
        }                                                  \
    }

struct WindowHandle
{
    enum class Type
    {
        Unknown,
        Win32Handle,
        NSWindowHandle,
        XLibHandle,
    };
    Type type;
    intptr_t handleValues[2];
    static WindowHandle FromHwnd(void* hwnd)
    {
        WindowHandle handle = {};
        handle.type = WindowHandle::Type::Win32Handle;
        handle.handleValues[0] = (intptr_t)(hwnd);
        return handle;
    }
    static WindowHandle FromNSWindow(void* nswindow)
    {
        WindowHandle handle = {};
        handle.type = WindowHandle::Type::NSWindowHandle;
        handle.handleValues[0] = (intptr_t)(nswindow);
        return handle;
    }
    static WindowHandle FromXWindow(void* xdisplay, uint32_t xwindow)
    {
        WindowHandle handle = {};
        handle.type = WindowHandle::Type::XLibHandle;
        handle.handleValues[0] = (intptr_t)(xdisplay);
        handle.handleValues[1] = xwindow;
        return handle;
    }
};

struct FaceMask
{
    enum Enum
    {
        Front = 1,
        Back = 2
    };
};

class IRenderPassLayout : public ISlangUnknown
{
public:
    enum class TargetLoadOp
    {
        Load,
        Clear,
        DontCare
    };
    enum class TargetStoreOp
    {
        Store,
        DontCare
    };
    struct TargetAccessDesc
    {
        TargetLoadOp loadOp;
        TargetLoadOp stencilLoadOp;
        TargetStoreOp storeOp;
        TargetStoreOp stencilStoreOp;
        ResourceState initialState;
        ResourceState finalState;
    };
    struct Desc
    {
        IFramebufferLayout* framebufferLayout = nullptr;
        GfxCount renderTargetCount;
        TargetAccessDesc* renderTargetAccess = nullptr;
        TargetAccessDesc* depthStencilAccess = nullptr;
    };
};
#define SLANG_UUID_IRenderPassLayout                       \
    {                                                      \
        0xdaab0b1a, 0xf45d, 0x4ae9,                        \
        {                                                  \
            0xbf, 0x2c, 0xe0, 0xbb, 0x76, 0x7d, 0xfa, 0xd1 \
        }                                                  \
    }

enum class QueryType
{
    Timestamp,
    AccelerationStructureCompactedSize,
    AccelerationStructureSerializedSize,
    AccelerationStructureCurrentSize,
};

class IQueryPool : public ISlangUnknown
{
public:
    struct Desc
    {
        QueryType type;
        GfxCount count;
    };

public:
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getResult(GfxIndex queryIndex, GfxCount count, uint64_t* data) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL reset() = 0;
};
#define SLANG_UUID_IQueryPool                              \
    {                                                      \
        0xc2cc3784, 0x12da, 0x480a,                        \
        {                                                  \
            0xa8, 0x74, 0x8b, 0x31, 0x96, 0x1c, 0xa4, 0x36 \
        }                                                  \
    }


class ICommandEncoder : public ISlangUnknown
{
    SLANG_COM_INTERFACE(
        0x77ea6383,
        0xbe3d,
        0x40aa,
        {0x8b, 0x45, 0xfd, 0xf0, 0xd7, 0x5b, 0xfa, 0x34});

public:
    virtual SLANG_NO_THROW void SLANG_MCALL endEncoding() = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL
    writeTimestamp(IQueryPool* queryPool, GfxIndex queryIndex) = 0;
};

struct IndirectDispatchArguments
{
    GfxCount ThreadGroupCountX;
    GfxCount ThreadGroupCountY;
    GfxCount ThreadGroupCountZ;
};

struct IndirectDrawArguments
{
    GfxCount VertexCountPerInstance;
    GfxCount InstanceCount;
    GfxIndex StartVertexLocation;
    GfxIndex StartInstanceLocation;
};

struct IndirectDrawIndexedArguments
{
    GfxCount IndexCountPerInstance;
    GfxCount InstanceCount;
    GfxIndex StartIndexLocation;
    GfxIndex BaseVertexLocation;
    GfxIndex StartInstanceLocation;
};

struct SamplePosition
{
    int8_t x;
    int8_t y;
};

struct ClearResourceViewFlags
{
    enum Enum : uint32_t
    {
        None = 0,
        ClearDepth = 1,
        ClearStencil = 2,
        FloatClearValues = 4
    };
};

enum class CooperativeVectorComponentType
{
    Float16 = 0,
    Float32 = 1,
    Float64 = 2,
    SInt8 = 3,
    SInt16 = 4,
    SInt32 = 5,
    SInt64 = 6,
    UInt8 = 7,
    UInt16 = 8,
    UInt32 = 9,
    UInt64 = 10,
    SInt8Packed = 11,
    UInt8Packed = 12,
    FloatE4M3 = 13,
    FloatE5M2 = 14,
};

struct CooperativeVectorProperties
{
    CooperativeVectorComponentType inputType;
    CooperativeVectorComponentType inputInterpretation;
    CooperativeVectorComponentType matrixInterpretation;
    CooperativeVectorComponentType biasInterpretation;
    CooperativeVectorComponentType resultType;
    bool transpose;
};


class IResourceCommandEncoder : public ICommandEncoder
{
    // {F99A00E9-ED50-4088-8A0E-3B26755031EA}
    SLANG_COM_INTERFACE(
        0xf99a00e9,
        0xed50,
        0x4088,
        {0x8a, 0xe, 0x3b, 0x26, 0x75, 0x50, 0x31, 0xea});

public:
    virtual SLANG_NO_THROW void SLANG_MCALL copyBuffer(
        IBufferResource* dst,
        Offset dstOffset,
        IBufferResource* src,
        Offset srcOffset,
        Size size) = 0;

    /// Copies texture from src to dst. If dstSubresource and srcSubresource has mipLevelCount = 0
    /// and layerCount = 0, the entire resource is being copied and dstOffset, srcOffset and extent
    /// arguments are ignored.
    virtual SLANG_NO_THROW void SLANG_MCALL copyTexture(
        ITextureResource* dst,
        ResourceState dstState,
        SubresourceRange dstSubresource,
        ITextureResource::Offset3D dstOffset,
        ITextureResource* src,
        ResourceState srcState,
        SubresourceRange srcSubresource,
        ITextureResource::Offset3D srcOffset,
        ITextureResource::Extents extent) = 0;

    /// Copies texture to a buffer. Each row is aligned to kTexturePitchAlignment.
    virtual SLANG_NO_THROW void SLANG_MCALL copyTextureToBuffer(
        IBufferResource* dst,
        Offset dstOffset,
        Size dstSize,
        Size dstRowStride,
        ITextureResource* src,
        ResourceState srcState,
        SubresourceRange srcSubresource,
        ITextureResource::Offset3D srcOffset,
        ITextureResource::Extents extent) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL uploadTextureData(
        ITextureResource* dst,
        SubresourceRange subResourceRange,
        ITextureResource::Offset3D offset,
        ITextureResource::Extents extent,
        ITextureResource::SubresourceData* subResourceData,
        GfxCount subResourceDataCount) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL
    uploadBufferData(IBufferResource* dst, Offset offset, Size size, void* data) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL textureBarrier(
        GfxCount count,
        ITextureResource* const* textures,
        ResourceState src,
        ResourceState dst) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL textureSubresourceBarrier(
        ITextureResource* texture,
        SubresourceRange subresourceRange,
        ResourceState src,
        ResourceState dst) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL bufferBarrier(
        GfxCount count,
        IBufferResource* const* buffers,
        ResourceState src,
        ResourceState dst) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL clearResourceView(
        IResourceView* view,
        ClearValue* clearValue,
        ClearResourceViewFlags::Enum flags) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL resolveResource(
        ITextureResource* source,
        ResourceState sourceState,
        SubresourceRange sourceRange,
        ITextureResource* dest,
        ResourceState destState,
        SubresourceRange destRange) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL resolveQuery(
        IQueryPool* queryPool,
        GfxIndex index,
        GfxCount count,
        IBufferResource* buffer,
        Offset offset) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL
    beginDebugEvent(const char* name, float rgbColor[3]) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL endDebugEvent() = 0;
    inline void textureBarrier(ITextureResource* texture, ResourceState src, ResourceState dst)
    {
        textureBarrier(1, &texture, src, dst);
    }
    inline void bufferBarrier(IBufferResource* buffer, ResourceState src, ResourceState dst)
    {
        bufferBarrier(1, &buffer, src, dst);
    }
};

class IRenderCommandEncoder : public IResourceCommandEncoder
{
    // {7A8D56D0-53E6-4AD6-85F7-D14DC110FDCE}
    SLANG_COM_INTERFACE(
        0x7a8d56d0,
        0x53e6,
        0x4ad6,
        {0x85, 0xf7, 0xd1, 0x4d, 0xc1, 0x10, 0xfd, 0xce})
public:
    // Sets the current pipeline state. This method returns a transient shader object for
    // writing shader parameters. This shader object will not retain any resources or
    // sub-shader-objects bound to it. The user must be responsible for ensuring that any
    // resources or shader objects that is set into `outRootShaderObject` stays alive during
    // the execution of the command buffer.
    virtual SLANG_NO_THROW Result SLANG_MCALL
    bindPipeline(IPipelineState* state, IShaderObject** outRootShaderObject) = 0;
    inline IShaderObject* bindPipeline(IPipelineState* state)
    {
        IShaderObject* rootObject = nullptr;
        SLANG_RETURN_NULL_ON_FAIL(bindPipeline(state, &rootObject));
        return rootObject;
    }

    // Sets the current pipeline state along with a pre-created mutable root shader object.
    virtual SLANG_NO_THROW Result SLANG_MCALL
    bindPipelineWithRootObject(IPipelineState* state, IShaderObject* rootObject) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL
    setViewports(GfxCount count, const Viewport* viewports) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL
    setScissorRects(GfxCount count, const ScissorRect* scissors) = 0;

    /// Sets the viewport, and sets the scissor rect to match the viewport.
    inline void setViewportAndScissor(Viewport const& viewport)
    {
        setViewports(1, &viewport);
        ScissorRect rect = {};
        rect.maxX = static_cast<gfx::Int>(viewport.extentX);
        rect.maxY = static_cast<gfx::Int>(viewport.extentY);
        setScissorRects(1, &rect);
    }

    virtual SLANG_NO_THROW void SLANG_MCALL setPrimitiveTopology(PrimitiveTopology topology) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL setVertexBuffers(
        GfxIndex startSlot,
        GfxCount slotCount,
        IBufferResource* const* buffers,
        const Offset* offsets) = 0;
    inline void setVertexBuffer(GfxIndex slot, IBufferResource* buffer, Offset offset = 0)
    {
        setVertexBuffers(slot, 1, &buffer, &offset);
    }

    virtual SLANG_NO_THROW void SLANG_MCALL
    setIndexBuffer(IBufferResource* buffer, Format indexFormat, Offset offset = 0) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    draw(GfxCount vertexCount, GfxIndex startVertex = 0) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    drawIndexed(GfxCount indexCount, GfxIndex startIndex = 0, GfxIndex baseVertex = 0) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL drawIndirect(
        GfxCount maxDrawCount,
        IBufferResource* argBuffer,
        Offset argOffset,
        IBufferResource* countBuffer = nullptr,
        Offset countOffset = 0) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL drawIndexedIndirect(
        GfxCount maxDrawCount,
        IBufferResource* argBuffer,
        Offset argOffset,
        IBufferResource* countBuffer = nullptr,
        Offset countOffset = 0) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL setStencilReference(uint32_t referenceValue) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL setSamplePositions(
        GfxCount samplesPerPixel,
        GfxCount pixelCount,
        const SamplePosition* samplePositions) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL drawInstanced(
        GfxCount vertexCount,
        GfxCount instanceCount,
        GfxIndex startVertex,
        GfxIndex startInstanceLocation) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL drawIndexedInstanced(
        GfxCount indexCount,
        GfxCount instanceCount,
        GfxIndex startIndexLocation,
        GfxIndex baseVertexLocation,
        GfxIndex startInstanceLocation) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL drawMeshTasks(int x, int y, int z) = 0;
};

class IComputeCommandEncoder : public IResourceCommandEncoder
{
    // {88AA9322-82F7-4FE6-A68A-29C7FE798737}
    SLANG_COM_INTERFACE(
        0x88aa9322,
        0x82f7,
        0x4fe6,
        {0xa6, 0x8a, 0x29, 0xc7, 0xfe, 0x79, 0x87, 0x37})

public:
    // Sets the current pipeline state. This method returns a transient shader object for
    // writing shader parameters. This shader object will not retain any resources or
    // sub-shader-objects bound to it. The user must be responsible for ensuring that any
    // resources or shader objects that is set into `outRooShaderObject` stays alive during
    // the execution of the command buffer.
    virtual SLANG_NO_THROW Result SLANG_MCALL
    bindPipeline(IPipelineState* state, IShaderObject** outRootShaderObject) = 0;
    inline IShaderObject* bindPipeline(IPipelineState* state)
    {
        IShaderObject* rootObject = nullptr;
        SLANG_RETURN_NULL_ON_FAIL(bindPipeline(state, &rootObject));
        return rootObject;
    }
    // Sets the current pipeline state along with a pre-created mutable root shader object.
    virtual SLANG_NO_THROW Result SLANG_MCALL
    bindPipelineWithRootObject(IPipelineState* state, IShaderObject* rootObject) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL dispatchCompute(int x, int y, int z) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    dispatchComputeIndirect(IBufferResource* cmdBuffer, Offset offset) = 0;
};

enum class AccelerationStructureCopyMode
{
    Clone,
    Compact
};

struct AccelerationStructureQueryDesc
{
    QueryType queryType;

    IQueryPool* queryPool;

    GfxIndex firstQueryIndex;
};

class IRayTracingCommandEncoder : public IResourceCommandEncoder
{
    SLANG_COM_INTERFACE(
        0x9a672b87,
        0x5035,
        0x45e3,
        {0x96, 0x7c, 0x1f, 0x85, 0xcd, 0xb3, 0x63, 0x4f})
public:
    virtual SLANG_NO_THROW void SLANG_MCALL buildAccelerationStructure(
        const IAccelerationStructure::BuildDesc& desc,
        GfxCount propertyQueryCount,
        AccelerationStructureQueryDesc* queryDescs) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL copyAccelerationStructure(
        IAccelerationStructure* dest,
        IAccelerationStructure* src,
        AccelerationStructureCopyMode mode) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL queryAccelerationStructureProperties(
        GfxCount accelerationStructureCount,
        IAccelerationStructure* const* accelerationStructures,
        GfxCount queryCount,
        AccelerationStructureQueryDesc* queryDescs) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL
    serializeAccelerationStructure(DeviceAddress dest, IAccelerationStructure* source) = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL
    deserializeAccelerationStructure(IAccelerationStructure* dest, DeviceAddress source) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    bindPipeline(IPipelineState* state, IShaderObject** outRootObject) = 0;
    // Sets the current pipeline state along with a pre-created mutable root shader object.
    virtual SLANG_NO_THROW Result SLANG_MCALL
    bindPipelineWithRootObject(IPipelineState* state, IShaderObject* rootObject) = 0;

    /// Issues a dispatch command to start ray tracing workload with a ray tracing pipeline.
    /// `rayGenShaderIndex` specifies the index into the shader table that identifies the ray
    /// generation shader.
    virtual SLANG_NO_THROW Result SLANG_MCALL dispatchRays(
        GfxIndex rayGenShaderIndex,
        IShaderTable* shaderTable,
        GfxCount width,
        GfxCount height,
        GfxCount depth) = 0;
};

class ICommandBuffer : public ISlangUnknown
{
public:
    // Only one encoder may be open at a time. User must call `ICommandEncoder::endEncoding`
    // before calling other `encode*Commands` methods.
    // Once `endEncoding` is called, the `ICommandEncoder` object becomes obsolete and is
    // invalid for further use. To continue recording, the user must request a new encoder
    // object by calling one of the `encode*Commands` methods again.
    virtual SLANG_NO_THROW void SLANG_MCALL encodeRenderCommands(
        IRenderPassLayout* renderPass,
        IFramebuffer* framebuffer,
        IRenderCommandEncoder** outEncoder) = 0;
    inline IRenderCommandEncoder* encodeRenderCommands(
        IRenderPassLayout* renderPass,
        IFramebuffer* framebuffer)
    {
        IRenderCommandEncoder* result;
        encodeRenderCommands(renderPass, framebuffer, &result);
        return result;
    }

    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeComputeCommands(IComputeCommandEncoder** outEncoder) = 0;
    inline IComputeCommandEncoder* encodeComputeCommands()
    {
        IComputeCommandEncoder* result;
        encodeComputeCommands(&result);
        return result;
    }

    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeResourceCommands(IResourceCommandEncoder** outEncoder) = 0;
    inline IResourceCommandEncoder* encodeResourceCommands()
    {
        IResourceCommandEncoder* result;
        encodeResourceCommands(&result);
        return result;
    }

    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeRayTracingCommands(IRayTracingCommandEncoder** outEncoder) = 0;
    inline IRayTracingCommandEncoder* encodeRayTracingCommands()
    {
        IRayTracingCommandEncoder* result;
        encodeRayTracingCommands(&result);
        return result;
    }

    virtual SLANG_NO_THROW void SLANG_MCALL close() = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) = 0;
};
#define SLANG_UUID_ICommandBuffer                          \
    {                                                      \
        0x5d56063f, 0x91d4, 0x4723,                        \
        {                                                  \
            0xa7, 0xa7, 0x7a, 0x15, 0xaf, 0x93, 0xeb, 0x48 \
        }                                                  \
    }

class ICommandBufferD3D12 : public ICommandBuffer
{
public:
    virtual SLANG_NO_THROW void SLANG_MCALL invalidateDescriptorHeapBinding() = 0;
    virtual SLANG_NO_THROW void SLANG_MCALL ensureInternalDescriptorHeapsBound() = 0;
};
#define SLANG_UUID_ICommandBufferD3D12                     \
    {                                                      \
        0xd56b7616, 0x6c14, 0x4841,                        \
        {                                                  \
            0x9d, 0x9c, 0x7b, 0x7f, 0xdb, 0x9f, 0xd9, 0xb8 \
        }                                                  \
    }

class ICommandQueue : public ISlangUnknown
{
public:
    enum class QueueType
    {
        Graphics
    };
    struct Desc
    {
        QueueType type;
    };

    // For D3D12, this is the pointer to the queue. For Vulkan, this is the queue itself.
    typedef uint64_t NativeHandle;

    virtual SLANG_NO_THROW const Desc& SLANG_MCALL getDesc() = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL executeCommandBuffers(
        GfxCount count,
        ICommandBuffer* const* commandBuffers,
        IFence* fenceToSignal,
        uint64_t newFenceValue) = 0;
    inline void executeCommandBuffer(
        ICommandBuffer* commandBuffer,
        IFence* fenceToSignal = nullptr,
        uint64_t newFenceValue = 0)
    {
        executeCommandBuffers(1, &commandBuffer, fenceToSignal, newFenceValue);
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL waitOnHost() = 0;

    /// Queues a device side wait for the given fences.
    virtual SLANG_NO_THROW Result SLANG_MCALL
    waitForFenceValuesOnDevice(GfxCount fenceCount, IFence** fences, uint64_t* waitValues) = 0;
};
#define SLANG_UUID_ICommandQueue                         \
    {                                                    \
        0x14e2bed0, 0xad0, 0x4dc8,                       \
        {                                                \
            0xb3, 0x41, 0x6, 0x3f, 0xe7, 0x2d, 0xbf, 0xe \
        }                                                \
    }

class ITransientResourceHeap : public ISlangUnknown
{
public:
    struct Flags
    {
        enum Enum
        {
            None = 0,
            AllowResizing = 0x1,
        };
    };
    struct Desc
    {
        Flags::Enum flags;
        Size constantBufferSize;
        GfxCount samplerDescriptorCount;
        GfxCount uavDescriptorCount;
        GfxCount srvDescriptorCount;
        GfxCount constantBufferDescriptorCount;
        GfxCount accelerationStructureDescriptorCount;
    };

    // Waits until GPU commands issued before last call to `finish()` has been completed, and resets
    // all transient resources holds by the heap.
    // This method must be called before using the transient heap to issue new GPU commands.
    // In most situations this method should be called at the beginning of each frame.
    virtual SLANG_NO_THROW Result SLANG_MCALL synchronizeAndReset() = 0;

    // Must be called when the application has done using this heap to issue commands. In most
    // situations this method should be called at the end of each frame.
    virtual SLANG_NO_THROW Result SLANG_MCALL finish() = 0;

    // Command buffers are one-time use. Once it is submitted to the queue via
    // `executeCommandBuffers` a command buffer is no longer valid to be used any more. Command
    // buffers must be closed before submission. The current D3D12 implementation has a limitation
    // that only one command buffer maybe recorded at a time. User must finish recording a command
    // buffer before creating another command buffer.
    virtual SLANG_NO_THROW Result SLANG_MCALL
    createCommandBuffer(ICommandBuffer** outCommandBuffer) = 0;
    inline ComPtr<ICommandBuffer> createCommandBuffer()
    {
        ComPtr<ICommandBuffer> result;
        SLANG_RETURN_NULL_ON_FAIL(createCommandBuffer(result.writeRef()));
        return result;
    }
};
#define SLANG_UUID_ITransientResourceHeap                 \
    {                                                     \
        0xcd48bd29, 0xee72, 0x41b8,                       \
        {                                                 \
            0xbc, 0xff, 0xa, 0x2b, 0x3a, 0xaa, 0x6d, 0xeb \
        }                                                 \
    }

class ITransientResourceHeapD3D12 : public ISlangUnknown
{
public:
    enum class DescriptorType
    {
        ResourceView,
        Sampler
    };
    virtual SLANG_NO_THROW Result SLANG_MCALL allocateTransientDescriptorTable(
        DescriptorType type,
        GfxCount count,
        Offset& outDescriptorOffset,
        void** outD3DDescriptorHeapHandle) = 0;
};
#define SLANG_UUID_ITransientResourceHeapD3D12             \
    {                                                      \
        0x9bc6a8bc, 0x5f7a, 0x454a,                        \
        {                                                  \
            0x93, 0xef, 0x3b, 0x10, 0x5b, 0xb7, 0x63, 0x7e \
        }                                                  \
    }

class ISwapchain : public ISlangUnknown
{
public:
    struct Desc
    {
        Format format;
        GfxCount width, height;
        GfxCount imageCount;
        ICommandQueue* queue;
        bool enableVSync;
    };
    virtual SLANG_NO_THROW const Desc& SLANG_MCALL getDesc() = 0;

    /// Returns the back buffer image at `index`.
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getImage(GfxIndex index, ITextureResource** outResource) = 0;

    /// Present the next image in the swapchain.
    virtual SLANG_NO_THROW Result SLANG_MCALL present() = 0;

    /// Returns the index of next back buffer image that will be presented in the next
    /// `present` call. If the swapchain is invalid/out-of-date, this method returns -1.
    virtual SLANG_NO_THROW int SLANG_MCALL acquireNextImage() = 0;

    /// Resizes the back buffers of this swapchain. All render target views and framebuffers
    /// referencing the back buffer images must be freed before calling this method.
    virtual SLANG_NO_THROW Result SLANG_MCALL resize(GfxCount width, GfxCount height) = 0;

    // Check if the window is occluded.
    virtual SLANG_NO_THROW bool SLANG_MCALL isOccluded() = 0;

    // Toggle full screen mode.
    virtual SLANG_NO_THROW Result SLANG_MCALL setFullScreenMode(bool mode) = 0;
};
#define SLANG_UUID_ISwapchain                             \
    {                                                     \
        0xbe91ba6c, 0x784, 0x4308,                        \
        {                                                 \
            0xa1, 0x0, 0x19, 0xc3, 0x66, 0x83, 0x44, 0xb2 \
        }                                                 \
    }

struct AdapterLUID
{
    uint8_t luid[16];

    bool operator==(const AdapterLUID& other) const
    {
        for (size_t i = 0; i < sizeof(AdapterLUID::luid); ++i)
            if (luid[i] != other.luid[i])
                return false;
        return true;
    }
    bool operator!=(const AdapterLUID& other) const { return !this->operator==(other); }
};

struct AdapterInfo
{
    // Descriptive name of the adapter.
    char name[128];

    // Unique identifier for the vendor (only available for D3D and Vulkan).
    uint32_t vendorID;

    // Unique identifier for the physical device among devices from the vendor (only available for
    // D3D and Vulkan)
    uint32_t deviceID;

    // Logically unique identifier of the adapter.
    AdapterLUID luid;
};

class AdapterList
{
public:
    AdapterList(ISlangBlob* blob)
        : m_blob(blob)
    {
    }

    const AdapterInfo* getAdapters() const
    {
        return reinterpret_cast<const AdapterInfo*>(m_blob ? m_blob->getBufferPointer() : nullptr);
    }

    GfxCount getCount() const
    {
        return (GfxCount)(m_blob ? m_blob->getBufferSize() / sizeof(AdapterInfo) : 0);
    }

private:
    ComPtr<ISlangBlob> m_blob;
};

struct DeviceLimits
{
    /// Maximum dimension for 1D textures.
    uint32_t maxTextureDimension1D;
    /// Maximum dimensions for 2D textures.
    uint32_t maxTextureDimension2D;
    /// Maximum dimensions for 3D textures.
    uint32_t maxTextureDimension3D;
    /// Maximum dimensions for cube textures.
    uint32_t maxTextureDimensionCube;
    /// Maximum number of texture layers.
    uint32_t maxTextureArrayLayers;

    /// Maximum number of vertex input elements in a graphics pipeline.
    uint32_t maxVertexInputElements;
    /// Maximum offset of a vertex input element in the vertex stream.
    uint32_t maxVertexInputElementOffset;
    /// Maximum number of vertex streams in a graphics pipeline.
    uint32_t maxVertexStreams;
    /// Maximum stride of a vertex stream.
    uint32_t maxVertexStreamStride;

    /// Maximum number of threads per thread group.
    uint32_t maxComputeThreadsPerGroup;
    /// Maximum dimensions of a thread group.
    uint32_t maxComputeThreadGroupSize[3];
    /// Maximum number of thread groups per dimension in a single dispatch.
    uint32_t maxComputeDispatchThreadGroups[3];

    /// Maximum number of viewports per pipeline.
    uint32_t maxViewports;
    /// Maximum viewport dimensions.
    uint32_t maxViewportDimensions[2];
    /// Maximum framebuffer dimensions.
    uint32_t maxFramebufferDimensions[3];

    /// Maximum samplers visible in a shader stage.
    uint32_t maxShaderVisibleSamplers;
};

struct DeviceInfo
{
    DeviceType deviceType;

    DeviceLimits limits;

    BindingStyle bindingStyle;

    ProjectionStyle projectionStyle;

    /// An projection matrix that ensures x, y mapping to pixels
    /// is the same on all targets
    float identityProjectionMatrix[16];

    /// The name of the graphics API being used by this device.
    const char* apiName = nullptr;

    /// The name of the graphics adapter.
    const char* adapterName = nullptr;

    /// The clock frequency used in timestamp queries.
    uint64_t timestampFrequency = 0;
};

enum class DebugMessageType
{
    Info,
    Warning,
    Error
};
enum class DebugMessageSource
{
    Layer,
    Driver,
    Slang
};
class IDebugCallback
{
public:
    virtual SLANG_NO_THROW void SLANG_MCALL
    handleMessage(DebugMessageType type, DebugMessageSource source, const char* message) = 0;
};

class IDevice : public ISlangUnknown
{
public:
    struct SlangDesc
    {
        slang::IGlobalSession* slangGlobalSession =
            nullptr; // (optional) A slang global session object. If null will create automatically.

        SlangMatrixLayoutMode defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;

        char const* const* searchPaths = nullptr;
        GfxCount searchPathCount = 0;

        slang::PreprocessorMacroDesc const* preprocessorMacros = nullptr;
        GfxCount preprocessorMacroCount = 0;

        const char* targetProfile = nullptr; // (optional) Target shader profile. If null this will
                                             // be set to platform dependent default.
        SlangFloatingPointMode floatingPointMode = SLANG_FLOATING_POINT_MODE_DEFAULT;
        SlangOptimizationLevel optimizationLevel = SLANG_OPTIMIZATION_LEVEL_DEFAULT;
        SlangTargetFlags targetFlags = kDefaultTargetFlags;
        SlangLineDirectiveMode lineDirectiveMode = SLANG_LINE_DIRECTIVE_MODE_DEFAULT;
    };

    struct ShaderCacheDesc
    {
        // The root directory for the shader cache. If not set, shader cache is disabled.
        const char* shaderCachePath = nullptr;
        // The maximum number of entries stored in the cache. By default, there is no limit.
        GfxCount maxEntryCount = 0;
    };

    struct InteropHandles
    {
        InteropHandle handles[3] = {};
    };

    struct Desc
    {
        // The underlying API/Platform of the device.
        DeviceType deviceType = DeviceType::Default;
        // The device's handles (if they exist) and their associated API. For D3D12, this contains a
        // single InteropHandle for the ID3D12Device. For Vulkan, the first InteropHandle is the
        // VkInstance, the second is the VkPhysicalDevice, and the third is the VkDevice. For CUDA,
        // this only contains a single value for the CUDADevice.
        InteropHandles existingDeviceHandles;
        // LUID of the adapter to use. Use getGfxAdapters() to get a list of available adapters.
        const AdapterLUID* adapterLUID = nullptr;
        // Number of required features.
        GfxCount requiredFeatureCount = 0;
        // Array of required feature names, whose size is `requiredFeatureCount`.
        const char** requiredFeatures = nullptr;
        // A command dispatcher object that intercepts and handles actual low-level API call.
        ISlangUnknown* apiCommandDispatcher = nullptr;
        // The slot (typically UAV) used to identify NVAPI intrinsics. If >=0 NVAPI is required.
        GfxIndex nvapiExtnSlot = -1;
        // Configurations for the shader cache.
        ShaderCacheDesc shaderCache = {};
        // Configurations for Slang compiler.
        SlangDesc slang = {};

        GfxCount extendedDescCount = 0;
        void** extendedDescs = nullptr;
    };

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getNativeDeviceHandles(InteropHandles* outHandles) = 0;

    virtual SLANG_NO_THROW bool SLANG_MCALL hasFeature(const char* feature) = 0;

    /// Returns a list of features supported by the renderer.
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getFeatures(const char** outFeatures, Size bufferSize, GfxCount* outFeatureCount) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getFormatSupportedResourceStates(Format format, ResourceStateSet* outStates) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getSlangSession(slang::ISession** outSlangSession) = 0;

    inline ComPtr<slang::ISession> getSlangSession()
    {
        ComPtr<slang::ISession> result;
        getSlangSession(result.writeRef());
        return result;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL createTransientResourceHeap(
        const ITransientResourceHeap::Desc& desc,
        ITransientResourceHeap** outHeap) = 0;
    inline ComPtr<ITransientResourceHeap> createTransientResourceHeap(
        const ITransientResourceHeap::Desc& desc)
    {
        ComPtr<ITransientResourceHeap> result;
        createTransientResourceHeap(desc, result.writeRef());
        return result;
    }

    /// Create a texture resource.
    ///
    /// If `initData` is non-null, then it must point to an array of
    /// `ITextureResource::SubresourceData` with one element for each
    /// subresource of the texture being created.
    ///
    /// The number of subresources in a texture is:
    ///
    ///     effectiveElementCount * mipLevelCount
    ///
    /// where the effective element count is computed as:
    ///
    ///     effectiveElementCount = (isArray ? arrayElementCount : 1) * (isCube ? 6 : 1);
    ///
    virtual SLANG_NO_THROW Result SLANG_MCALL createTextureResource(
        const ITextureResource::Desc& desc,
        const ITextureResource::SubresourceData* initData,
        ITextureResource** outResource) = 0;

    /// Create a texture resource. initData holds the initialize data to set the contents of the
    /// texture when constructed.
    inline SLANG_NO_THROW ComPtr<ITextureResource> createTextureResource(
        const ITextureResource::Desc& desc,
        const ITextureResource::SubresourceData* initData = nullptr)
    {
        ComPtr<ITextureResource> resource;
        SLANG_RETURN_NULL_ON_FAIL(createTextureResource(desc, initData, resource.writeRef()));
        return resource;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL createTextureFromNativeHandle(
        InteropHandle handle,
        const ITextureResource::Desc& srcDesc,
        ITextureResource** outResource) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL createTextureFromSharedHandle(
        InteropHandle handle,
        const ITextureResource::Desc& srcDesc,
        const Size size,
        ITextureResource** outResource) = 0;

    /// Create a buffer resource
    virtual SLANG_NO_THROW Result SLANG_MCALL createBufferResource(
        const IBufferResource::Desc& desc,
        const void* initData,
        IBufferResource** outResource) = 0;

    inline SLANG_NO_THROW ComPtr<IBufferResource> createBufferResource(
        const IBufferResource::Desc& desc,
        const void* initData = nullptr)
    {
        ComPtr<IBufferResource> resource;
        SLANG_RETURN_NULL_ON_FAIL(createBufferResource(desc, initData, resource.writeRef()));
        return resource;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL createBufferFromNativeHandle(
        InteropHandle handle,
        const IBufferResource::Desc& srcDesc,
        IBufferResource** outResource) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL createBufferFromSharedHandle(
        InteropHandle handle,
        const IBufferResource::Desc& srcDesc,
        IBufferResource** outResource) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createSamplerState(ISamplerState::Desc const& desc, ISamplerState** outSampler) = 0;

    inline ComPtr<ISamplerState> createSamplerState(ISamplerState::Desc const& desc)
    {
        ComPtr<ISamplerState> sampler;
        SLANG_RETURN_NULL_ON_FAIL(createSamplerState(desc, sampler.writeRef()));
        return sampler;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL createTextureView(
        ITextureResource* texture,
        IResourceView::Desc const& desc,
        IResourceView** outView) = 0;

    inline ComPtr<IResourceView> createTextureView(
        ITextureResource* texture,
        IResourceView::Desc const& desc)
    {
        ComPtr<IResourceView> view;
        SLANG_RETURN_NULL_ON_FAIL(createTextureView(texture, desc, view.writeRef()));
        return view;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL createBufferView(
        IBufferResource* buffer,
        IBufferResource* counterBuffer,
        IResourceView::Desc const& desc,
        IResourceView** outView) = 0;

    inline ComPtr<IResourceView> createBufferView(
        IBufferResource* buffer,
        IBufferResource* counterBuffer,
        IResourceView::Desc const& desc)
    {
        ComPtr<IResourceView> view;
        SLANG_RETURN_NULL_ON_FAIL(createBufferView(buffer, counterBuffer, desc, view.writeRef()));
        return view;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL createFramebufferLayout(
        IFramebufferLayout::Desc const& desc,
        IFramebufferLayout** outFrameBuffer) = 0;
    inline ComPtr<IFramebufferLayout> createFramebufferLayout(IFramebufferLayout::Desc const& desc)
    {
        ComPtr<IFramebufferLayout> fb;
        SLANG_RETURN_NULL_ON_FAIL(createFramebufferLayout(desc, fb.writeRef()));
        return fb;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createFramebuffer(IFramebuffer::Desc const& desc, IFramebuffer** outFrameBuffer) = 0;
    inline ComPtr<IFramebuffer> createFramebuffer(IFramebuffer::Desc const& desc)
    {
        ComPtr<IFramebuffer> fb;
        SLANG_RETURN_NULL_ON_FAIL(createFramebuffer(desc, fb.writeRef()));
        return fb;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL createRenderPassLayout(
        const IRenderPassLayout::Desc& desc,
        IRenderPassLayout** outRenderPassLayout) = 0;
    inline ComPtr<IRenderPassLayout> createRenderPassLayout(const IRenderPassLayout::Desc& desc)
    {
        ComPtr<IRenderPassLayout> rs;
        SLANG_RETURN_NULL_ON_FAIL(createRenderPassLayout(desc, rs.writeRef()));
        return rs;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL createSwapchain(
        ISwapchain::Desc const& desc,
        WindowHandle window,
        ISwapchain** outSwapchain) = 0;
    inline ComPtr<ISwapchain> createSwapchain(ISwapchain::Desc const& desc, WindowHandle window)
    {
        ComPtr<ISwapchain> swapchain;
        SLANG_RETURN_NULL_ON_FAIL(createSwapchain(desc, window, swapchain.writeRef()));
        return swapchain;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createInputLayout(IInputLayout::Desc const& desc, IInputLayout** outLayout) = 0;

    inline ComPtr<IInputLayout> createInputLayout(IInputLayout::Desc const& desc)
    {
        ComPtr<IInputLayout> layout;
        SLANG_RETURN_NULL_ON_FAIL(createInputLayout(desc, layout.writeRef()));
        return layout;
    }

    inline Result createInputLayout(
        Size vertexSize,
        InputElementDesc const* inputElements,
        GfxCount inputElementCount,
        IInputLayout** outLayout)
    {
        VertexStreamDesc streamDesc = {vertexSize, InputSlotClass::PerVertex, 0};

        IInputLayout::Desc inputLayoutDesc = {};
        inputLayoutDesc.inputElementCount = inputElementCount;
        inputLayoutDesc.inputElements = inputElements;
        inputLayoutDesc.vertexStreamCount = 1;
        inputLayoutDesc.vertexStreams = &streamDesc;
        return createInputLayout(inputLayoutDesc, outLayout);
    }

    inline ComPtr<IInputLayout> createInputLayout(
        Size vertexSize,
        InputElementDesc const* inputElements,
        GfxCount inputElementCount)
    {
        ComPtr<IInputLayout> layout;
        SLANG_RETURN_NULL_ON_FAIL(
            createInputLayout(vertexSize, inputElements, inputElementCount, layout.writeRef()));
        return layout;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createCommandQueue(const ICommandQueue::Desc& desc, ICommandQueue** outQueue) = 0;
    inline ComPtr<ICommandQueue> createCommandQueue(const ICommandQueue::Desc& desc)
    {
        ComPtr<ICommandQueue> queue;
        SLANG_RETURN_NULL_ON_FAIL(createCommandQueue(desc, queue.writeRef()));
        return queue;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL createShaderObject(
        slang::TypeReflection* type,
        ShaderObjectContainerType container,
        IShaderObject** outObject) = 0;

    inline ComPtr<IShaderObject> createShaderObject(slang::TypeReflection* type)
    {
        ComPtr<IShaderObject> object;
        SLANG_RETURN_NULL_ON_FAIL(
            createShaderObject(type, ShaderObjectContainerType::None, object.writeRef()));
        return object;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL createMutableShaderObject(
        slang::TypeReflection* type,
        ShaderObjectContainerType container,
        IShaderObject** outObject) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL createShaderObjectFromTypeLayout(
        slang::TypeLayoutReflection* typeLayout,
        IShaderObject** outObject) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL createMutableShaderObjectFromTypeLayout(
        slang::TypeLayoutReflection* typeLayout,
        IShaderObject** outObject) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createMutableRootShaderObject(IShaderProgram* program, IShaderObject** outObject) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createShaderTable(const IShaderTable::Desc& desc, IShaderTable** outTable) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL createProgram(
        const IShaderProgram::Desc& desc,
        IShaderProgram** outProgram,
        ISlangBlob** outDiagnosticBlob = nullptr) = 0;

    inline ComPtr<IShaderProgram> createProgram(const IShaderProgram::Desc& desc)
    {
        ComPtr<IShaderProgram> program;
        SLANG_RETURN_NULL_ON_FAIL(createProgram(desc, program.writeRef()));
        return program;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL createProgram2(
        const IShaderProgram::CreateDesc2& createDesc,
        IShaderProgram** outProgram,
        ISlangBlob** outDiagnosticBlob = nullptr) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL createGraphicsPipelineState(
        const GraphicsPipelineStateDesc& desc,
        IPipelineState** outState) = 0;

    inline ComPtr<IPipelineState> createGraphicsPipelineState(const GraphicsPipelineStateDesc& desc)
    {
        ComPtr<IPipelineState> state;
        SLANG_RETURN_NULL_ON_FAIL(createGraphicsPipelineState(desc, state.writeRef()));
        return state;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createComputePipelineState(const ComputePipelineStateDesc& desc, IPipelineState** outState) = 0;

    inline ComPtr<IPipelineState> createComputePipelineState(const ComputePipelineStateDesc& desc)
    {
        ComPtr<IPipelineState> state;
        SLANG_RETURN_NULL_ON_FAIL(createComputePipelineState(desc, state.writeRef()));
        return state;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL createRayTracingPipelineState(
        const RayTracingPipelineStateDesc& desc,
        IPipelineState** outState) = 0;

    /// Read back texture resource and stores the result in `outBlob`.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL readTextureResource(
        ITextureResource* resource,
        ResourceState state,
        ISlangBlob** outBlob,
        Size* outRowPitch,
        Size* outPixelSize) = 0;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    readBufferResource(IBufferResource* buffer, Offset offset, Size size, ISlangBlob** outBlob) = 0;

    /// Get the type of this renderer
    virtual SLANG_NO_THROW const DeviceInfo& SLANG_MCALL getDeviceInfo() const = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createQueryPool(const IQueryPool::Desc& desc, IQueryPool** outPool) = 0;


    virtual SLANG_NO_THROW Result SLANG_MCALL getAccelerationStructurePrebuildInfo(
        const IAccelerationStructure::BuildInputs& buildInputs,
        IAccelerationStructure::PrebuildInfo* outPrebuildInfo) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL createAccelerationStructure(
        const IAccelerationStructure::CreateDesc& desc,
        IAccelerationStructure** outView) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createFence(const IFence::Desc& desc, IFence** outFence) = 0;

    /// Wait on the host for the fences to signals.
    /// `timeout` is in nanoseconds, can be set to `kTimeoutInfinite`.
    virtual SLANG_NO_THROW Result SLANG_MCALL waitForFences(
        GfxCount fenceCount,
        IFence** fences,
        uint64_t* values,
        bool waitForAll,
        uint64_t timeout) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL getTextureAllocationInfo(
        const ITextureResource::Desc& desc,
        Size* outSize,
        Size* outAlignment) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL getTextureRowAlignment(Size* outAlignment) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL getCooperativeVectorProperties(
        CooperativeVectorProperties* properties,
        uint32_t* propertyCount) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL createShaderObject2(
        slang::ISession* slangSession,
        slang::TypeReflection* type,
        ShaderObjectContainerType container,
        IShaderObject** outObject) = 0;

    virtual SLANG_NO_THROW Result SLANG_MCALL createMutableShaderObject2(
        slang::ISession* slangSession,
        slang::TypeReflection* type,
        ShaderObjectContainerType container,
        IShaderObject** outObject) = 0;
};

#define SLANG_UUID_IDevice                                 \
    {                                                      \
        0x715bdf26, 0x5135, 0x11eb,                        \
        {                                                  \
            0xAE, 0x93, 0x02, 0x42, 0xAC, 0x13, 0x00, 0x02 \
        }                                                  \
    }

struct ShaderCacheStats
{
    GfxCount hitCount;
    GfxCount missCount;
    GfxCount entryCount;
};

// These are exclusively used to track hit/miss counts for shader cache entries. Entry hit and
// miss counts specifically indicate if the file containing relevant shader code was found in
// the cache, while the general hit and miss counts indicate whether the file was both found and
// up-to-date.
class IShaderCache : public ISlangUnknown
{
public:
    virtual SLANG_NO_THROW Result SLANG_MCALL clearShaderCache() = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL getShaderCacheStats(ShaderCacheStats* outStats) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL resetShaderCacheStats() = 0;
};

#define SLANG_UUID_IShaderCache                            \
    {                                                      \
        0x8eccc8ec, 0x5c04, 0x4a51,                        \
        {                                                  \
            0x99, 0x75, 0x13, 0xf8, 0xfe, 0xa1, 0x59, 0xf3 \
        }                                                  \
    }

class IPipelineCreationAPIDispatcher : public ISlangUnknown
{
public:
    virtual SLANG_NO_THROW Result SLANG_MCALL createComputePipelineState(
        IDevice* device,
        slang::IComponentType* program,
        void* pipelineDesc,
        void** outPipelineState) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL createGraphicsPipelineState(
        IDevice* device,
        slang::IComponentType* program,
        void* pipelineDesc,
        void** outPipelineState) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL createMeshPipelineState(
        IDevice* device,
        slang::IComponentType* program,
        void* pipelineDesc,
        void** outPipelineState) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    beforeCreateRayTracingState(IDevice* device, slang::IComponentType* program) = 0;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    afterCreateRayTracingState(IDevice* device, slang::IComponentType* program) = 0;
};
#define SLANG_UUID_IPipelineCreationAPIDispatcher         \
    {                                                     \
        0xc3d5f782, 0xeae1, 0x4da6,                       \
        {                                                 \
            0xab, 0x40, 0x75, 0x32, 0x31, 0x2, 0xb7, 0xdc \
        }                                                 \
    }

#define SLANG_UUID_IVulkanPipelineCreationAPIDispatcher    \
    {                                                      \
        0x4fcf1274, 0x8752, 0x4743,                        \
        {                                                  \
            0xb3, 0x51, 0x47, 0xcb, 0x83, 0x71, 0xef, 0x99 \
        }                                                  \
    }

// Global public functions

extern "C"
{
    /// Checks if format is compressed
    SLANG_GFX_API bool SLANG_MCALL gfxIsCompressedFormat(Format format);

    /// Checks if format is typeless
    SLANG_GFX_API bool SLANG_MCALL gfxIsTypelessFormat(Format format);

    /// Gets information about the format
    SLANG_GFX_API SlangResult SLANG_MCALL gfxGetFormatInfo(Format format, FormatInfo* outInfo);

    /// Gets a list of available adapters for a given device type
    SLANG_GFX_API SlangResult SLANG_MCALL
    gfxGetAdapters(DeviceType type, ISlangBlob** outAdaptersBlob);

    /// Given a type returns a function that can construct it, or nullptr if there isn't one
    SLANG_GFX_API SlangResult SLANG_MCALL
    gfxCreateDevice(const IDevice::Desc* desc, IDevice** outDevice);

    /// Reports current set of live objects in gfx.
    /// Currently this only calls D3D's ReportLiveObjects.
    SLANG_GFX_API SlangResult SLANG_MCALL gfxReportLiveObjects();

    /// Sets a callback for receiving debug messages.
    /// The layer does not hold a strong reference to the callback object.
    /// The user is responsible for holding the callback object alive.
    SLANG_GFX_API SlangResult SLANG_MCALL gfxSetDebugCallback(IDebugCallback* callback);

    /// Enables debug layer. The debug layer will check all `gfx` calls and verify that uses are
    /// valid.
    SLANG_GFX_API void SLANG_MCALL gfxEnableDebugLayer();

    SLANG_GFX_API const char* SLANG_MCALL gfxGetDeviceTypeName(DeviceType type);
}

/// Gets a list of available adapters for a given device type
inline AdapterList gfxGetAdapters(DeviceType type)
{
    ComPtr<ISlangBlob> blob;
    gfxGetAdapters(type, blob.writeRef());
    return AdapterList(blob);
}

// Extended descs.
struct D3D12ExperimentalFeaturesDesc
{
    StructType structType = StructType::D3D12ExperimentalFeaturesDesc;
    uint32_t numFeatures;
    const void* featureIIDs;
    void* configurationStructs;
    uint32_t* configurationStructSizes;
};

struct D3D12DeviceExtendedDesc
{
    StructType structType = StructType::D3D12DeviceExtendedDesc;
    const char* rootParameterShaderAttributeName = nullptr;
    bool debugBreakOnD3D12Error = false;
    uint32_t highestShaderModel = 0;
};

struct SlangSessionExtendedDesc
{
    StructType structType = StructType::SlangSessionExtendedDesc;
    uint32_t compilerOptionEntryCount = 0;
    slang::CompilerOptionEntry* compilerOptionEntries = nullptr;
};

/// Whether to enable ray tracing validation (currently only Vulkan - D3D requires app layer to use
/// NVAPI)
struct RayTracingValidationDesc
{
    StructType structType = StructType::RayTracingValidationDesc;
    bool enableRaytracingValidation = false;
};

} // namespace gfx
