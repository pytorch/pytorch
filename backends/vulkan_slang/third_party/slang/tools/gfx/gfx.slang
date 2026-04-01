import slang;

public namespace gfx
{
public typedef slang.Result Result;

public typedef intptr_t Int;
public typedef uintptr_t UInt;
public typedef uint64_t DeviceAddress;
public typedef int GfxIndex;
public typedef int GfxCount;
public typedef intptr_t Size;
public typedef intptr_t Offset;

public const uint64_t kTimeoutInfinite = 0xFFFFFFFFFFFFFFFF;

public enum class StructType
{
    D3D12ExtendedDesc,
};

public enum class StageType
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

public enum class DeviceType
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
    CountOf,
};

public enum class ProjectionStyle
{
    Unknown,
    OpenGl,
    DirectX,
    Vulkan,
    Metal,
    CountOf,
};

public enum class BindingStyle
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

public enum class AccessFlag
{
    None,
    Read,
    Write,
};

public static const GfxCount kMaxRenderTargetCount = 8;

// Defines how linking should be performed for a shader program.
public enum class LinkingStyle
{
    // Compose all entry-points in a single program, then compile all entry-points together with the same
    // set of root shader arguments.
    SingleProgram,

    // Link and compile each entry-point individually, potentially with different specializations.
    SeparateEntryPointCompilation
};

public enum class ShaderModuleSourceType
{
    SlangSource,           // a slang source string in memory.
    SlangModuleBinary,     // a slang module binary code in memory.
    SlangSourceFile,       // a slang source from file.
    SlangModuleBinaryFile, // a slang module binary code from file.
};

public struct ShaderProgramDesc2
{
    public ShaderModuleSourceType sourceType = ShaderModuleSourceType::SlangSource;
    public void *sourceData = nullptr;
    public Size sourceDataSize = 0;

    // Number of entry points to include in the shader program. 0 means include all entry points
    // defined in the module.
    public GfxCount entryPointCount = 0;
    // Names of entry points to include in the shader program. The size of the array must be
    // `entryPointCount`.
    public NativeString* entryPointNames = nullptr;
};

[COM("9d32d0ad-915c-4ffd-91e2-508554a04a76")]
public interface IShaderProgram
{
    public slang::TypeReflection* findTypeByName(NativeString name);
};

public enum class Format
{
    // D3D formats omitted: 19-22, 44-47, 65-66, 68-70, 73, 76, 79, 82, 88-89, 92-94, 97, 100-114
    // These formats are omitted due to lack of a corresponding Vulkan format. D24_UNORM_S8_UINT (DXGI_FORMAT 45)
    // has a matching Vulkan format but is also omitted as it is only supported by Nvidia.
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

    R64_UINT,

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

    R64_SINT,

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

    _Count,
};

public struct FormatInfo
{
    public GfxCount channelCount; ///< The amount of channels in the format. Only set if the channelType is set
    public uint8_t channelType;   ///< One of SlangScalarType None if type isn't made up of elements of type. TODO: Change to uint32_t?

    public Size blockSizeInBytes;   ///< The size of a block in bytes.
    public GfxCount pixelsPerBlock; ///< The number of pixels contained in a block.
    public GfxCount blockWidth;     ///< The width of a block in pixels.
    public GfxCount blockHeight;    ///< The height of a block in pixels.
};

public enum class InputSlotClass
{
    PerVertex, PerInstance
};

public struct InputElementDesc
{
    public NativeString semanticName; ///< The name of the corresponding parameter in shader code.
    public GfxIndex semanticIndex;   ///< The index of the corresponding parameter in shader code. Only needed if multiple parameters share a semantic name.
    public Format format;            ///< The format of the data being fetched for this element.
    public Offset offset;            ///< The offset in bytes of this element from the start of the corresponding chunk of vertex stream data.
    public GfxIndex bufferSlotIndex; ///< The index of the vertex stream to fetch this element's data from.
};

public struct VertexStreamDesc
{
    public Size stride;                   ///< The stride in bytes for this vertex stream.
    public InputSlotClass slotClass;      ///< Whether the stream contains per-vertex or per-instance data.
    public GfxCount instanceDataStepRate; ///< How many instances to draw per chunk of data.
};

public enum class PrimitiveType
{
    Point, Line, Triangle, Patch
};

public enum class PrimitiveTopology
{
    TriangleList, TriangleStrip, PointList, LineList, LineStrip
};

public enum class ResourceState
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
    _Count
};

public struct ResourceStateSet
{
    public uint64_t m_bitFields;

    [mutating]
    public void add(ResourceState state) { m_bitFields |= (1LL << (uint32_t)state); }

    public bool contains(ResourceState state) { return (m_bitFields & (1LL << (uint32_t)state)) != 0; }
    public __init() { m_bitFields = 0; }
    public __init(ResourceState state) { add(state); }
};

public ResourceStateSet operator &(ResourceStateSet val, ResourceStateSet that)
{
    ResourceStateSet result;
    result.m_bitFields = val.m_bitFields & that.m_bitFields;
    return result;
}

/// Describes how memory for the resource should be allocated for CPU access.
public enum class MemoryType
{
    DeviceLocal,
    Upload,
    ReadBack,
};

public enum class InteropHandleAPI
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

public struct InteropHandle
{
    public InteropHandleAPI api = InteropHandleAPI::Unknown;
    public uint64_t handleValue = 0LLU;
};

// Declare opaque type
public struct InputLayoutDesc
{
    public InputElementDesc *inputElements;
    public GfxCount inputElementCount;
    public VertexStreamDesc *vertexStreams;
    public GfxCount vertexStreamCount;
};

[COM("45223711-a84b-455c-befa-4937421e8e2e")]
public interface IInputLayout
{   
};

/// The type of resource.
/// NOTE! The order needs to be such that all texture types are at or after Texture1D (otherwise isTexture won't work correctly)
public enum class ResourceType
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
public struct ResourceDescBase
{
    public ResourceType type = ResourceType::Unknown;
    public ResourceState defaultState = ResourceState::Undefined;
    public ResourceStateSet allowedStates = {};
    public MemoryType memoryType = MemoryType::DeviceLocal;
    public InteropHandle existingHandle = {};
    public bool isShared = false;
};

[COM("a0e39f34-8398-4522-95c2-ebc0f984ef3f")]
public interface IResource
{
    public ResourceType getType();
    public Result getNativeResourceHandle(out InteropHandle outHandle);
    public Result getSharedHandle(out InteropHandle outHandle);
    public Result setDebugName(NativeString name);
    public NativeString getDebugName();
};

public struct MemoryRange
{
    // TODO: Change to Offset/Size?
    public uint64_t offset;
    public uint64_t size;
};

public struct BufferResourceDesc : ResourceDescBase
{
    public Size sizeInBytes = 0; ///< Total size in bytes
    public Size elementSize = 0; ///< Get the element stride. If > 0, this is a structured buffer
    public Format format = Format::Unknown;
};

[COM("1b274efe-5e37-492b-826e-7ee7e8f5a49b")]
public interface IBufferResource : IResource
{
    public BufferResourceDesc *getDesc();
    public DeviceAddress getDeviceAddress();
    public Result map(MemoryRange *rangeToRead, void **outPointer);
    public Result unmap(MemoryRange* writtenRange);
};

public struct DepthStencilClearValue
{
    public float depth = 1.0f;
    public uint32_t stencil = 0;
};

public struct ColorClearValue
{
    public float4 values;

    [mutating]
    public void setValue(uint4 uintVal)
    {
        values = reinterpret<float4, uint4>(uintVal);
    }

    [mutating]
    public void setValue(float4 floatVal)
    {
        values = floatVal;
    }
};

public struct ClearValue
{
    public ColorClearValue color;
    public DepthStencilClearValue depthStencil;
};

public struct BufferRange
{
    public Offset offset;   ///< Offset in bytes.
    public Size size;       ///< Size in bytes.
};

public enum class TextureAspect : uint32_t
{
    Default = 0,
    Color = 0x00000001,
    Depth = 0x00000002,
    Stencil = 0x00000004,
    MetaData = 0x00000008,
    Plane0 = 0x00000010,
    Plane1 = 0x00000020,
    Plane2 = 0x00000040,

    DepthStencil = 0x6,
};

public struct SubresourceRange
{
    public TextureAspect aspectMask;
    public GfxIndex mipLevel;
    public GfxCount mipLevelCount;
    public GfxIndex baseArrayLayer; // For Texture3D, this is WSlice.
    public GfxCount layerCount;     // For cube maps, this is a multiple of 6.
};

public static const Size kRemainingTextureSize = 0xFFFFFFFF;
public struct TextureResourceSampleDesc
{
    public GfxCount numSamples; ///< Number of samples per pixel
    public int quality;         ///< The quality measure for the samples
};

public struct TextureResourceDesc : ResourceDescBase
{
    public int3 size;

    public GfxCount arraySize = 0; ///< Array size

    public GfxCount numMipLevels = 0;         ///< Number of mip levels - if 0 will create all mip levels
    public Format format;             ///< The resources format
    public TextureResourceSampleDesc sampleDesc; ///< How the resource is sampled
    public ClearValue* optimalClearValue;
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
public struct SubresourceData
{
    /// Pointer to texel data for the subresource tensor.
    public void *data;

    /// Stride in bytes between rows of the subresource tensor.
    ///
    /// This is the number of bytes to add to a pointer to a texel
    /// at (X,Y,Z) to get to a texel at (X,Y+1,Z).
    ///
    /// Devices may not support all possible values for `strideY`.
    /// In particular, they may only support strictly positive strides.
    ///
    public gfx::Size strideY;

    /// Stride in bytes between layers of the subresource tensor.
    ///
    /// This is the number of bytes to add to a pointer to a texel
    /// at (X,Y,Z) to get to a texel at (X,Y,Z+1).
    ///
    /// Devices may not support all possible values for `strideZ`.
    /// In particular, they may only support strictly positive strides.
    ///
    public gfx::Size strideZ;
};

[COM("cf88a31c-6187-46c5-a4b7-eb-58-c7-33-40-17")]
public interface ITextureResource : IResource
{
    public TextureResourceDesc* getDesc();
};

public enum class ComparisonFunc : uint8_t
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

public enum class TextureFilteringMode
{
    Point,
    Linear,
};

public enum class TextureAddressingMode
{
    Wrap,
    ClampToEdge,
    ClampToBorder,
    MirrorRepeat,
    MirrorOnce,
};

public enum class TextureReductionOp
{
    Average,
    Comparison,
    Minimum,
    Maximum,
};

public struct SamplerStateDesc
{
    public TextureFilteringMode minFilter;
    public TextureFilteringMode magFilter;
    public TextureFilteringMode mipFilter;
    public TextureReductionOp reductionOp;
    public TextureAddressingMode addressU;
    public TextureAddressingMode addressV;
    public TextureAddressingMode addressW;
    public float mipLODBias;
    public uint32_t maxAnisotropy;
    public ComparisonFunc comparisonFunc;
    public float4 borderColor;
    public float minLOD;
    public float maxLOD;
    public __init()
    {
        minFilter = TextureFilteringMode::Linear;
        magFilter = TextureFilteringMode::Linear;
        mipFilter = TextureFilteringMode::Linear;
        reductionOp = TextureReductionOp::Average;
        addressU = TextureAddressingMode::Wrap;
        addressV = TextureAddressingMode::Wrap;
        addressW = TextureAddressingMode::Wrap;
        mipLODBias = 0.0f;
        maxAnisotropy = 1;
        comparisonFunc = ComparisonFunc::Never;
        borderColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
        minLOD = -float.maxValue;
        maxLOD = float.maxValue;
    }
};

[COM("8b8055df-9377-401d-91ff-3f-a3-bf-66-64-f4")]
public interface ISamplerState
{
    /// Returns a native API handle representing this sampler state object.
    /// When using D3D12, this will be a D3D12_CPU_DESCRIPTOR_HANDLE.
    /// When using Vulkan, this will be a VkSampler.
    public Result getNativeHandle(InteropHandle *outNativeHandle);
};

public enum class ResourceViewType
{
    Unknown,

    RenderTarget,
    DepthStencil,
    ShaderResource,
    UnorderedAccess,
    AccelerationStructure,

    CountOf_,
};

public struct RenderTargetDesc
{
    // The resource shape of this render target view.
    public ResourceType shape;
};

public struct ResourceViewDesc
{
    public ResourceViewType type;
    public Format format;

    // Required fields for `RenderTarget` and `DepthStencil` views.
    public RenderTargetDesc renderTarget;
    // Specifies the range of a texture resource for a ShaderRsource/UnorderedAccess/RenderTarget/DepthStencil view.
    public SubresourceRange subresourceRange;
    // Specifies the range of a buffer resource for a ShaderResource/UnorderedAccess view.
    public BufferRange bufferRange;
};

[COM("7b6c4926-0884-408c-ad8a-50-3a-8e-23-98-a4")]
public interface IResourceView
{
    public ResourceViewDesc* getViewDesc();

    /// Returns a native API handle representing this resource view object.
    /// When using D3D12, this will be a D3D12_CPU_DESCRIPTOR_HANDLE or a buffer device address depending
    /// on the type of the resource view.
    /// When using Vulkan, this will be a VkImageView, VkBufferView, VkAccelerationStructure or a VkBuffer
    /// depending on the type of the resource view.
    public Result getNativeHandle(InteropHandle *outNativeHandle);
};

public enum class AccelerationStructureKind
{
    TopLevel,
    BottomLevel
};

// The public enum values are intentionally consistent with
// D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS.
public enum AccelerationStructureBuildFlags
{
    None,
    AllowUpdate = 1,
    AllowCompaction = 2,
    PreferFastTrace = 4,
    PreferFastBuild = 8,
    MinimizeMemory = 16,
    PerformUpdate = 32
};

public enum class GeometryType
{
    Triangles, ProcedurePrimitives
};

public struct GeometryFlags
{
    // The public enum values are intentionally consistent with
    // D3D12_RAYTRACING_GEOMETRY_FLAGS.
    public enum Enum
    {
        None,
        Opaque = 1,
        NoDuplicateAnyHitInvocation = 2
    };
};

public struct TriangleDesc
{
    public DeviceAddress transform3x4;
    public Format indexFormat;
    public Format vertexFormat;
    public GfxCount indexCount;
    public GfxCount vertexCount;
    public DeviceAddress indexData;
    public DeviceAddress vertexData;
    public Size vertexStride;
};

public struct ProceduralAABB
{
    public float minX;
    public float minY;
    public float minZ;
    public float maxX;
    public float maxY;
    public float maxZ;
};

public struct ProceduralAABBDesc
{
    /// Number of AABBs.
    public GfxCount count;

    /// Pointer to an array of `ProceduralAABB` values in device memory.
    public DeviceAddress data;

    /// Stride in bytes of the AABB values array.
    public Size stride;
};

public struct GeometryDesc
{
    public GeometryType type;
    public GeometryFlags::Enum flags;
    public TriangleDesc triangles;
    public property ProceduralAABBDesc proceduralAABBs
    {
        get { return reinterpret<ProceduralAABBDesc, TriangleDesc>(triangles); }
        set { triangles = reinterpret<TriangleDesc, ProceduralAABBDesc>(newValue); }
    }
};

// The public enum values are kept consistent with D3D12_RAYTRACING_INSTANCE_FLAGS
// and VkGeometryInstanceFlagBitsKHR.
public enum GeometryInstanceFlags
{
    None = 0,
    TriangleFacingCullDisable = 0x00000001,
    TriangleFrontCounterClockwise = 0x00000002,
    ForceOpaque = 0x00000004,
    NoOpaque = 0x00000008
};

// TODO: Should any of these be changed?
// The layout of this public struct is intentionally consistent with D3D12_RAYTRACING_INSTANCE_DESC
// and VkAccelerationStructureInstanceKHR.
public struct InstanceDesc
{
    public float transform[3][4];
    public uint32_t instanceID24_mask8;
    public property uint32_t instanceID { get { return instanceID24_mask8 & 0xFFFFFF; } set { instanceID24_mask8 = (instanceID24_mask8 & 0xFF000000) | (newValue & 0xFFFFFF); } }
    public property uint32_t instanceMask { get { return instanceID24_mask8 >> 24; } set { instanceID24_mask8 = (newValue << 24) | (instanceID24_mask8 & 0x00FFFFFF); } }

    public uint32_t instanceContributionToHitGroupIndex24_flags8;
    public property uint32_t instanceContributionToHitGroupIndex
    {
        get { return instanceContributionToHitGroupIndex24_flags8 & 0xFFFFFF; }
        set { instanceContributionToHitGroupIndex24_flags8 = (instanceContributionToHitGroupIndex24_flags8 & 0xFF000000) | (newValue & 0xFFFFFF); }
    }
    public property GeometryInstanceFlags flags
    {
        get { return (GeometryInstanceFlags)(instanceContributionToHitGroupIndex24_flags8 >> 24); }
        set { instanceContributionToHitGroupIndex24_flags8 = ((uint32_t)newValue << 24) | (instanceContributionToHitGroupIndex24_flags8 & 0x00FFFFFF); }
    }
    public DeviceAddress accelerationStructure;
};

public struct AccelerationStructurePrebuildInfo
{
    public Size resultDataMaxSize;
    public Size scratchDataSize;
    public Size updateScratchDataSize;
};

public struct AccelerationStructureBuildInputs
{
    public AccelerationStructureKind kind;

    public AccelerationStructureBuildFlags flags;

    public GfxCount descCount;

    /// Array of `InstanceDesc` values in device memory.
    /// Used when `kind` is `TopLevel`.
    public DeviceAddress instanceDescs;

    /// Array of `GeometryDesc` values.
    /// Used when `kind` is `BottomLevel`.
    public GeometryDesc *geometryDescs;
};

public struct AccelerationStructureCreateDesc
{
    public AccelerationStructureKind kind;
    public NativeRef<IBufferResource> buffer;
    public Offset offset;
    public Size size;
};

public struct AccelerationStructureBuildDesc
{
    public AccelerationStructureBuildInputs inputs;
    public NativeRef<IAccelerationStructure> source;
    public NativeRef<IAccelerationStructure> dest;
    public DeviceAddress scratchData;
};

[COM("a5cdda3c-1d4e-4df7-8ef2-b7-3f-ce-04-de-3b")]
public interface IAccelerationStructure : IResourceView
{
    public DeviceAddress getDeviceAddress();
};

public struct FenceDesc
{
    public uint64_t initialValue;
    public bool isShared;
};

[COM("7fe1c283-d3f4-48ed-aaf3-01-51-96-4e-7c-b5")]
public interface IFence
{
    /// Returns the currently signaled value on the device.
    public Result getCurrentValue(uint64_t *outValue);

    /// Signals the fence from the host with the specified value.
    public Result setCurrentValue(uint64_t value);

    public Result getSharedHandle(InteropHandle *outHandle);
    public Result getNativeHandle(InteropHandle *outNativeHandle);
};

public struct ShaderOffset
{
    public Int uniformOffset = 0; // TODO: Change to Offset?
    public GfxIndex bindingRangeIndex = 0;
    public GfxIndex bindingArrayIndex = 0;
}

public enum class ShaderObjectContainerType
{
    None, Array, StructuredBuffer
};

[COM("c1fa997e-5ca2-45ae-9bcb-c4-35-9e-85-05-85")]
public interface IShaderObject
{
    public slang::TypeLayoutReflection* getElementTypeLayout();
    public ShaderObjectContainerType getContainerType();
    public GfxCount getEntryPointCount();
    public Result getEntryPoint(GfxIndex index, out Optional<IShaderObject> entryPoint);
    public Result setData(ShaderOffset *offset, void *data, Size size);
    public Result getObject(ShaderOffset *offset, out Optional<IShaderObject> object);
    public Result setObject(ShaderOffset* offset, IShaderObject object);
    public Result setResource(ShaderOffset* offset, IResourceView resourceView);
    public Result setSampler(ShaderOffset* offset, ISamplerState sampler);
    public Result setCombinedTextureSampler(ShaderOffset* offset, IResourceView textureView, ISamplerState sampler);

    /// Manually overrides the specialization argument for the sub-object binding at `offset`.
    /// Specialization arguments are passed to the shader compiler to specialize the type
    /// of interface-typed shader parameters.
    public Result setSpecializationArgs(
        ShaderOffset* offset,
        slang::SpecializationArg *args,
        GfxCount count);

    public Result getCurrentVersion(
        ITransientResourceHeap transientHeap,
        out IShaderObject outObject);

    public void* getRawData();

    public Size getSize();

    /// Use the provided constant buffer instead of the internally created one.
    public Result setConstantBufferOverride(IBufferResource constantBuffer);
};

public enum class StencilOp : uint8_t
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

public enum class FillMode : uint8_t
{
    Solid,
    Wireframe,
};

public enum class CullMode : uint8_t
{
    None,
    Front,
    Back,
};

public enum class FrontFaceMode : uint8_t
{
    CounterClockwise,
    Clockwise,
};

public struct DepthStencilOpDesc
{
    public StencilOp stencilFailOp = StencilOp::Keep;
    public StencilOp stencilDepthFailOp = StencilOp::Keep;
    public StencilOp stencilPassOp = StencilOp::Keep;
    public ComparisonFunc stencilFunc = ComparisonFunc::Always;
    public __init()
    {
        stencilFailOp = StencilOp::Keep;
        stencilDepthFailOp = StencilOp::Keep;
        stencilPassOp = StencilOp::Keep;
        stencilFunc = ComparisonFunc::Always;
    }
};

public struct DepthStencilDesc
{
    public bool depthTestEnable = false;
    public bool depthWriteEnable = true;
    public ComparisonFunc depthFunc = ComparisonFunc::Less;

    public bool stencilEnable = false;
    public uint32_t stencilReadMask = 0xFFFFFFFF;
    public uint32_t stencilWriteMask = 0xFFFFFFFF;
    public DepthStencilOpDesc frontFace;
    public DepthStencilOpDesc backFace;

    public uint32_t stencilRef = 0;

    public __init()
    {
        depthTestEnable = false;
        depthWriteEnable = true;
        depthFunc = ComparisonFunc::Less;
        stencilEnable = false;
        stencilReadMask = 0xFFFFFFFF;
        stencilWriteMask = 0xFFFFFFFF;
        stencilRef = 0;
    }
};

public struct RasterizerDesc
{
    public FillMode fillMode = FillMode::Solid;
    public CullMode cullMode = CullMode::None;
    public FrontFaceMode frontFace = FrontFaceMode::CounterClockwise;
    public int32_t depthBias = 0;
    public float depthBiasClamp = 0.0f;
    public float slopeScaledDepthBias = 0.0f;
    public bool depthClipEnable = true;
    public bool scissorEnable = false;
    public bool multisampleEnable = false;
    public bool antialiasedLineEnable = false;
    public bool enableConservativeRasterization = false;
    public uint32_t forcedSampleCount = 0;

    public __init()
    {
        fillMode = FillMode::Solid;
        cullMode = CullMode::None;
        frontFace = FrontFaceMode::CounterClockwise;
        depthBias = 0;
        depthBiasClamp = 0.0f;
        slopeScaledDepthBias = 0.0f;
        depthClipEnable = true;
        scissorEnable = false;
        multisampleEnable = false;
        antialiasedLineEnable = false;
        enableConservativeRasterization = false;
        forcedSampleCount = 0;
    }
};

public enum class LogicOp
{
    NoOp,
};

public enum class BlendOp
{
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
};

public enum class BlendFactor
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

public enum RenderTargetWriteMask
{
    EnableNone = 0,
    EnableRed = 0x01,
    EnableGreen = 0x02,
    EnableBlue = 0x04,
    EnableAlpha = 0x08,
    EnableAll = 0x0F,
};

public struct AspectBlendDesc
{
    public BlendFactor srcFactor = BlendFactor::One;
    public BlendFactor dstFactor = BlendFactor::Zero;
    public BlendOp op = BlendOp::Add;

    __init()
    {
        srcFactor = BlendFactor::One;
        dstFactor = BlendFactor::Zero;
        op = BlendOp::Add;
    }
};

public struct TargetBlendDesc
{
    public AspectBlendDesc color;
    public AspectBlendDesc alpha;
    public bool enableBlend;
    public LogicOp logicOp;
    public RenderTargetWriteMask writeMask;
    public __init()
    {
        enableBlend = false;
        logicOp = LogicOp::NoOp;
        writeMask = RenderTargetWriteMask::EnableAll;
    }
};

public struct BlendDesc
{
    public TargetBlendDesc targets[kMaxRenderTargetCount] = {};
    public GfxCount targetCount = 0;

    public bool alphaToCoverageEnable = false;
};

public struct FramebufferTargetLayout
{
    public Format format;
    public GfxCount sampleCount;
};

public struct FramebufferLayoutDesc
{
    public GfxCount renderTargetCount;
    public FramebufferTargetLayout *renderTargets;
    public FramebufferTargetLayout *depthStencil;
};

[COM("0a838785-c13a-4832-ad88-64-06-b5-4b-5e-ba")]
public interface IFramebufferLayout
{
};

public struct GraphicsPipelineStateDesc
{
    public NativeRef<IShaderProgram> program;

    public NativeRef<IInputLayout> inputLayout;
    public NativeRef<IFramebufferLayout> framebufferLayout;
    public PrimitiveType primitiveType;
    public DepthStencilDesc depthStencil;
    public RasterizerDesc rasterizer;
    public BlendDesc blend;

    public __init()
    {
        program = {IShaderProgram()};
        inputLayout = {IInputLayout()};
        framebufferLayout = {IFramebufferLayout()};
        primitiveType = PrimitiveType::Triangle;
        depthStencil = {};
        rasterizer = {};
        blend = {};
    }
};

public struct ComputePipelineStateDesc
{
    public NativeRef<IShaderProgram> program;
    public void *d3d12RootSignatureOverride;
};

public enum RayTracingPipelineFlags
{
    None = 0,
    SkipTriangles = 1,
    SkipProcedurals = 2,
};

public struct HitGroupDesc
{
    public NativeString hitGroupName;
    public NativeString closestHitEntryPoint;
    public NativeString anyHitEntryPoint;
    public NativeString intersectionEntryPoint;
};

public struct RayTracingPipelineStateDesc
{
    public NativeRef<IShaderProgram> program;
    public GfxCount hitGroupCount = 0;
    public HitGroupDesc *hitGroups;
    public int maxRecursion = 0;
    public Size maxRayPayloadSize = 0;
    public Size maxAttributeSizeInBytes = 8;
    public RayTracingPipelineFlags flags = RayTracingPipelineFlags::None;
};

// Specifies the bytes to overwrite into a record in the shader table.
public struct ShaderRecordOverwrite
{
    public Offset offset;   // Offset within the shader record.
    public Size size;       // Number of bytes to overwrite.
    public uint8_t data[8]; // Content to overwrite.
};

public struct ShaderTableDesc
{
    public GfxCount rayGenShaderCount;
    public NativeString* rayGenShaderEntryPointNames;
    public ShaderRecordOverwrite *rayGenShaderRecordOverwrites;

    public GfxCount missShaderCount;
    public NativeString *missShaderEntryPointNames;
    public ShaderRecordOverwrite *missShaderRecordOverwrites;

    public GfxCount hitGroupCount;
    public NativeString *hitGroupNames;
    public ShaderRecordOverwrite *hitGroupRecordOverwrites;

    NativeRef<IShaderProgram> program;
};

[COM("a721522c-df31-4c2f-a5e7-3b-e0-12-4b-31-78")]
public interface IShaderTable
{

};

[COM("0ca7e57d-8a90-44f3-bdb1-fe-9b-35-3f-5a-72")]
public interface IPipelineState
{
    Result getNativeHandle(InteropHandle *outHandle);
};

public struct ScissorRect
{
    public int32_t minX;
    public int32_t minY;
    public int32_t maxX;
    public int32_t maxY;
};

public struct Viewport
{
    public float originX = 0.0f;
    public float originY = 0.0f;
    public float extentX = 0.0f;
    public float extentY = 0.0f;
    public float minZ = 0.0f;
    public float maxZ = 1.0f;
};

public struct FramebufferDesc
{
    public GfxCount renderTargetCount;
    public NativeRef<IResourceView> *renderTargetViews;
    public NativeRef<IResourceView> depthStencilView;
    public NativeRef<IFramebufferLayout> layout;
};

[COM("0f0c0d9a-4ef3-4e18-9ba9-34-60-ea-69-87-95")]
public interface IFramebuffer
{
};

public enum class WindowHandleType
{
    Unknown,
    Win32Handle,
    XLibHandle,
};

public struct WindowHandle
{
    public WindowHandleType type;
    public void* handleValues[2];
    public static WindowHandle fromHwnd(void *hwnd)
    {
        WindowHandle handle = {WindowHandleType::Unknown, {nullptr, nullptr}};
        handle.type = WindowHandleType::Win32Handle;
        handle.handleValues[0] = hwnd;
        return handle;
    }
    public static WindowHandle fromXWindow(void *xdisplay, uint32_t xwindow)
    {
        WindowHandle handle = {WindowHandleType::Unknown, {nullptr, nullptr}};
        handle.type = WindowHandleType::XLibHandle;
        handle.handleValues[0] = xdisplay;
        handle.handleValues[1] = (void*)xwindow;
        return handle;
    }
};

public enum FaceMask
{
    Front = 1, Back = 2
};

public enum class TargetLoadOp
{
    Load, Clear, DontCare
};
public enum class TargetStoreOp
{
    Store, DontCare
};
public struct TargetAccessDesc
{
    public TargetLoadOp loadOp;
    public TargetLoadOp stencilLoadOp;
    public TargetStoreOp storeOp;
    public TargetStoreOp stencilStoreOp;
    public ResourceState initialState;
    public ResourceState finalState;
};
public struct RenderPassLayoutDesc
{
    public NativeRef<IFramebufferLayout> framebufferLayout;
    public GfxCount renderTargetCount;
    public TargetAccessDesc *renderTargetAccess;
    public TargetAccessDesc *depthStencilAccess;
};

[COM("daab0b1a-f45d-4ae9-bf2c-e0-bb-76-7d-fa-d1")]
public interface IRenderPassLayout
{
};

public enum class QueryType
{
    Timestamp,
    AccelerationStructureCompactedSize,
    AccelerationStructureSerializedSize,
    AccelerationStructureCurrentSize,
};

public struct QueryPoolDesc
{
    public QueryType type;
    public GfxCount count;
};

[COM("c2cc3784-12da-480a-a874-8b-31-96-1c-a4-36")]
public interface IQueryPool
{
    public Result getResult(GfxIndex queryIndex, GfxCount count, uint64_t *data);
    public Result reset();
};

[COM("77ea6383-be3d-40aa-8b45-fd-f0-d7-5b-fa-34")]
public interface ICommandEncoder
{
    public void endEncoding();
    public void writeTimestamp(IQueryPool queryPool, GfxIndex queryIndex);
};

public struct IndirectDispatchArguments
{
    public GfxCount ThreadGroupCountX;
    public GfxCount ThreadGroupCountY;
    public GfxCount ThreadGroupCountZ;
};

public struct IndirectDrawArguments
{
    public GfxCount VertexCountPerInstance;
    public GfxCount InstanceCount;
    public GfxIndex StartVertexLocation;
    public GfxIndex StartInstanceLocation;
};

public struct IndirectDrawIndexedArguments
{
    public GfxCount IndexCountPerInstance;
    public GfxCount InstanceCount;
    public GfxIndex StartIndexLocation;
    public GfxIndex BaseVertexLocation;
    public GfxIndex StartInstanceLocation;
};

public struct SamplePosition
{
    public int8_t x;
    public int8_t y;
};

public enum ClearResourceViewFlags
{
    None = 0,
    ClearDepth = 1,
    ClearStencil = 2,
    FloatClearValues = 4
};

[COM("F99A00E9-ED50-4088-8A0E-3B26755031EA")]
public interface IResourceCommandEncoder : ICommandEncoder
{
    public void copyBuffer(
                 IBufferResource dst,
                 Offset dstOffset,
                 IBufferResource src,
                 Offset srcOffset,
                 Size size);
    /// Copies texture from src to dst. If dstSubresource and srcSubresource has mipLevelCount = 0
    /// and layerCount = 0, the entire resource is being copied and dstOffset, srcOffset and extent
    /// arguments are ignored.
    public void copyTexture(
        ITextureResource dst,
        ResourceState dstState,
        SubresourceRange dstSubresource,
        int3 dstOffset,
        NativeRef<ITextureResource> src,
        ResourceState srcState,
        SubresourceRange srcSubresource,
        int3 srcOffset,
        int3 extent);

    /// Copies texture to a buffer. Each row is aligned to kTexturePitchAlignment.
    public void copyTextureToBuffer(
        IBufferResource dst,
        Offset dstOffset,
        Size dstSize,
        Size dstRowStride,
        ITextureResource src,
        ResourceState srcState,
        SubresourceRange srcSubresource,
        int3 srcOffset,
        int3 extent);
    public void uploadTextureData(
        ITextureResource dst,
        SubresourceRange subResourceRange,
        int3 offset,
        int3 extent,
        SubresourceData *subResourceData,
        GfxCount subResourceDataCount);
    public void uploadBufferData(IBufferResource dst, Offset offset, Size size, void *data);
    public void textureBarrier(
        GfxCount count, NativeRef<ITextureResource> *textures, ResourceState src, ResourceState dst);
    public void textureSubresourceBarrier(
        ITextureResource texture,
        SubresourceRange subresourceRange,
        ResourceState src,
        ResourceState dst);
    public void bufferBarrier(
        GfxCount count, NativeRef<IBufferResource> *buffers, ResourceState src, ResourceState dst);
    public void clearResourceView(
        IResourceView view, ClearValue *clearValue, ClearResourceViewFlags flags);
    public void resolveResource(
        ITextureResource source,
        ResourceState sourceState,
        SubresourceRange sourceRange,
        ITextureResource dest,
        ResourceState destState,
        SubresourceRange destRange);
    public void resolveQuery(
        IQueryPool queryPool,
        GfxIndex index,
        GfxCount count,
        IBufferResource buffer,
        Offset offset);
    public void beginDebugEvent(NativeString name, float rgbColor[3]);
    public void endDebugEvent();
};

[COM("7A8D56D0-53E6-4AD6-85F7-D14DC110FDCE")]
public interface IRenderCommandEncoder : IResourceCommandEncoder
{
    // Sets the current pipeline state. This method returns a transient shader object for
    // writing shader parameters. This shader object will not retain any resources or
    // sub-shader-objects bound to it. The user must be responsible for ensuring that any
    // resources or shader objects that is set into `outRootShaderObject` stays alive during
    // the execution of the command buffer.
    public Result bindPipeline(IPipelineState state, out IShaderObject outRootShaderObject);

    // Sets the current pipeline state along with a pre-created mutable root shader object.
    public Result bindPipelineWithRootObject(IPipelineState state, NativeRef<IShaderObject> rootObject);

    public void setViewports(GfxCount count, Viewport *viewports);
    public void setScissorRects(GfxCount count, ScissorRect *scissors);

    public void setPrimitiveTopology(PrimitiveTopology topology);
    public void setVertexBuffers(
        GfxIndex startSlot,
        GfxCount slotCount,
        NativeRef<IBufferResource>* buffers,
        Offset *offsets);

    public void setIndexBuffer(IBufferResource buffer, Format indexFormat, Offset offset);
    public void draw(GfxCount vertexCount, GfxIndex startVertex);
    public void drawIndexed(GfxCount indexCount, GfxIndex startIndex = 0, GfxIndex baseVertex = 0);
    public void drawIndirect(
        GfxCount maxDrawCount,
        IBufferResource argBuffer,
        Offset argOffset,
        NativeRef<IBufferResource> countBuffer,
        Offset countOffset = 0);
    public void drawIndexedIndirect(
        GfxCount maxDrawCount,
        IBufferResource argBuffer,
        Offset argOffset,
        NativeRef<IBufferResource> countBuffer,
        Offset countOffset = 0);
    public void setStencilReference(uint32_t referenceValue);
    public Result setSamplePositions(
        GfxCount samplesPerPixel, GfxCount pixelCount, SamplePosition *samplePositions);
    public void drawInstanced(
        GfxCount vertexCount,
        GfxCount instanceCount,
        GfxIndex startVertex,
        GfxIndex startInstanceLocation);
    public void drawIndexedInstanced(
        GfxCount indexCount,
        GfxCount instanceCount,
        GfxIndex startIndexLocation,
        GfxIndex baseVertexLocation,
        GfxIndex startInstanceLocation);
};

[COM("88AA9322-82F7-4FE6-A68A-29C7FE798737")]
public interface IComputeCommandEncoder : IResourceCommandEncoder
{
    // Sets the current pipeline state. This method returns a transient shader object for
    // writing shader parameters. This shader object will not retain any resources or
    // sub-shader-objects bound to it. The user must be responsible for ensuring that any
    // resources or shader objects that is set into `outRooShaderObject` stays alive during
    // the execution of the command buffer.
    public Result bindPipeline(IPipelineState state, out Optional<IShaderObject> outRootShaderObject);

    // Sets the current pipeline state along with a pre-created mutable root shader object.
    public Result bindPipelineWithRootObject(IPipelineState state, IShaderObject rootObject);

    public void dispatchCompute(int x, int y, int z);
    public void dispatchComputeIndirect(IBufferResource cmdBuffer, Offset offset);
};

public enum class AccelerationStructureCopyMode
{
    Clone, Compact
};

public struct AccelerationStructureQueryDesc
{
    public QueryType queryType;

    public NativeRef<IQueryPool> queryPool;

    public GfxIndex firstQueryIndex;
};

[COM("9a672b87-5035-45e3-967c-1f-85-cd-b3-63-4f")]
public interface IRayTracingCommandEncoder : IResourceCommandEncoder
{
    public void buildAccelerationStructure(
        AccelerationStructureBuildDesc *desc,
        GfxCount propertyQueryCount,
        AccelerationStructureQueryDesc *queryDescs);
    public void copyAccelerationStructure(
        NativeRef<IAccelerationStructure> dest,
        NativeRef<IAccelerationStructure> src,
        AccelerationStructureCopyMode mode);
    public void queryAccelerationStructureProperties(
        GfxCount accelerationStructureCount,
        NativeRef<IAccelerationStructure> *accelerationStructures,
        GfxCount queryCount,
        AccelerationStructureQueryDesc *queryDescs);
    public void serializeAccelerationStructure(DeviceAddress dest, IAccelerationStructure source);
    public void deserializeAccelerationStructure(IAccelerationStructure dest, DeviceAddress source);

    public Result bindPipeline(IPipelineState state, out IShaderObject rootObject);
    // Sets the current pipeline state along with a pre-created mutable root shader object.
    public Result bindPipelineWithRootObject(IPipelineState state, IShaderObject rootObject);

    /// Issues a dispatch command to start ray tracing workload with a ray tracing pipeline.
    /// `rayGenShaderIndex` specifies the index into the shader table that identifies the ray generation shader.
    public void dispatchRays(
        GfxIndex rayGenShaderIndex,
        NativeRef<IShaderTable> shaderTable,
        GfxCount width,
        GfxCount height,
        GfxCount depth);
};

[COM("5d56063f-91d4-4723-a7a7-7a-15-af-93-eb-48")]
public interface ICommandBuffer
{
    // Only one encoder may be open at a time. User must call `ICommandEncoder::endEncoding`
    // before calling other `encode*Commands` methods.
    // Once `endEncoding` is called, the `ICommandEncoder` object becomes obsolete and is
    // invalid for further use. To continue recording, the user must request a new encoder
    // object by calling one of the `encode*Commands` methods again.
    public void encodeRenderCommands(
        IRenderPassLayout renderPass,
        IFramebuffer framebuffer,
        out IRenderCommandEncoder outEncoder);

    public void encodeComputeCommands(out Optional<IComputeCommandEncoder> encoder);

    public void encodeResourceCommands(out Optional<IResourceCommandEncoder> outEncoder);

    public void encodeRayTracingCommands(out Optional<IRayTracingCommandEncoder> outEncoder);

    public void close();

    public Result getNativeHandle(out InteropHandle outHandle);
};

public enum class QueueType
{
    Graphics
};
public struct CommandQueueDesc
{
    public QueueType type;
};

[COM("14e2bed0-0ad0-4dc8-b341-06-3f-e7-2d-bf-0e")]
public interface ICommandQueue
{
    public const CommandQueueDesc* getDesc();

    public void executeCommandBuffers(
        GfxCount count,
        NativeRef<ICommandBuffer> *commandBuffers,
        Optional<IFence> fenceToSignal,
        uint64_t newFenceValue);

    public Result getNativeHandle(out InteropHandle outHandle);

    public void waitOnHost();

    /// Queues a device side wait for the given fences.
    public Result waitForFenceValuesOnDevice(GfxCount fenceCount, NativeRef<IFence> *fences, uint64_t *waitValues);
};

public enum TransientResourceHeapFlags
{
    None = 0,
    AllowResizing = 0x1,
};

public struct TransientResourceHeapDesc
{
    public TransientResourceHeapFlags flags;
    public Size constantBufferSize;
    public GfxCount samplerDescriptorCount;
    public GfxCount uavDescriptorCount;
    public GfxCount srvDescriptorCount;
    public GfxCount constantBufferDescriptorCount;
    public GfxCount accelerationStructureDescriptorCount;
};

[COM("cd48bd29-ee72-41b8-bcff-0a-2b-3a-aa-6d-0b")]
public interface ITransientResourceHeap
{
    // Waits until GPU commands issued before last call to `finish()` has been completed, and resets
    // all transient resources holds by the heap.
    // This method must be called before using the transient heap to issue new GPU commands.
    // In most situations this method should be called at the beginning of each frame.
    public Result synchronizeAndReset();

    // Must be called when the application has done using this heap to issue commands. In most situations
    // this method should be called at the end of each frame.
    public Result finish();

    // Command buffers are one-time use. Once it is submitted to the queue via
    // `executeCommandBuffers` a command buffer is no longer valid to be used any more. Command
    // buffers must be closed before submission. The current D3D12 implementation has a limitation
    // that only one command buffer maybe recorded at a time. User must finish recording a command
    // buffer before creating another command buffer.
    public Result createCommandBuffer(out Optional<ICommandBuffer> outCommandBuffer);
};

public struct SwapchainDesc
{
    public Format format;
    public GfxCount width, height;
    public GfxCount imageCount;
    public NativeRef<ICommandQueue> queue;
    public bool enableVSync;
};

[COM("be91ba6c-0784-4308-a1-00-19-c3-66-83-44-b2")]
public interface ISwapchain
{
    public const SwapchainDesc* getDesc();

    /// Returns the back buffer image at `index`.
    public Result getImage(GfxIndex index, out ITextureResource outResource);

    /// Present the next image in the swapchain.
    public Result present();

    /// Returns the index of next back buffer image that will be presented in the next
    /// `present` call. If the swapchain is invalid/out-of-date, this method returns -1.
    public int acquireNextImage();

    /// Resizes the back buffers of this swapchain. All render target views and framebuffers
    /// referencing the back buffer images must be freed before calling this method.
    public Result resize(GfxCount width, GfxCount height);

    // Check if the window is occluded.
    public bool isOccluded();

    // Toggle full screen mode.
    public Result setFullScreenMode(bool mode);
};

public struct DeviceInfo
{
    public DeviceType deviceType;

    public BindingStyle bindingStyle;

    public ProjectionStyle projectionStyle;

    /// An projection matrix that ensures x, y mapping to pixels
    /// is the same on all targets
    public float identityProjectionMatrix[16];

    /// The name of the graphics API being used by this device.
    public NativeString apiName;

    /// The name of the graphics adapter.
    public NativeString adapterName;

    /// The clock frequency used in timestamp queries.
    public uint64_t timestampFrequency;
};

public enum class DebugMessageType
{
    Info, Warning, Error
};
public enum class DebugMessageSource
{
    Layer, Driver, Slang
};

[COM("B219D7E8-255A-2572-D46C-A0E5D99CEB90")]
public interface IDebugCallback
{
    public void handleMessage(DebugMessageType type, DebugMessageSource source, NativeString message);
};

public struct SlangDesc
{
    public NativeRef<slang::IGlobalSession> slangGlobalSession = {slang::IGlobalSession()}; // (optional) A slang global session object. If null will create automatically.

    public slang::SlangMatrixLayoutMode defaultMatrixLayoutMode = slang::SlangMatrixLayoutMode::SLANG_MATRIX_LAYOUT_ROW_MAJOR;

    public NativeString *searchPaths = nullptr;
    public GfxCount searchPathCount = 0;

    public slang::PreprocessorMacroDesc *preprocessorMacros = nullptr;
    public GfxCount preprocessorMacroCount = 0;

    public NativeString targetProfile = ""; // (optional) Target shader profile. If null this will be set to platform dependent default.
    public slang::SlangFloatingPointMode floatingPointMode = slang::SlangFloatingPointMode::SLANG_FLOATING_POINT_MODE_DEFAULT;
    public slang::SlangOptimizationLevel optimizationLevel = slang::SlangOptimizationLevel::SLANG_OPTIMIZATION_LEVEL_DEFAULT;
    public slang::SlangTargetFlags targetFlags = slang::SlangTargetFlags.None;
    public slang::SlangLineDirectiveMode lineDirectiveMode = slang::SlangLineDirectiveMode::SLANG_LINE_DIRECTIVE_MODE_DEFAULT;
};

public struct ShaderCacheDesc
{
    // The root directory for the shader cache. If not set, shader cache is disabled.
    public NativeString shaderCachePath = "";
    // The maximum number of entries stored in the cache.
    public GfxCount maxEntryCount = 0;
};

public struct DeviceInteropHandles
{
    public InteropHandle handles[3] = {};
};

public struct DeviceDesc
{
    // The underlying API/Platform of the device.
    public DeviceType deviceType = DeviceType::Default;
    // The device's handles (if they exist) and their associated API. For D3D12, this contains a single InteropHandle
    // for the ID3D12Device. For Vulkan, the first InteropHandle is the VkInstance, the second is the VkPhysicalDevice,
    // and the third is the VkDevice. For CUDA, this only contains a single value for the CUDADevice.
    public DeviceInteropHandles existingDeviceHandles = {};
    // Name to identify the adapter to use
    public NativeString adapter = "";
    // Number of required features.
    public GfxCount requiredFeatureCount = 0;
    // Array of required feature names, whose size is `requiredFeatureCount`.
    public NativeString *requiredFeatures = nullptr;
    // A command dispatcher object that intercepts and handles actual low-level API call.
    void *apiCommandDispatcher = nullptr;
    // The slot (typically UAV) used to identify NVAPI intrinsics. If >=0 NVAPI is required.
    public GfxIndex nvapiExtnSlot = -1;
    // Configurations for the shader cache.
    public ShaderCacheDesc shaderCache = {};
    // Configurations for Slang compiler.
    public SlangDesc slang = {};

    public GfxCount extendedDescCount = 0;
    public void **extendedDescs = nullptr;
};

[COM("715bdf26-5135-11eb-AE93-02-42-AC-13-00-02")]
public interface IDevice
{
    public Result getNativeDeviceHandles(out DeviceInteropHandles outHandles);

    public bool hasFeature(NativeString feature);

    /// Returns a list of features supported by the renderer.
    public Result getFeatures(NativeString *outFeatures, Size bufferSize, GfxCount *outFeatureCount);

    public Result getFormatSupportedResourceStates(Format format, ResourceStateSet *outStates);

    public Result getSlangSession(NativeRef<slang::ISession>* outSlangSession);

    public Result createTransientResourceHeap(
        TransientResourceHeapDesc *desc,
        out Optional<ITransientResourceHeap> outHeap);

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
    public Result createTextureResource(
        TextureResourceDesc* desc,
        SubresourceData *initData,
        out ITextureResource outResource);

    public Result createTextureFromNativeHandle(
        InteropHandle handle,
        TextureResourceDesc* srcDesc,
        out ITextureResource outResource);

    public Result createTextureFromSharedHandle(
        InteropHandle handle,
        TextureResourceDesc *srcDesc,
        Size size,
        out ITextureResource outResource);

    /// Create a buffer resource
    public Result createBufferResource(
        BufferResourceDesc* desc,
        void *initData,
        out Optional<IBufferResource> outResource);

    public Result createBufferFromNativeHandle(
        InteropHandle handle,
        BufferResourceDesc* srcDesc,
        out IBufferResource outResource);

    public Result createBufferFromSharedHandle(
        InteropHandle handle,
        BufferResourceDesc* srcDesc,
        out IBufferResource outResource);

    public Result createSamplerState(SamplerStateDesc* desc, out ISamplerState outSampler);

    public Result createTextureView(
        ITextureResource texture, ResourceViewDesc* desc, out IResourceView outView);

    public Result createBufferView(
        IBufferResource buffer,
        Optional<IBufferResource> counterBuffer,
        ResourceViewDesc* desc,
        out Optional<IResourceView> outView);

    public Result createFramebufferLayout(FramebufferLayoutDesc* desc, out IFramebufferLayout outFrameBuffer);

    public Result createFramebuffer(FramebufferDesc* desc, out IFramebuffer outFrameBuffer);

    public Result createRenderPassLayout(
        RenderPassLayoutDesc* desc,
        out IRenderPassLayout outRenderPassLayout);

    public Result createSwapchain(
        SwapchainDesc* desc, WindowHandle window, out ISwapchain outSwapchain);

    public Result createInputLayout(
        InputLayoutDesc* desc, out IInputLayout outLayout);

    public Result createCommandQueue(CommandQueueDesc* desc, out Optional<ICommandQueue> outQueue);

    public Result createShaderObject(
        slang::TypeReflection *type,
        ShaderObjectContainerType container,
        out IShaderObject outObject);

    public Result createMutableShaderObject(
        slang::TypeReflection *type,
        ShaderObjectContainerType container,
        out IShaderObject outObject);

    public Result createShaderObjectFromTypeLayout(
        slang::TypeLayoutReflection *typeLayout, out IShaderObject outObject);

    public Result createMutableShaderObjectFromTypeLayout(
        slang::TypeLayoutReflection *typeLayout, out IShaderObject outObject);

    public Result createMutableRootShaderObject(
        IShaderProgram program,
        out IShaderObject outObject);

    public Result createShaderTable(ShaderTableDesc* desc, out IShaderTable outTable);

    public Result createProgram(
        void *desc,
        out IShaderProgram outProgram,
        out slang::ISlangBlob outDiagnosticBlob);

    public Result createProgram2(
        ShaderProgramDesc2 *desc,
        out Optional<IShaderProgram> outProgram,
        out Optional<slang::ISlangBlob> outDiagnosticBlob);

    public Result createGraphicsPipelineState(
        GraphicsPipelineStateDesc *desc,
        out Optional<IPipelineState> outState);

    public Result createComputePipelineState(
        ComputePipelineStateDesc* desc,
        out Optional<IPipelineState> outState);

    public Result createRayTracingPipelineState(
        RayTracingPipelineStateDesc *desc, out Optional<IPipelineState> outState);

    /// Read back texture resource and stores the result in `outBlob`.
    public Result readTextureResource(
        ITextureResource resource,
        ResourceState state,
        out slang::ISlangBlob outBlob,
        out Size outRowPitch,
        out Size outPixelSize);

    public Result readBufferResource(
        IBufferResource buffer,
        Offset offset,
        Size size,
        out Optional<slang::ISlangBlob> outBlob);

    /// Get the type of this renderer
    public DeviceInfo* getDeviceInfo();

    public Result createQueryPool(
        QueryPoolDesc* desc, out IQueryPool outPool);

    public Result getAccelerationStructurePrebuildInfo(
        AccelerationStructureBuildInputs* buildInputs,
        out AccelerationStructurePrebuildInfo outPrebuildInfo);

    public Result createAccelerationStructure(
        AccelerationStructureCreateDesc* desc,
        out IAccelerationStructure outView);

    public Result createFence(FenceDesc* desc, out IFence outFence);

    /// Wait on the host for the fences to signals.
    /// `timeout` is in nanoseconds, can be set to `kTimeoutInfinite`.
    public Result waitForFences(
        GfxCount fenceCount,
        NativeRef<IFence>* fences,
        uint64_t *values,
        bool waitForAll,
        uint64_t timeout);

    public Result getTextureAllocationInfo(
        TextureResourceDesc* desc, out Size outSize, out Size outAlignment);

    public Result getTextureRowAlignment(out Size outAlignment);
};

public struct ShaderCacheStats
{
    public GfxCount hitCount;
    public GfxCount missCount;
    public GfxCount entryCount;
};

[COM("715bdf26-5135-11eb-AE93-02-42-AC-13-00-02")]
public interface IShaderCache
{
    public Result clearShaderCache();
    public Result getShaderCacheStats(out ShaderCacheStats outStats);
    public Result resetShaderCacheStats();
};

#define SLANG_GFX_IMPORT [DllImport("gfx")]
/// Checks if format is compressed
SLANG_GFX_IMPORT public bool gfxIsCompressedFormat(Format format);

/// Checks if format is typeless
SLANG_GFX_IMPORT public bool gfxIsTypelessFormat(Format format);

/// Gets information about the format
SLANG_GFX_IMPORT public Result gfxGetFormatInfo(Format format, FormatInfo *outInfo);

/// Given a type returns a function that can conpublic struct it, or nullptr if there isn't one
SLANG_GFX_IMPORT public Result gfxCreateDevice(const DeviceDesc* desc, out Optional<IDevice> outDevice);

/// Reports current set of live objects in gfx.
/// Currently this only calls D3D's ReportLiveObjects.
SLANG_GFX_IMPORT public Result gfxReportLiveObjects();

/// Sets a callback for receiving debug messages.
/// The layer does not hold a strong reference to the callback object.
/// The user is responsible for holding the callback object alive.
SLANG_GFX_IMPORT public Result gfxSetDebugCallback(IDebugCallback callback);

/// Enables debug layer. The debug layer will check all `gfx` calls and verify that uses are valid.
SLANG_GFX_IMPORT public void gfxEnableDebugLayer();

SLANG_GFX_IMPORT public NativeString gfxGetDeviceTypeName(DeviceType type);

public bool succeeded(Result code)
{
    return code >= 0;
}

}
