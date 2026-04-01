// Slang HLSL compatibility library
//@hidden:

typedef uint UINT;

__intrinsic_op($(kIROp_RequireTargetExtension))
void __requireTargetExtension(constexpr String extensionName);

/// Built-in values or system value semantics represented as in/out global variables.
/// This allows the built-ins to be arbitrarily used from a global scope without being
/// explicitly passed as entry point parameters.
in uint __builtinWaveLaneIndex : SV_WaveLaneIndex;
in uint __builtinWaveLaneCount : SV_WaveLaneCount;
in uint __builtinQuadLaneIndex : SV_QuadLaneIndex;

//@public:
/// Represents an interface for buffer data layout.
/// This interface is used as a base for defining specific data layouts for buffers.
[sealed]
[builtin]
__magic_type(IBufferDataLayoutType)
interface IBufferDataLayout
{
}

/// @category misc_types
__intrinsic_type($(kIROp_DefaultBufferLayoutType))
__magic_type(DefaultDataLayoutType)
struct DefaultDataLayout : IBufferDataLayout
{};

/// @category misc_types
__intrinsic_type($(kIROp_Std140BufferLayoutType))
[require(spirv)]
[require(glsl)]
__magic_type(Std140DataLayoutType)
struct Std140DataLayout : IBufferDataLayout
{};

/// @category misc_types
__intrinsic_type($(kIROp_Std430BufferLayoutType))
[require(spirv)]
[require(glsl)]
__magic_type(Std430DataLayoutType)
struct Std430DataLayout : IBufferDataLayout
{};

/// @category misc_types
__intrinsic_type($(kIROp_ScalarBufferLayoutType))
__magic_type(ScalarDataLayoutType)
struct ScalarDataLayout : IBufferDataLayout
{};

//@hidden:
__generic<T, L : IBufferDataLayout = DefaultDataLayout>
__intrinsic_type($(kIROp_GLSLShaderStorageBufferType))
__magic_type(GLSLShaderStorageBufferType)
struct GLSLShaderStorageBuffer {}

__generic<T,L:IBufferDataLayout>
__intrinsic_op($(kIROp_StructuredBufferGetDimensions))
[require(cpp_cuda_glsl_hlsl_metal_spirv, appendstructuredbuffer)]
uint2 __structuredBufferGetDimensions(AppendStructuredBuffer<T,L> buffer);

__generic<T,L:IBufferDataLayout>
__intrinsic_op($(kIROp_StructuredBufferGetDimensions))
[require(cpp_cuda_glsl_hlsl_metal_spirv, consumestructuredbuffer)]
uint2 __structuredBufferGetDimensions(ConsumeStructuredBuffer<T,L> buffer);

__intrinsic_op($(kIROp_StructuredBufferGetDimensions))
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, structuredbuffer)]
uint2 __structuredBufferGetDimensions<T,L:IBufferDataLayout>(StructuredBuffer<T,L> buffer);

__intrinsic_op($(kIROp_StructuredBufferGetDimensions))
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, structuredbuffer_rw)]
uint2 __structuredBufferGetDimensions<T,L:IBufferDataLayout>(RWStructuredBuffer<T,L> buffer);

__intrinsic_op($(kIROp_StructuredBufferGetDimensions))
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, structuredbuffer_rw)]
uint2 __structuredBufferGetDimensions<T,L:IBufferDataLayout>(RasterizerOrderedStructuredBuffer<T,L> buffer);

//@public:
/**
Represents an opaque handle to an append structured buffer allocated in global memory.
A structured buffer can be viewed as an array of the specified element type.
An append structure buffer internally maintains an atomic counter to keep track of the number of elements in the buffer,
and provide an atomic operation to append a new element to the buffer.
 @param T The element type of the buffer.
 @param L The memory layout of the buffer.
 @remarks
This type is supported natively when targeting HLSL.
When generating code for other targets, this type is translated into a pair or an ordinary `RWStructuredBuffer` and
a separate `RWStructuredBuffer` that holds the atomic counter.
The `L` generic parameter is used to specify the memory layout of the buffer when
generating SPIRV.
`L` must be one of `DefaultDataLayout`, `Std140DataLayout`, `Std430DataLayout` or `ScalarDataLayout`.
The default value is `DefaultDataLayout`.
When generating code for other targets, this parameter is ignored and has no effect on the generated code.
 @see `RWStructuredBuffer`, `ConsumeStructuredBuffer`, `RasterizerOrderedStructuredBuffer`.
 @category buffer_types
*/
__generic<T, L:IBufferDataLayout=DefaultDataLayout>
__magic_type(HLSLAppendStructuredBufferType)
__intrinsic_type($(kIROp_HLSLAppendStructuredBufferType))
[require(cpp_cuda_glsl_hlsl_spirv, appendstructuredbuffer)]
struct AppendStructuredBuffer
{
    __intrinsic_op($(kIROp_StructuredBufferAppend))
    /// Appends a new element to the buffer.
    ///@param value The element to be appended to the buffer.
    void Append(T value);

    /// Get information about the number of elements and stride of the buffer.
    ///@param numStructs The number of elements in the buffer.
    ///@param stride The stride of the buffer.
    [ForceInline]
    void GetDimensions(
        out uint numStructs,
        out uint stride)
    {
        let result = __structuredBufferGetDimensions(this);
        numStructs = result.x;
        stride = result.y;
    }
};

//@public:
/**
Represents an opaque handle to a read-only buffer allocated in global memory that is indexed in bytes.
ByteAddressBuffer can be used when working with raw buffers. Raw buffer can be viewed as a bag of bits to
which you want raw access, that is, a buffer that you can conveniently access through chunks of one to
four 32-bit typeless address values.
 @remarks
This type is supported natively when targeting HLSL.
For all other targets, this type maps to a buffer of 32bit unsigned integers.
 @category buffer_types
*/
__magic_type(HLSLByteAddressBufferType)
__intrinsic_type($(kIROp_HLSLByteAddressBufferType))
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, byteaddressbuffer)]
struct ByteAddressBuffer
{
    /// Get the number of bytes in the buffer.
    ///@param[out] dim The number of bytes in the buffer.
    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, structuredbuffer)]
    void GetDimensions(out uint dim)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm ".GetDimensions";
        case cuda: __intrinsic_asm ".GetDimensions";
        case hlsl: __intrinsic_asm ".GetDimensions";
        default:
            dim = __structuredBufferGetDimensions(__getEquivalentStructuredBuffer<uint>(this)).x*4;
        }
    }

    /// Load a 32-bit unsigned integer or value with type of `T` from the buffer at the specified location.
    ///@param T The type of the value to load from the buffer.
    ///@param location The input address in bytes, which must be a multiple of 4.
    ///@param alignment Specifies the alignment of the location, which must be a multiple of 4.
    ///@param[out] status The status of the operation.
    ///@return The value loaded from the buffer.
    ///
    ///@remarks
    /// You can't access the output parameter `status` directly; instead,
    /// pass the status to the `CheckAccessFullyMapped` intrinsic function.
    /// `CheckAccessFullyMapped` returns TRUE if all values from the corresponding Sample,
    /// Gather, or Load operation accessed mapped tiles in a tiled resource.
    /// If any values were taken from an unmapped tile, `CheckAccessFullyMapped` returns FALSE.
    /// When targeting non-HLSL, the status is always 0.
    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer)]
    uint Load(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load";
        default:
            return __byteAddressBufferLoad<uint>(this, location, 0);
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl, byteaddressbuffer)]
    uint Load(int location, out uint status)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load";
        }
    }

    /// Load two 32-bit unsigned integers from the buffer at the specified location
    /// with additional alignment.
    ///@param location The input address in bytes.
    ///@param alignment Specifies the alignment of the location, which must be a multiple of 4.
    ///@param[out] status The status of the operation.
    ///@return Two 32-bit unsigned integers loaded from the buffer.
    ///
    ///@remarks
    /// This function only supports when targeting HLSL.
    /// You can't access the output parameter `status` directly; instead,
    /// pass the status to the `CheckAccessFullyMapped` intrinsic function.
    /// `CheckAccessFullyMapped` returns TRUE if all values from the corresponding Sample,
    /// Gather, or Load operation accessed mapped tiles in a tiled resource.
    /// If any values were taken from an unmapped tile, `CheckAccessFullyMapped` returns FALSE.
    /// When targeting non-HLSL, the status is always 0.
    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer)]
    uint2 Load2(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load2";
        default:
            return __byteAddressBufferLoad<uint2>(this, location, 0);
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer)]
    uint2 Load2(int location, int alignment)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load2";
        default:
            return __byteAddressBufferLoad<uint2>(this, location, alignment);
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl, byteaddressbuffer)]
    uint2 Load2(int location, out uint status)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load2";
        }
    }

    /// Load two 32-bit unsigned integers from the buffer at the specified location with alignment
    /// of stride of `uint2`, which is 8.
    ///@param location The input address in bytes, which must be a multiple of alignment of 8. Invalid
    /// value of location will cause undefined behavior.
    ///@return `uint2` Two 32-bit unsigned integers loaded from the buffer.
    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer)]
    uint2 Load2Aligned(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load2";
        default:
            return __byteAddressBufferLoad<uint2>(this, location, __naturalStrideOf<uint2>());
        }
    }

    /// Load three 32-bit unsigned integers from the buffer at the specified location.
    ///@param location The input address in bytes, which must be a multiple of 4.
    ///@param alignment Specifies the alignment of the location, which must be a multiple of 4.
    ///@param[out] status The status of the operation.
    ///@return `uint3` Three 32-bit unsigned integer value loaded from the buffer.
    ///
    ///@remarks
    /// This function only supports when targeting HLSL.
    /// You can't access the output parameter `status` directly; instead,
    /// pass the status to the `CheckAccessFullyMapped` intrinsic function.
    /// `CheckAccessFullyMapped` returns TRUE if all values from the corresponding Sample,
    /// Gather, or Load operation accessed mapped tiles in a tiled resource.
    /// If any values were taken from an unmapped tile, `CheckAccessFullyMapped` returns FALSE.
    /// When targeting non-HLSL, the status is always 0.
    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer)]
    uint3 Load3(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load3";
        default:
            return __byteAddressBufferLoad<uint3>(this, location, 0);
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer)]
    uint3 Load3(int location, int alignment)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load3";
        default:
            return __byteAddressBufferLoad<uint3>(this, location, alignment);
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl, byteaddressbuffer)]
    uint3 Load3(int location, out uint status)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load3";
        }
    }

    /// Load three 32-bit unsigned integers from the buffer at the specified location with alignment
    /// of stride of `uint3`, which is 12.
    ///@param location The input address in bytes which must be a multiple of alignment of 12.
    ///@return `uint3` Three 32-bit unsigned integer value loaded from the buffer.
    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer)]
    uint3 Load3Aligned(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load3";
        default:
            return __byteAddressBufferLoad<uint3>(this, location, __naturalStrideOf<uint3>());
        }
    }

    /// Load four 32-bit unsigned integers from the buffer at the specified location.
    ///@param location The input address in bytes which must be a multiple of alignment of 4.
    ///@param alignment Specifies the alignment of the location, which must be a multiple of 4.
    ///@param[out] status The status of the operation.
    ///@return `uint4` Four 32-bit unsigned integer value loaded from the buffer.
    ///
    ///@remarks
    /// This function only supports when targeting HLSL.
    /// You can't access the output parameter `status` directly; instead,
    /// pass the status to the `CheckAccessFullyMapped` intrinsic function.
    /// `CheckAccessFullyMapped` returns TRUE if all values from the corresponding Sample,
    /// Gather, or Load operation accessed mapped tiles in a tiled resource.
    /// If any values were taken from an unmapped tile, `CheckAccessFullyMapped` returns FALSE.
    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer)]
    uint4 Load4(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load4";
        default:
            return __byteAddressBufferLoad<uint4>(this, location, 0);
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer)]
    uint4 Load4(int location, int alignment)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load4";
        default:
            return __byteAddressBufferLoad<uint4>(this, location, alignment);
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl, byteaddressbuffer)]
    uint4 Load4(int location, out uint status)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load4";
        }
    }

    /// Load four 32-bit unsigned integers from the buffer at the specified location with alignment
    /// of `uint4`, which is 16.
    ///@param location The input address in bytes which must be a multiple of alignment of 16.
    ///@return `uint4` Four 32-bit unsigned integer value loaded from the buffer.
    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer)]
    uint4 Load4Aligned(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load4";
        default:
            return __byteAddressBufferLoad<uint4>(this, location, __naturalStrideOf<uint4>());
        }
    }

    [__readNone]
    [ForceInline]
    T Load<T>(int location)
    {
        return __byteAddressBufferLoad<T>(this, location, 0);
    }

    [__readNone]
    [ForceInline]
    T Load<T>(int location, int alignment)
    {
        return __byteAddressBufferLoad<T>(this, location, alignment);
    }

    /// Load an element with type `T` from the buffer at the specified location with alignment of `T`.
    ///@param location The input address in bytes which must be a multiply of size of `T`.
    ///@return T value with type `T` loaded from the buffer.
    ///@remarks
    ///Currently, this function only supports when `T` is scalar, vector or matrix type.
    [__readNone]
    [ForceInline]
    T LoadAligned<T>(int location)
    {
        return __byteAddressBufferLoad<T>(this, location, __naturalStrideOf<T>());
    }
};

// Texture

/// Represent a texture shape type that can be used to specify the shape of a texture.
/// Used for the `Shape` parameter of the `_Texture` type.
///
/// Implemented by `__Shape1D`, `__Shape2D`, `__Shape3D`, `__ShapeCube` and `__ShapeBuffer`.
/// @see `_Texture`.
/// @internal
[sealed]
[builtin]
interface __ITextureShape
{
    static const int flavor;
    static const int dimensions;
    static const int planeDimensions;
}

/// Represent a 1D, 2D or 3D texture shape that can be used as the `Shape` parameter of the `_Texture` type.
///
/// Implemented by `__Shape1D`, `__Shape2D` and `__Shape3D`.
/// @see `_Texture`.
/// @internal
[sealed]
[builtin]
interface __ITextureShape1D2D3D : __ITextureShape
{
}

/// When used as the `Shape` parameter of the `_Texture` type, specifies a 1D texture.
/// @category misc_types Miscelaneous types
__magic_type(TextureShape1DType)
__intrinsic_type($(kIROp_TextureShape1DType))
struct __Shape1D : __ITextureShape1D2D3D
{
    static const int flavor = $(SLANG_TEXTURE_1D);
    static const int dimensions = 1;
    static const int planeDimensions = 1;
}

/// When used as the `Shape` parameter of the `_Texture` type, specifies a 2D texture.
/// @category misc_types
__magic_type(TextureShape2DType)
__intrinsic_type($(kIROp_TextureShape2DType))
struct __Shape2D : __ITextureShape1D2D3D
{
    static const int flavor = $(SLANG_TEXTURE_2D);
    static const int dimensions = 2;
    static const int planeDimensions = 2;
}

/// When used as the `Shape` parameter of the `_Texture` type, specifies a 3D texture.
/// @category misc_types
__magic_type(TextureShape3DType)
__intrinsic_type($(kIROp_TextureShape3DType))
struct __Shape3D : __ITextureShape1D2D3D
{
    static const int flavor = $(SLANG_TEXTURE_3D);
    static const int dimensions = 3;
    static const int planeDimensions = 3;
}

/// When used as the `Shape` parameter of the `_Texture` type, specifies a Cube texture.
/// @category misc_types
__magic_type(TextureShapeCubeType)
__intrinsic_type($(kIROp_TextureShapeCubeType))
struct __ShapeCube : __ITextureShape
{
    static const int flavor = $(SLANG_TEXTURE_CUBE);
    static const int dimensions = 3;
    static const int planeDimensions = 2;
}

/// When used as the `Shape` parameter of the `_Texture` type, specifies a buffer texture.
/// @category misc_types
__magic_type(TextureShapeBufferType)
__intrinsic_type($(kIROp_TextureShapeBufferType))
struct __ShapeBuffer : __ITextureShape
{
    static const int flavor = $(SLANG_TEXTURE_BUFFER);
    static const int dimensions = 1;
    static const int planeDimensions = 1;
}

//@hidden:
__intrinsic_op(vectorReshape)
vector<T,N> __vectorReshape<let N : int, T, let M : int>(vector<T,M> vin);

__intrinsic_op(makeVector)
__generic<T, let N:int>
vector<T,N+1> __makeVector(vector<T,N> vec, T scalar);

//@public:
/// Represent types that can be used as texel element.
[sealed]
[builtin]
interface ITexelElement
{
    associatedtype Element : __BuiltinArithmeticType;
    static const int elementCount;
    __init(Element x);
}

extension<T:__BuiltinArithmeticType> T : ITexelElement
{
    typealias Element = T;
    static const int elementCount = 1;
    __intrinsic_op(0) __init(Element x);
}

${{{
// Scalar types that can be used as texel element.
const char* texeElementScalarTypes[] = {
    "half",
    "float",
    "int",
    "uint",
    "int8_t",
    "int16_t",
    "uint8_t",
    "uint16_t"
};
for (auto elementType : texeElementScalarTypes)
{
}}}
extension<int N> vector<$(elementType), N> : ITexelElement
{
    typealias Element = $(elementType);
    static const int elementCount = N;
    __intrinsic_op($(kIROp_MakeVectorFromScalar)) __init(Element x);
}
${{{
} // end for texelElementScalarTypes.
}}}

// Additional 64-bit types that can be used as texel element.
extension double:ITexelElement
{
    typealias Element = double;
    static const int elementCount = 1;
    __intrinsic_op(0) __init(Element x);
}
extension double2:ITexelElement
{
    typealias Element = double;
    static const int elementCount = 2;
    __intrinsic_op($(kIROp_MakeVectorFromScalar)) __init(Element x);
}
extension uint64_t:ITexelElement
{
    typealias Element = uint64_t;
    static const int elementCount = 1;
    __intrinsic_op(0) __init(Element x);
}
extension int64_t:ITexelElement
{
    typealias Element = int64_t;
    static const int elementCount = 1;
    __intrinsic_op(0) __init(Element x);
}
extension vector<uint64_t,2>:ITexelElement
{
    typealias Element = uint64_t;
    static const int elementCount = 2;
    __intrinsic_op($(kIROp_MakeVectorFromScalar)) __init(Element x);
}
extension vector<int64_t,2>:ITexelElement
{
    typealias Element = int64_t;
    static const int elementCount = 2;
    __intrinsic_op($(kIROp_MakeVectorFromScalar)) __init(Element x);
}

//@public:
/// A parameterized type that represents all flavors of texture types supported by the Slang language.
/// Please note that this type is not intended to be used directly in user code, and not all combinations
/// of the generic arguments are valid.
/// Instead, use the specific texture types such as `Texture1D`, `Texture2DArray` and `Sampler2D` etc.
/// This documentation is provided for reference purposes only.
/// @param T The element type of the texture. Must be a scalar or vector type.
/// @param Shape The shape of the texture. Must be one of `__Shape1D`, `__Shape2D`, `__Shape3D`, `__ShapeCube` or `__ShapeBuffer`.
/// @param isArray Indicates whether the texture is an array texture.
/// @param isMS Indicates whether the texture is a multisampled texture.
/// @param sampleCount The number of samples of a multisampled texture.
/// @param access The access mode of the texture. 0 for read-only, 1 for read-write, 2 for rasterizer-ordered, 3 for feedback.
/// @param isShadow Indicates whether the texture is a shadow texture (for combined texture-sampler only).
/// @param isCombined Indicates whether the texture is a combined texture-sampler.
/// @param format The storage format of the texture. Users should specify the format using an `[format("...")]` attribute instead.
/// @see `Texture1D`, `Texture2D`, `Texture3D`, `TextureCube`, `Texture1DArray`,
/// `Texture2DArray`, `TextureCubeArray`, `Sampler1D`, `Sampler2D`, `Sampler3D`, `SamplerCube`, `Sampler1DArray`, `Sampler2DArray`, `SamplerCubeArray`,
/// `Texture2DMS`, `Texture2DMSArray`, `RWTexture1D`, `RWTexture2D`, `RWTexture3D`, `RWTexture1DArray`, `RWTexture2DArray`,
/// `RWTexture2DMS`, `RWTexture2DMSArray`, `Buffer`, `RWBuffer`, `FeedbackTexture2D`, `FeedbackTexture2DArray`.
/// @remarks
/// HLSL texture types are implemented as typealiases to the builtin `_Texture` type. Users
/// are advised to use the HLSL-specific texture types instead of `_Texture` directly.
///
/// For read-write textures, Slang will automatically infer `format` from `T`.
/// To explicitly specify texel storage formats for read-write textures,
/// use the `[format("formatString")]` attribute on the texture parameter declaration.
/// Allowed `formatString` values are:
///
/// |id | Format string        | Meaning           |
/// |:--|:---------------------|:------------------|
/// |1  |`"rgba32f"`           | 4 channel 32-bit floating point texture |
/// |2  |`"rgba16f"`           | 4 channel 16-bit floating point texture |
/// |3  |`"rg32f"`             | 2 channel 32-bit floating point texture |
/// |4  |`"rg16f"`             | 2 channel 16-bit floating point texture |
/// |5  |`"r11f_g11f_b10f"`    | 3 channel 11/11/10-bit floating point texture |
/// |6  |`"r32f"`              | 1 channel 32-bit floating point texture |
/// |7  |`"r16f"`              | 1 channel 16-bit floating point texture |
/// |8  |`"rgba16"`            | 4 channel 16-bit normalized unsigned integer texture |
/// |9  |`"rgb10_a2"`          | 4 channel 10/10/10/2-bit signed integer texture |
/// |10 |`"rgba8"`             | 4 channel 8-bit normalized unsigned integer texture |
/// |11 |`"rg16"`              | 2 channel 16-bit normalized unsigned integer texture |
/// |12 |`"rg8"`               | 2 channel 8-bit normalized unsigned integer texture |
/// |13 |`"r16"`               | 1 channel 16-bit normalized unsigned integer texture |
/// |14 |`"r8"`                | 1 channel 8-bit normalized unsigned integer texture |
/// |15 |`"rgba16_snorm"`      | 4 channel 16-bit normalized signed integer texture |
/// |16 |`"rgba8_snorm"`       | 4 channel 8-bit normalized signed integer texture |
/// |17 |`"rg16_snorm"`        | 2 channel 16-bit normalized signed integer texture |
/// |18 |`"rg8_snorm"`         | 2 channel 8-bit normalized signed integer texture |
/// |19 |`"r16_snorm"`         | 1 channel 16-bit normalized signed integer texture |
/// |20 |`"r8_snorm"`          | 1 channel 8-bit normalized signed integer texture |
/// |21 |`"rgba32i"`           | 4 channel 32-bit signed integer texture |
/// |22 |`"rgba16i"`           | 4 channel 16-bit signed integer texture |
/// |23 |`"rgba8i"`            | 4 channel 8-bit signed integer texture |
/// |24 |`"rg32i"`             | 2 channel 32-bit signed integer texture |
/// |25 |`"rg16i"`             | 2 channel 16-bit signed integer texture |
/// |26 |`"rg8i"`              | 2 channel 8-bit signed integer texture |
/// |27 |`"r32i"`              | 1 channel 32-bit signed integer texture |
/// |28 |`"r16i"`              | 1 channel 16-bit signed integer texture |
/// |29 |`"r8i"`               | 1 channel 8-bit signed integer texture |
/// |30 |`"rgba32ui"`          | 4 channel 32-bit unsigned integer texture |
/// |31 |`"rgba16ui"`          | 4 channel 16-bit unsigned integer texture |
/// |32 |`"rgb10_a2ui"`        | 4 channel 10/10/10/2-bit unsigned integer texture |
/// |33 |`"rgba8ui"`           | 4 channel 8-bit unsigned integer texture |
/// |34 |`"rg32ui"`            | 2 channel 32-bit unsigned integer texture |
/// |35 |`"rg16ui"`            | 2 channel 16-bit unsigned integer texture |
/// |36 |`"rg8ui"`             | 2 channel 8-bit unsigned integer texture |
/// |37 |`"r32ui"`             | 1 channel 32-bit unsigned integer texture |
/// |38 |`"r16ui"`             | 1 channel 16-bit unsigned integer texture |
/// |39 |`"r8ui"`              | 1 channel 8-bit unsigned integer texture |
/// |40 |`"r64ui"`             | 1 channel 64-bit unsigned integer texture |
/// |41 |`"r64i"`              | 1 channel 64-bit signed integer texture |
///
/// When targeting Vulkan, a combined-texture-sampler type (`isCombined==1`) translates to a `OpTypeSampledImage` type in SPIR-V.
/// For other targets, the combined-texture-sampler type is translated to a pair of a `Texture` and `SamplerState`.
/// `isShadow` is only applicable to combined-texture-sampler types and must be `0` for non-combined texture types.
/// @internal
/// @category texture_types Texture types
__magic_type(TextureType)
__intrinsic_type($(kIROp_TextureType))
struct _Texture<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let isMS:int, let sampleCount:int, let access:int, let isShadow:int, let isCombined:int, let format:int>
{
}

//@hidden:
// Combined texture sampler specific functions

[require(glsl, texture_sm_4_1)]
float __glsl_texture<TSampler, TCoord>(TSampler s, TCoord value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "texture($0, $1)";
    }
}

__glsl_extension(GL_EXT_texture_shadow_lod)
[require(glsl, texture_shadowlod)]
float __glsl_texture_1d_shadow<TSampler, TCoord>(TSampler s, TCoord value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "texture($0, $1)";
    }
}

__glsl_extension(GL_EXT_texture_shadow_lod)
[require(glsl, texture_shadowlod)]
float __glsl_texture_3d_array_shadow<TSampler, TCoord>(TSampler s, TCoord value, float compare)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "texture($0, $1, $2)";
    }
}

__glsl_extension(GL_EXT_texture_shadow_lod)
[require(glsl, texture_sm_4_1)]
float __glsl_texture_offset<TSampler, TCoord, TOffset>( TSampler s, TCoord value, constexpr TOffset offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureOffset($0, $1, $2)";
    }
}

__glsl_extension(GL_EXT_texture_shadow_lod)
[require(glsl, texture_shadowlod)]
float __glsl_texture_offset_1d_shadow<TSampler, TCoord, TOffset>(TSampler s, TCoord value, constexpr TOffset offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureOffset($0, $1, $2)";
    }
}

__glsl_extension(GL_EXT_texture_shadow_lod)
[require(glsl, texture_sm_4_1)]
float __glsl_texture_level<TSampler, TCoord>(TSampler s, TCoord value, float level)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLod($0, $1, $2)";
    }
}

__glsl_extension(GL_EXT_texture_shadow_lod)
[require(glsl, texture_shadowlod)]
float __glsl_texture_level_1d_shadow<TSampler, TCoord>(TSampler s, TCoord value, float level)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLod($0, $1, $2)";
    }
}

__glsl_extension(GL_EXT_texture_shadow_lod)
[require(glsl, texture_shadowlod)]
float __glsl_texture_level_offset<TSampler, TCoord, TOffset>(TSampler s, TCoord value, float level, constexpr TOffset offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLodOffset($0, $1, $2, $3)";
    }
}

__glsl_extension(GL_EXT_texture_shadow_lod)
[require(glsl, texture_shadowlod)]
float __glsl_texture_level_offset_1d_shadow<TSampler, TCoord, TOffset>(TSampler s, TCoord value, float level, constexpr TOffset offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLodOffset($0, $1, $2, $3)";
    }
}

[require(glsl, texture_sm_4_1)]
float __glsl_texture<TTexture, TCoord>(TTexture t, SamplerComparisonState s, TCoord value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "texture($p, $2)";
    }
}

__glsl_extension(GL_EXT_texture_shadow_lod)
[require(glsl, texture_shadowlod)]
float __glsl_texture_1d_shadow<TTexture, TCoord>(TTexture t, SamplerComparisonState s, TCoord value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "texture($p, $2)";
    }
}

__glsl_extension(GL_EXT_texture_shadow_lod)
[require(glsl, texture_shadowlod)]
float __glsl_texture_3d_array_shadow<TTexture, TCoord>(TTexture t, SamplerComparisonState s, TCoord value, float compare)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "texture($p, $2, $3)";
    }
}

[require(glsl, texture_sm_4_1)]
float __glsl_texture_offset<TTexture, TCoord, TOffset>(TTexture t,SamplerComparisonState s, TCoord value, constexpr TOffset offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureOffset($p, $2, $3)";
    }
}

__glsl_extension(GL_EXT_texture_shadow_lod)
[require(glsl, texture_shadowlod)]
float __glsl_texture_offset_1d_shadow<TTexture, TCoord, TOffset>(TTexture t,SamplerComparisonState s, TCoord value, constexpr TOffset offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureOffset($p, $2, $3)";
    }
}

[require(glsl, texture_sm_4_1)]
float __glsl_texture_level<TTexture, TCoord>(TTexture t,SamplerComparisonState s, TCoord value, float level)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLod($p, $2, $3)";
    }
}

__glsl_extension(GL_EXT_texture_shadow_lod)
[require(glsl, texture_shadowlod)]
float __glsl_texture_level_1d_shadow<TTexture, TCoord>(TTexture t,SamplerComparisonState s, TCoord value, float level)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLod($p, $2, $3)";
    }
}

__glsl_extension(GL_EXT_texture_shadow_lod)
[require(glsl, texture_shadowlod)]
float __glsl_texture_level_offset<TTexture, TCoord, TOffset>(TTexture t,SamplerComparisonState s, TCoord value, float level, constexpr TOffset offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLodOffset($p, $2, $3, $4)";
    }
}

__glsl_extension(GL_EXT_texture_shadow_lod)
[require(glsl, texture_shadowlod)]
float __glsl_texture_level_offset_1d_shadow<TTexture, TCoord, TOffset>(TTexture t,SamplerComparisonState s, TCoord value, float level, constexpr TOffset offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLodOffset($p, $2, $3, $4)";
    }
}

__intrinsic_op($(kIROp_MetalCastToDepthTexture))
__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let isMS:int, let sampleCount:int, let access:int, let isShadow:int, let isCombined:int, let format:int>
_Texture<T,Shape,isArray,isMS,sampleCount,access,1,isCombined,format> __metal_asDepthTexture(_Texture<T,Shape,isArray,isMS,sampleCount,access,isShadow,isCombined,format> tex);

//@public:

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let isMS:int, let sampleCount:int, let isShadow:int, let format:int>
extension _Texture<T,Shape,isArray,isMS,sampleCount,0,isShadow,1,format>
{
    //@hidden:
    static const int access = 0;

    //@public:
    typealias TextureCoord = vector<float, Shape.dimensions>;

    __intrinsic_op($(kIROp_CombinedTextureSamplerGetTexture))
    _Texture<T, Shape, isArray, isMS, sampleCount, 0, isShadow, 0, format> __getTexture();

    __intrinsic_op($(kIROp_CombinedTextureSamplerGetSampler))
    SamplerState __getSampler();

    __intrinsic_op($(kIROp_CombinedTextureSamplerGetSampler))
    SamplerComparisonState __getComparisonSampler();

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv, texture_querylod)]
    float CalculateLevelOfDetail(TextureCoord location)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            return __getTexture().CalculateLevelOfDetail(__getSampler(), location);
        case metal:
            return __getTexture().CalculateLevelOfDetail(__getSampler(), location);
        case glsl:
            __intrinsic_asm "textureQueryLod($0, $1).x";
        case spirv:
            return (spirv_asm
            {
                OpCapability ImageQuery;
                result:$$float2 = OpImageQueryLod $this $location
            }).x;
        }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv, texture_querylod)]
    float CalculateLevelOfDetailUnclamped(TextureCoord location)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            return __getTexture().CalculateLevelOfDetailUnclamped(__getSampler(), location);
        case metal:
            return __getTexture().CalculateLevelOfDetailUnclamped(__getSampler(), location);
        case glsl:
            __intrinsic_asm "textureQueryLod($0, $1).y";
        case spirv:
            return (spirv_asm
            {
                OpCapability ImageQuery;
                result:$$float2 = OpImageQueryLod $this $location
            }).y;
        }
    }

    /// Samples the texture at the given location.
    ///
    ///@param s The `SamplerState` to use for the sampling operation. This parameter is omitted when `this` is a combined texture sampler type (`isCombined == 0`).
    ///@param location The location to sample the texture at.
    ///@param offset Texel offset to apply.
    ///@param clamp The max level of detail to use.
    ///@param[out] status The result status of the operation.
    ///                   This parameter is currently only used when targeting HLSL.
    ///                   For other targets, the result status is always 0.
    ///@return The sampled texture value.
    ///@see `SampleBias`, `SampleLevel`, `SampleGrad`, `SampleCmp`, `SampleCmpLevelZero`, `SampleCmpLevel`.
    ///@remarks
    /// The `Sample` function is defined for all read-only texture types, including
    /// `Texture1D`, `Texture2D`, `Texture3D`, `TextureCube`,
    /// `Texture1DArray`, `Texture2DArray` and `TextureCubeArray`.
    ///
    /// The function is not available for read-write texture types.
    ///
    /// For HLSL/D3D targets, the texture element type must be a scalar or vector of float or half types.
    ///
    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0_fragment)]
    T Sample(vector<float, Shape.dimensions+isArray> location)
    {
        __requireComputeDerivative();
        __target_switch
        {
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                return __getTexture().Sample(__getSampler(), location);
            case cpp:
            case metal:
                return __getTexture().Sample(__getSampler(), location);
            case glsl:
                __intrinsic_asm "$ctexture($0, $1)$z";
            case cuda:
                if (isArray != 0)
                {
                    switch(Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "tex1DLayered<$T0>($0, ($1).x, int(($1).y))";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "tex2DLayered<$T0>($0, ($1).x, ($1).y, int(($1).z))";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "texCubemapLayered<$T0>($0, ($1).x, ($1).y, ($1).z, int(($1).w))";
                    default: __intrinsic_asm "invalid texture shape";
                    }
                }
                else
                {
                    switch(Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "tex1D<$T0>($0, ($1))";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "tex2D<$T0>($0, ($1).x, ($1).y)";
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "tex3D<$T0>($0, ($1).x, ($1).y, ($1).z)";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "texCubemap<$T0>($0, ($1).x, ($1).y, ($1).z)";
                    default: __intrinsic_asm "invalid texture shape";
                    }
                }
            case spirv:
                return spirv_asm
                {
                    %sampled : __sampledType(T) = OpImageSampleImplicitLod $this $location None;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                return __getTexture().Sample(__getSampler(), location);
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0_fragment)]
    T Sample(vector<float, Shape.dimensions+isArray> location, constexpr vector<int, Shape.planeDimensions> offset)
    {
        __requireComputeDerivative();
        __target_switch
        {
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                return __getTexture().Sample(__getSampler(), location, offset);
            case cpp:
            case metal:
                return __getTexture().Sample(__getSampler(), location, offset);
            case glsl:
                __intrinsic_asm "$ctextureOffsetClampARB($0, $1, $2)$z";
            case spirv:
                return spirv_asm
                {
                    OpCapability MinLod;
                    %sampled : __sampledType(T) = OpImageSampleImplicitLod $this $location None|ConstOffset $offset;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                return __getTexture().Sample(__getSampler(), location, offset);
        }
    }

    [__readNone]
    [ForceInline]
    __glsl_extension(GL_ARB_sparse_texture_clamp)
    [require(cpp_glsl_hlsl_metal_spirv, texture_sm_4_1_clamp_fragment)]
    T Sample(vector<float, Shape.dimensions+isArray> location, vector<int, Shape.planeDimensions> offset, float clamp)
    {
        __requireComputeDerivative();
        __target_switch
        {
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                return __getTexture().Sample(__getSampler(), location, offset, clamp);
            case cpp:
            case metal:
                return __getTexture().Sample(__getSampler(), location, offset, clamp);
            case glsl:
                __intrinsic_asm "$ctextureOffsetClampARB($0, $1, $2, $3)$z";
            case spirv:
                return spirv_asm
                {
                    OpCapability MinLod;
                    %sampled : __sampledType(T) = OpImageSampleImplicitLod $this $location None|ConstOffset|MinLod $offset $clamp;
                    __truncate $$T result __sampledType(T) %sampled;
                };
        }
    }

    [__readNone]
    [ForceInline]
    T Sample(vector<float, Shape.dimensions+isArray> location, vector<int, Shape.planeDimensions> offset, float clamp, out uint status)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            __intrinsic_asm ".Sample";
        case spirv:
            return spirv_asm
            {
                OpCapability MinLod;
                OpCapability SparseResidency;
                %sparseResultType = OpTypeStruct $$uint __sampledType(T);

                %sparseResult:%sparseResultType = OpImageSparseSampleImplicitLod $this $location ConstOffset|MinLod $offset $clamp;
                %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                %sampled:__sampledType(T) = OpCompositeExtract %sparseResult 1;

                OpStore &status %residentCode;
                __truncate $$T result __sampledType(T) %sampled;
            };
        default:
            status = 0;
            return Sample(location, offset, clamp);
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0_fragment)]
    T SampleBias(vector<float, Shape.dimensions+isArray> location, float bias)
    {
        __requireComputeDerivative();
        __target_switch
        {
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                return __getTexture().SampleBias(__getSampler(), location, bias);
            case cpp:
            case metal:
                return __getTexture().SampleBias(__getSampler(), location, bias);
            case glsl:
                __intrinsic_asm "$ctexture($0, $1, $2)$z";
            case spirv:
                return spirv_asm
                {
                    %sampled : __sampledType(T) = OpImageSampleImplicitLod $this $location None|Bias $bias;
                    __truncate $$T result __sampledType(T) %sampled;

                };
            case wgsl:
                return __getTexture().SampleBias(__getSampler(), location, bias);
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0_fragment)]
    T SampleBias(vector<float, Shape.dimensions+isArray> location, float bias, constexpr vector<int, Shape.planeDimensions> offset)
    {
        __requireComputeDerivative();
        __target_switch
        {
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                return __getTexture().SampleBias(__getSampler(), location, bias, offset);
            case cpp:
            case metal:
                return __getTexture().SampleBias(__getSampler(), location, bias, offset);
            case glsl:
                __intrinsic_asm "$ctextureOffset($0, $1, $3, $2)$z";
            case spirv:
                return spirv_asm
                {
                    %sampled : __sampledType(T) = OpImageSampleImplicitLod $this $location None|Bias|ConstOffset $bias $offset;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                return __getTexture().SampleBias(__getSampler(), location, bias, offset);
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl_spirv, sm_5_0)]
    T SampleBias(vector<float, Shape.dimensions+isArray> location, float bias, constexpr vector<int, Shape.planeDimensions> offset, float clamp, out uint status)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            return __getTexture().SampleBias(__getSampler(), location, bias, offset, clamp, status);
        case spirv:
            return spirv_asm
            {
                OpCapability SparseResidency;
                %sparseResultType = OpTypeStruct $$uint __sampledType(T);

                %sparseResult:%sparseResultType = OpImageSparseSampleImplicitLod $this $location Bias|ConstOffset $bias $offset;
                %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                %sampled:__sampledType(T) = OpCompositeExtract  %sparseResult 1;

                OpStore &status %residentCode;
                __truncate $$T result __sampledType(T) %sampled;
            };
        }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv_wgsl, texture_shadowlod)]
    float SampleCmp(vector<float, Shape.dimensions+isArray> location, float compareValue)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case glsl:
            if (Shape.dimensions == 1 && isArray == 0)
            {
                return __glsl_texture_1d_shadow(this, __makeVector(__makeVector(location, 0.0), compareValue));
            }
            else if (Shape.dimensions == 3 && isArray == 1)
            {
                return __glsl_texture_3d_array_shadow(this, location, compareValue);
            }
            else
            {
                return __glsl_texture(this, __makeVector(location, compareValue));
            }
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            return __getTexture().SampleCmp(__getComparisonSampler(), location, compareValue);
        case metal:
            return __getTexture().SampleCmp(__getComparisonSampler(), location, compareValue);
        case spirv:
            return spirv_asm
            {
                result:$$float = OpImageSampleDrefImplicitLod $this $location $compareValue;
            };
        case wgsl:
            return __getTexture().SampleCmp(__getComparisonSampler(), location, compareValue);
        }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv_wgsl, texture_shadowlod)]
    float SampleCmpLevelZero(vector<float, Shape.dimensions+isArray> location, float compareValue)
    {
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            return __getTexture().SampleCmpLevelZero(__getComparisonSampler(), location, compareValue);
        case wgsl:
            return __getTexture().SampleCmpLevelZero(__getComparisonSampler(), location, compareValue);
        default:
            return SampleCmpLevel(location, compareValue, 0.0);
        }
    }

    [__readNone] 
    [ForceInline]
    [require(glsl_hlsl_metal_spirv_wgsl, texture_shadowlod)]
    float SampleCmp(vector<float, Shape.dimensions+isArray> location, float compareValue, constexpr vector<int, Shape.planeDimensions> offset)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case glsl:
            if (Shape.dimensions == 1 && isArray == 0)
            {
                return __glsl_texture_offset_1d_shadow(this, __makeVector(__makeVector(location, 0.0), compareValue), offset);
            }
            else
            {
                return __glsl_texture_offset(this, __makeVector(location, compareValue), offset);
            }
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            return __getTexture().SampleCmp(__getComparisonSampler(), location, compareValue, offset);
        case metal:
            return __getTexture().SampleCmp(__getComparisonSampler(), location, compareValue, offset);
        case spirv:
            return spirv_asm
            {
                result:$$float = OpImageSampleDrefImplicitLod $this $location $compareValue ConstOffset $offset;
            };
        case wgsl:
            return __getTexture().SampleCmp(__getComparisonSampler(), location, compareValue, offset);
        }
    }


    [__readNone] 
    [ForceInline]
    [require(hlsl_spirv, sm_5_0)]
    float SampleCmp(vector<float, Shape.dimensions+isArray> location, float compareValue, constexpr vector<int, Shape.planeDimensions> offset, float clamp, out uint status)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            return __getTexture().SampleCmp(__getComparisonSampler(), location, compareValue, offset, clamp, status);
        case spirv:
            return spirv_asm
            {
                OpCapability MinLod;
                OpCapability SparseResidency;
                %sparseResultType = OpTypeStruct $$uint $$float;

                %sparseResult:%sparseResultType = OpImageSparseSampleDrefImplicitLod $this $location $compareValue ConstOffset|MinLod $offset $clamp;

                %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                OpStore &status %residentCode;
                result:$$float = OpCompositeExtract %sparseResult 1;
            };
        }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv_wgsl, texture_shadowlod)]
    float SampleCmpLevelZero(vector<float, Shape.dimensions+isArray> location, float compareValue, constexpr vector<int, Shape.planeDimensions> offset)
    {
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            return __getTexture().SampleCmpLevelZero(__getComparisonSampler(), location, compareValue, offset);
        case wgsl:
            return __getTexture().SampleCmpLevelZero(__getComparisonSampler(), location, compareValue, offset);
        default:
            return SampleCmpLevel(location, compareValue, 0.0, offset);
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl_spirv, sm_5_0)]
    float SampleCmpLevelZero(vector<float, Shape.dimensions+isArray> location, float compareValue, constexpr vector<int, Shape.planeDimensions> offset, out uint status)
    {
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            return __getTexture().SampleCmpLevelZero(__getComparisonSampler(), location, compareValue, offset, status);
        case spirv:
            return SampleCmpLevel(location, compareValue, 0.0, offset, status);
        }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv, texture_shadowlod)]
    float SampleCmpLevel(vector<float, Shape.dimensions+isArray> location, float compareValue, float level)
    {
        __target_switch
        {
        case glsl:
            if (Shape.dimensions == 1 && isArray == 0)
            {
                return __glsl_texture_level_1d_shadow(this, __makeVector(__makeVector(location, 0.0), compareValue), level);
            }
            else
            {
                return __glsl_texture_level(this, __makeVector(location, compareValue), level);
            }
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            return __getTexture().SampleCmpLevel(__getComparisonSampler(), location, compareValue, level);
        case metal:
            return __getTexture().SampleCmpLevel(__getComparisonSampler(), location, compareValue, level);
        case spirv:
            return spirv_asm
            {
                result:$$float = OpImageSampleDrefExplicitLod $this $location $compareValue Lod $level;
            };
        }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv, texture_shadowlod)]
    float SampleCmpLevel(vector<float, Shape.dimensions+isArray> location, float compareValue, float level, constexpr vector<int, Shape.planeDimensions> offset)
    {
        __target_switch
        {
        case glsl:
            if (Shape.dimensions == 1 && isArray == 0)
            {
                return __glsl_texture_level_offset_1d_shadow(this, __makeVector(__makeVector(location,0.0), compareValue), level, offset);
            }
            else
            {
                return __glsl_texture_level_offset(this, __makeVector(location, compareValue), level, offset);
            }
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            return __getTexture().SampleCmpLevel(__getComparisonSampler(), location, compareValue, level, offset);
        case metal:
            return __getTexture().SampleCmpLevel(__getComparisonSampler(), location, compareValue, level, offset);
        case spirv:
            return spirv_asm
            {
                result:$$float = OpImageSampleDrefExplicitLod $this $location $compareValue Lod|ConstOffset $level $offset;
            };
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl_spirv, texture_shadowlod)]
    float SampleCmpLevel(vector<float, Shape.dimensions+isArray> location, float compareValue, float level, constexpr vector<int, Shape.planeDimensions> offset, out uint status)
    {
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            return __getTexture().SampleCmpLevel(__getComparisonSampler(), location, compareValue, level, offset, status);
        case spirv:
            return spirv_asm
            {
                OpCapability SparseResidency;
                %sparseResultType = OpTypeStruct $$uint $$float;

                %sparseResult:%sparseResultType = OpImageSparseSampleDrefExplicitLod $this $location $compareValue Lod|ConstOffset $level $offset;

                %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                OpStore &status %residentCode;
                result:$$float = OpCompositeExtract %sparseResult 1; 
            };
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0)]
    T SampleGrad(vector<float, Shape.dimensions+isArray> location, vector<float, Shape.dimensions> gradX, vector<float, Shape.dimensions> gradY)
    {
        __target_switch
        {
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                return __getTexture().SampleGrad(__getSampler(), location, gradX, gradY);
            case cpp:
            case metal:
                return __getTexture().SampleGrad(__getSampler(), location, gradX, gradY);
            case glsl:
                __intrinsic_asm "$ctextureGrad($0, $1, $2, $3)$z";
            case spirv:
                return spirv_asm
                {
                    %sampled : __sampledType(T) = OpImageSampleExplicitLod $this $location None|Grad $gradX $gradY;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                return __getTexture().SampleGrad(__getSampler(), location, gradX, gradY);
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0)]
    T SampleGrad(vector<float, Shape.dimensions+isArray> location, vector<float, Shape.dimensions> gradX, vector<float, Shape.dimensions> gradY, constexpr vector<int, Shape.dimensions> offset)
    {
        __target_switch
        {
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                return __getTexture().SampleGrad(__getSampler(), location, gradX, gradY, offset);
            case cpp:
            case metal:
                return __getTexture().SampleGrad(__getSampler(), location, gradX, gradY, offset);
            case glsl:
                __intrinsic_asm "$ctextureGradOffset($0, $1, $2, $3, $4)$z";
            case spirv:
                return spirv_asm
                {
                    %sampled : __sampledType(T) = OpImageSampleExplicitLod $this $location None|Grad|ConstOffset $gradX $gradY $offset;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                return __getTexture().SampleGrad(__getSampler(), location, gradX, gradY, offset);
        }
    }

    [__readNone]
    [ForceInline]
    __glsl_extension(GL_ARB_sparse_texture_clamp)
    [require(cpp_glsl_hlsl_metal_spirv, texture_sm_4_1_clamp_fragment)]
    T SampleGrad(vector<float, Shape.dimensions+isArray> location, vector<float, Shape.dimensions> gradX, vector<float, Shape.dimensions> gradY, constexpr vector<int, Shape.dimensions> offset, float lodClamp)
    {
        __target_switch
        {
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                return __getTexture().SampleGrad(__getSampler(), location, gradX, gradY, offset, lodClamp);
            case cpp:
            case metal:
                return __getTexture().SampleGrad(__getSampler(), location, gradX, gradY, offset, lodClamp);
            case glsl:
            __intrinsic_asm "$ctextureGradOffsetClampARB($0, $1, $2, $3, $4, $5)$z";
            case spirv:
                return spirv_asm
                {
                    OpCapability MinLod;
                    %sampled : __sampledType(T) = OpImageSampleExplicitLod $this $location None|Grad|ConstOffset|MinLod $gradX $gradY $offset $lodClamp;
                    __truncate $$T result __sampledType(T) %sampled;
                };
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl_spirv, sm_5_0)]
    T SampleGrad(vector<float, Shape.dimensions+isArray> location, vector<float, Shape.dimensions> gradX, vector<float, Shape.dimensions> gradY, constexpr vector<int, Shape.dimensions> offset, float lodClamp, out uint status)
    {
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            return __getTexture().SampleGrad(__getSampler(), location, gradX, gradY, offset, lodClamp, status);
        case spirv:
            return spirv_asm
            {
                OpCapability MinLod;
                OpCapability SparseResidency;
                %sparseResultType = OpTypeStruct $$uint __sampledType(T);

                %sparseResult:%sparseResultType = OpImageSparseSampleExplicitLod $this $location Grad|ConstOffset|MinLod $gradX $gradY $offset $lodClamp;
                %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                %sampled:__sampledType(T) = OpCompositeExtract %sparseResult 1; 

                OpStore &status %residentCode;
                __truncate $$T result __sampledType(T) %sampled;
            };
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0)]
    T SampleLevel(vector<float, Shape.dimensions+isArray> location, float level)
    {
        __target_switch
        {
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                return __getTexture().SampleLevel(__getSampler(), location, level);
            case cpp:
            case metal:
                return __getTexture().SampleLevel(__getSampler(), location, level);
            case glsl:
                __intrinsic_asm "$ctextureLod($0, $1, $2)$z";
            case cuda:
                if (isArray != 0)
                {
                    switch(Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "tex1DLayeredLod<$T0>($0, ($1).x, int(($1).y), ($2))";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "tex2DLayeredLod<$T0>($0, ($1).x, ($1).y, int(($1).z), ($2))";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "texCubemapLayeredLod<$T0>($0, ($1).x, ($1).y, ($1).z, int(($1).w), ($2))";
                    default:
                        __intrinsic_asm "<invalid intrinsic>";
                    }
                }
                else
                {
                    switch(Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "tex1DLod<$T0>($0, ($1), ($2))";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "tex2DLod<$T0>($0, ($1).x, ($1).y, ($2))";
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "tex3DLod<$T0>($0, ($1).x, ($1).y, ($1).z, ($2))";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "texCubemapLod<$T0>($0, ($1).x, ($1).y, ($1).z, ($2))";
                    default:
                        __intrinsic_asm "<invalid intrinsic>";
                    }
                }
            case spirv:
                return spirv_asm
                {
                    %sampled : __sampledType(T) = OpImageSampleExplicitLod $this $location None|Lod $level;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                return __getTexture().SampleLevel(__getSampler(), location, level);
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0)]
    T SampleLevel(vector<float, Shape.dimensions+isArray> location, float level, constexpr vector<int, Shape.planeDimensions> offset)
    {
        __target_switch
        {
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                return __getTexture().SampleLevel(__getSampler(), location, level, offset);
            case cpp:
            case metal:
                return __getTexture().SampleLevel(__getSampler(), location, level, offset);
            case glsl:
                __intrinsic_asm "$ctextureLodOffset($0, $1, $2, $3)$z";
            case spirv:
                return spirv_asm
                {
                    %sampled : __sampledType(T) = OpImageSampleExplicitLod $this $location None|Lod|ConstOffset $level $offset;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                return __getTexture().SampleLevel(__getSampler(), location, level, offset);
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl_spirv, sm_5_0)]
    T SampleLevel(vector<float, Shape.dimensions+isArray> location, float level, constexpr vector<int, Shape.planeDimensions> offset, out uint status)
    {
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            return __getTexture().SampleLevel(__getSampler(), location, level, offset, status);
        case spirv:
            return spirv_asm
            {
                OpCapability SparseResidency;
                %sparseResultType = OpTypeStruct $$uint __sampledType(T);

                %sparseResult:%sparseResultType = OpImageSparseSampleExplicitLod $this $location Lod|ConstOffset $level $offset;
                %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                %sampled:__sampledType(T) = OpCompositeExtract %sparseResult 1; 

                OpStore &status %residentCode;
                __truncate $$T result __sampledType(T) %sampled;
            };
        }
    }
}

// Non-combined texture types specific functions

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let isMS:int, let sampleCount:int, let access:int, let isShadow:int, let format:int>
extension _Texture<T,Shape,isArray,isMS,sampleCount,access,isShadow,0,format>
{
    typealias TextureCoord = vector<float, Shape.dimensions>;

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv, texture_querylod)]
    float CalculateLevelOfDetail(SamplerState s, TextureCoord location)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case hlsl:
            __intrinsic_asm ".CalculateLevelOfDetail";
        case metal:
            __intrinsic_asm ".calculate_clamped_lod";
        case glsl:
            __intrinsic_asm "textureQueryLod($p, $2).x";
        case spirv:
            return (spirv_asm {
                OpCapability ImageQuery;
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                result:$$float2 = OpImageQueryLod %sampledImage $location;
            }).x;
        }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv, texture_querylod)]
    float CalculateLevelOfDetailUnclamped(SamplerState s, TextureCoord location)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case hlsl:
            __intrinsic_asm ".CalculateLevelOfDetailUnclamped";
        case metal:
            __intrinsic_asm ".calculate_unclamped_lod";
        case glsl:
            __intrinsic_asm "textureQueryLod($p, $2).y";
        case spirv:
            return (spirv_asm {
                OpCapability ImageQuery;
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                result:$$float2 = OpImageQueryLod %sampledImage $location;
            }).y;
        }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv, texture_querylod)]
    float CalculateLevelOfDetail(SamplerComparisonState s, TextureCoord location)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case hlsl:
            __intrinsic_asm ".CalculateLevelOfDetail";
        case metal:
            __intrinsic_asm ".calculate_clamped_lod";
        case glsl:
            __intrinsic_asm "textureQueryLod($p, $2).x";
        case spirv:
            return (spirv_asm {
                OpCapability ImageQuery;
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                result:$$float2 = OpImageQueryLod %sampledImage $location;
            }).x;
        }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv, texture_querylod)]
    float CalculateLevelOfDetailUnclamped(SamplerComparisonState s, TextureCoord location)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case hlsl:
            __intrinsic_asm ".CalculateLevelOfDetailUnclamped";
        case metal:
            __intrinsic_asm ".calculate_unclamped_lod";
        case glsl:
            __intrinsic_asm "textureQueryLod($p, $2).y";
        case spirv:
            return (spirv_asm {
                OpCapability ImageQuery;
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                result:$$float2 = OpImageQueryLod %sampledImage $location;
            }).y;
        }
    }
}
__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let isMS:int, let sampleCount:int, let format:int>
float __metal_SampleCmp(_Texture<T,Shape,isArray,isMS,sampleCount,0,1,0,format> t, SamplerComparisonState s, vector<float, Shape.dimensions+isArray> location, float compareValue)
{
    if (isArray == 1)
    {
        switch (Shape.flavor)
        {
        case $(SLANG_TEXTURE_2D):
            __intrinsic_asm "$0.sample_compare($1, ($2).xy, uint(($2).z), $3)";
        case $(SLANG_TEXTURE_CUBE):
            __intrinsic_asm "$0.sample_compare($1, ($2).xyz, uint(($2).w), $3)";
        }
    }
    else
    {
        switch (Shape.flavor)
        {
        case $(SLANG_TEXTURE_2D):
        case $(SLANG_TEXTURE_CUBE):
            __intrinsic_asm ".sample_compare";
        }
    }
    __intrinsic_asm "<invalid intrinsic>";
}
__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let isMS:int, let sampleCount:int, let format:int>
float __metal_SampleCmp(_Texture<T,Shape,isArray,isMS,sampleCount,0,1,0,format> t, SamplerComparisonState s, vector<float, Shape.dimensions+isArray> location, float compareValue, constexpr vector<int, Shape.planeDimensions> offset)
{
    if (isArray == 1)
    {
        switch (Shape.flavor)
        {
        case $(SLANG_TEXTURE_2D):
            __intrinsic_asm "$0.sample_compare($1, ($2).xy, uint(($2).z), $3, $4)";
        }
    }
    else
    {
        switch (Shape.flavor)
        {
        case $(SLANG_TEXTURE_2D):
            __intrinsic_asm "$0.sample_compare($1, $2, $3, $4)";
        }
    }
    __intrinsic_asm "<invalid intrinsic>";
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let isMS:int, let sampleCount:int, let format:int>
float __metal_SampleCmpLevel(_Texture<T,Shape,isArray,isMS,sampleCount,0,1,0,format> t, SamplerComparisonState s, vector<float, Shape.dimensions+isArray> location, float compareValue, float level)
{
    if (isArray == 1)
    {
        switch (Shape.flavor)
        {
        case $(SLANG_TEXTURE_2D):
            __intrinsic_asm "$0.sample_compare($1, ($2).xy, uint(($2).z), $3, level($4))";
        case $(SLANG_TEXTURE_CUBE):
            __intrinsic_asm "$0.sample_compare($1, ($2).xyz, uint(($2).w), $3, level($4))";
        }
    }
    else
    {
        switch (Shape.flavor)
        {
        case $(SLANG_TEXTURE_2D):
        case $(SLANG_TEXTURE_CUBE):
            __intrinsic_asm "$0.sample_compare($1, $2, $3, level($4))";
        }
    }
    __intrinsic_asm "<invalid intrinsic>";
}
__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let isMS:int, let sampleCount:int, let format:int>
float __metal_SampleCmpLevel(_Texture<T,Shape,isArray,isMS,sampleCount,0,1,0,format> t, SamplerComparisonState s, vector<float, Shape.dimensions+isArray> location, float compareValue, float level, constexpr vector<int, Shape.planeDimensions> offset)
{
    switch (Shape.flavor)
    {
    case $(SLANG_TEXTURE_2D):
        if (isArray == 1)
        {
            // T sample_compare(sampler s, float2 coord, uint array, float compare_value, lod_options options, int2 offset = int2(0)) const
            __intrinsic_asm "$0.sample_compare($1, ($2).xy, uint(($2).z), $3, level($4), $5)";
        }
        else
        {
            // T sample_compare(sampler s, float2 coord, float compare_value, lod_options options, int2 offset = int2(0)) const
            __intrinsic_asm "$0.sample_compare($1, $2, $3, level($4), $5)";
        }
        break;
    }
    __intrinsic_asm "<invalid intrinsic>";
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let isMS:int, let sampleCount:int, let isShadow:int, let format:int>
extension _Texture<T,Shape,isArray,isMS,sampleCount,0,isShadow,0,format>
{
    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0_fragment)]
    T Sample(SamplerState s, vector<float, Shape.dimensions+isArray> location)
    {
        __requireComputeDerivative();
        __target_switch
        {
            case cpp:
                __intrinsic_asm ".Sample";
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                __intrinsic_asm ".Sample";
            case metal:
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "$c$0.sample($1, ($2).x, uint(($2).y))$z";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "$c$0.sample($1, ($2).xy, uint(($2).z))$z";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "$c$0.sample($1, ($2).xyz, uint(($2).w))$z";
                    }
                }
                else
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                    case $(SLANG_TEXTURE_2D):
                    case $(SLANG_TEXTURE_3D):
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "$c$0.sample($1, $2)$z";
                    }
                }
                // TODO: This needs to be handled by the capability system
                __intrinsic_asm "<invalid intrinsic>";
            case glsl:
                __intrinsic_asm "$ctexture($p, $2)$z";
            case cuda:
                if (isArray != 0)
                {
                    switch(Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "tex1DLayered<$T0>($0, ($2).x, int(($2).y))";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "tex2DLayered<$T0>($0, ($2).x, ($2).y, int(($2).z))";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "texCubemapLayered<$T0>($0, ($2).x, ($2).y, ($2).z, int(($2).w))";
                    default:
                        __intrinsic_asm "<invalid intrinsic>";
                    }
                }
                else
                {
                    switch(Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "tex1D<$T0>($0, ($2))";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "tex2D<$T0>($0, ($2).x, ($2).y)";
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "tex3D<$T0>($0, ($2).x, ($2).y, ($2).z)";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "texCubemap<$T0>($0, ($2).x, ($2).y, ($2).z)";
                    default:
                        __intrinsic_asm "<invalid intrinsic>";
                    }
                }
            case spirv:
                return spirv_asm
                {
                    %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                    %sampled : __sampledType(T) = OpImageSampleImplicitLod %sampledImage $location None;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    , "WGSL supports only f32 type textures");
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "textureSample($0, $1, ($2).x, i32(($2).y))$z";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "textureSample($0, $1, ($2).xy, i32(($2).z))$z";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "textureSample($0, $1, ($2).xyz, i32(($2).w))$z";
                    }
                }
                __intrinsic_asm "textureSample($0, $1, $2)$z";
        }
    }


    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0_fragment)]
    T Sample(SamplerState s, vector<float, Shape.dimensions+isArray> location, constexpr vector<int, Shape.planeDimensions> offset)
    {
        __requireComputeDerivative();
        __target_switch
        {
            case cpp:
                __intrinsic_asm ".Sample";
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                __intrinsic_asm ".Sample";
            case metal:
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "$c$0.sample($1, ($2).xy, uint(($2).z), $3)$z";
                    }
                }
                else
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "$c$0.sample($1, $2, $3)$z";
                    }
                }
                // TODO: This needs to be handled by the capability system
                __intrinsic_asm "<invalid intrinsic>";
            case glsl:
                __intrinsic_asm "$ctextureOffset($p, $2, $3)$z";
            case spirv:
                return spirv_asm
                {
                    %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                    %sampled : __sampledType(T) = OpImageSampleImplicitLod %sampledImage $location None|ConstOffset $offset;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    , "WGSL supports only f32 type textures");
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "textureSample($0, $1, ($2).x, i32(($2).y), $3)$z";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "textureSample($0, $1, ($2).xy, i32(($2).z), $3)$z";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "textureSample($0, $1, ($2).xyz, i32(($2).w), $3)$z";
                    }
                }
                __intrinsic_asm "textureSample($0, $1, $2, $3)$z";
        }
    }

    [__readNone]
    [ForceInline]
    __glsl_extension(GL_ARB_sparse_texture_clamp)
    [require(cpp_glsl_hlsl_metal_spirv, texture_sm_4_0_fragment)]
    T Sample(SamplerState s, vector<float, Shape.dimensions+isArray> location, constexpr vector<int, Shape.planeDimensions> offset, float clamp)
    {
        __requireComputeDerivative();
        __target_switch
        {
            case cpp:
                __intrinsic_asm ".Sample";
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                __intrinsic_asm ".Sample";
            case metal:
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "$c$0.sample($1, ($2).xy, uint(($2).z), min_lod_clamp($4), $3)$z";
                    }
                }
                else
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "$c$0.sample($1, $2, min_lod_clamp($4), $3)$z";
                    }
                }
                // TODO: This needs to be handled by the capability system
                __intrinsic_asm "<invalid intrinsic>";
            case glsl:
                __intrinsic_asm "$ctextureOffsetClampARB($p, $2, $3, $4)$z";
            case spirv:
                return spirv_asm
                {
                    OpCapability MinLod;
                    %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                    %sampled : __sampledType(T) = OpImageSampleImplicitLod %sampledImage $location None|ConstOffset|MinLod $offset $clamp;
                    __truncate $$T result __sampledType(T) %sampled;
                };
        }
    }

    [__readNone]
    [ForceInline]
    T Sample(SamplerState s, vector<float, Shape.dimensions+isArray> location, constexpr vector<int, Shape.planeDimensions> offset, float clamp, out uint status)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            __intrinsic_asm ".Sample";
        case spirv:
            return spirv_asm
            {
                OpCapability MinLod;
                OpCapability SparseResidency;
                %sparseResultType = OpTypeStruct $$uint __sampledType(T);
                %sampledImage:__sampledImageType(this) = OpSampledImage $this $s;

                %sparseResult:%sparseResultType = OpImageSparseSampleImplicitLod %sampledImage $location ConstOffset|MinLod $offset $clamp;
                %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                %sampled:__sampledType(T) = OpCompositeExtract %sparseResult 1;

                OpStore &status %residentCode;
                __truncate $$T result __sampledType(T) %sampled;
            };
        default:
            status = 0;
            return Sample(s, location, offset, clamp);
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0_fragment)]
    T SampleBias(SamplerState s, vector<float, Shape.dimensions+isArray> location, float bias)
    {
        __requireComputeDerivative();
        __target_switch
        {
            case cpp:
                __intrinsic_asm ".SampleBias";
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                __intrinsic_asm ".SampleBias";
            case metal:
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "$c$0.sample($1, ($2).xy, uint(($2).z), bias($3))$z";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "$c$0.sample($1, ($2).xyz, uint(($2).w), bias($3))$z";
                    }
                }
                else
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                    case $(SLANG_TEXTURE_3D):
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "$c$0.sample($1, $2, bias($3))$z";
                    }
                }
                // TODO: This needs to be handled by the capability system
                __intrinsic_asm "<invalid intrinsic>";
            case glsl:
                __intrinsic_asm "$ctexture($p, $2, $3)$z";
            case spirv:
                return spirv_asm
                {
                    %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                    %sampled : __sampledType(T) = OpImageSampleImplicitLod %sampledImage $location None|Bias $bias;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    , "WGSL supports only f32 type textures");
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "textureSampleBias($0, $1, ($2).x, i32(($2).y), $3)$z";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "textureSampleBias($0, $1, ($2).xy, i32(($2).z), $3)$z";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "textureSampleBias($0, $1, ($2).xyz, i32(($2).w), $3)$z";
                    }
                }
                __intrinsic_asm "textureSampleBias($0, $1, $2, $3)$z";
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0_fragment)]
    T SampleBias(SamplerState s, vector<float, Shape.dimensions+isArray> location, float bias, constexpr vector<int, Shape.planeDimensions> offset)
    {
        __requireComputeDerivative();
        __target_switch
        {
            case cpp:
                __intrinsic_asm ".SampleBias";
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                __intrinsic_asm ".SampleBias";
            case metal:
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "$c$0.sample($1, ($2).xy, uint(($2).z), bias($3), $4)$z";
                    }
                }
                else
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "$c$0.sample($1, $2, bias($3), $4)$z";
                    }
                }
                // TODO: This needs to be handled by the capability system
                __intrinsic_asm "<invalid intrinsic>";
            case glsl:
                __intrinsic_asm "$ctextureOffset($p, $2, $4, $3)$z";
            case spirv:
                return spirv_asm
                {
                    %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                    %sampled : __sampledType(T) = OpImageSampleImplicitLod %sampledImage $location None|Bias|ConstOffset $bias $offset;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    , "WGSL supports only f32 type textures");
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "textureSampleBias($0, $1, ($2).x, i32(($2).y), $3, $4)$z";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "textureSampleBias($0, $1, ($2).xy, i32(($2).z), $3, $4)$z";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "textureSampleBias($0, $1, ($2).xyz, i32(($2).w), $3, $4)$z";
                    }
                }
                __intrinsic_asm "textureSampleBias($0, $1, $2, $3, $4)$z";
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl_spirv, sm_5_0)]
    T SampleBias(SamplerState s, vector<float, Shape.dimensions+isArray> location, float bias, constexpr vector<int, Shape.planeDimensions> offset, float clamp, out uint status)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            __intrinsic_asm ".SampleBias";
        case spirv:
            return spirv_asm
            {
                OpCapability MinLod;
                OpCapability SparseResidency;
                %sparseResultType = OpTypeStruct $$uint __sampledType(T);
                %sampledImage:__sampledImageType(this) = OpSampledImage $this $s;

                %sparseResult:%sparseResultType = OpImageSparseSampleImplicitLod %sampledImage $location Bias|ConstOffset|MinLod $bias $offset $clamp;
                %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                %sampled:__sampledType(T) = OpCompositeExtract %sparseResult 1;

                OpStore &status %residentCode;
                __truncate $$T result __sampledType(T) %sampled;
            };
       }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv_wgsl, texture_shadowlod)]
    float SampleCmp(SamplerComparisonState s, vector<float, Shape.dimensions+isArray> location, float compareValue)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case glsl:
            if (Shape.dimensions == 1 && isArray == 0)
            {
                return __glsl_texture_1d_shadow(this, s, __makeVector(__makeVector(location, 0.0), compareValue));
            }
            else if (Shape.dimensions == 3 && isArray == 1)
            {
                return __glsl_texture_3d_array_shadow(this, s, location, compareValue);
            }
            else
            {
                return __glsl_texture(this, s, __makeVector(location,compareValue));
            }
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            __intrinsic_asm ".SampleCmp";
        case metal:
            return __metal_SampleCmp(__metal_asDepthTexture(this), s, location, compareValue);
        case spirv:
            return spirv_asm
            {
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                result:$$float = OpImageSampleDrefImplicitLod %sampledImage $location $compareValue;
            };
        case wgsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                , "WGSL supports only f32 type textures");
            if (isArray == 1)
            {
                switch (Shape.flavor)
                {
                case $(SLANG_TEXTURE_1D):
                    __intrinsic_asm "textureSampleCompare($0, $1, ($2).x, i32(($2).y), $3)";
                case $(SLANG_TEXTURE_2D):
                    __intrinsic_asm "textureSampleCompare($0, $1, ($2).xy, i32(($2).z), $3)";
                case $(SLANG_TEXTURE_CUBE):
                    __intrinsic_asm "textureSampleCompare($0, $1, ($2).xyz, i32(($2).w), $3)";
                }
            }
            __intrinsic_asm "textureSampleCompare($0, $1, $2, $3)";
        }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv_wgsl, texture_shadowlod)]
    float SampleCmpLevelZero(SamplerComparisonState s, vector<float, Shape.dimensions+isArray> location, float compareValue)
    {
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            __intrinsic_asm ".SampleCmpLevelZero";
        case wgsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                , "WGSL supports only f32 type textures");
            if (isArray == 1)
            {
                switch (Shape.flavor)
                {
                case $(SLANG_TEXTURE_1D):
                    __intrinsic_asm "textureSampleCompareLevel($0, $1, ($2).x, i32(($2).y), $3)";
                case $(SLANG_TEXTURE_2D):
                    __intrinsic_asm "textureSampleCompareLevel($0, $1, ($2).xy, i32(($2).z), $3)";
                case $(SLANG_TEXTURE_CUBE):
                    __intrinsic_asm "textureSampleCompareLevel($0, $1, ($2).xyz, i32(($2).w), $3)";
                }
            }
            __intrinsic_asm "textureSampleCompareLevel($0, $1, $2, $3)";
        default:
            return SampleCmpLevel(s, location, compareValue, 0.0);
        }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv_wgsl, texture_shadowlod)]
    float SampleCmp(SamplerComparisonState s, vector<float, Shape.dimensions+isArray> location, float compareValue, constexpr vector<int, Shape.planeDimensions> offset)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case glsl:
            if (Shape.dimensions == 1 && isArray == 0)
            {
                return __glsl_texture_offset_1d_shadow(this, s, __makeVector(__makeVector(location, 0.0), compareValue), offset);
            }
            else
            {
                return __glsl_texture_offset(this, s, __makeVector(location,compareValue), offset);
            }
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            __intrinsic_asm ".SampleCmp";
        case metal:
            return __metal_SampleCmp(__metal_asDepthTexture(this), s, location, compareValue, offset);
        case spirv:
            return spirv_asm
            {
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                result:$$float = OpImageSampleDrefImplicitLod %sampledImage $location $compareValue ConstOffset $offset;
            };
        case wgsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                , "WGSL supports only f32 type textures");
            if (isArray == 1)
            {
                switch (Shape.flavor)
                {
                case $(SLANG_TEXTURE_1D):
                    __intrinsic_asm "textureSampleCompare($0, $1, ($2).x, i32(($2).y), $3, $4)";
                case $(SLANG_TEXTURE_2D):
                    __intrinsic_asm "textureSampleCompare($0, $1, ($2).xy, i32(($2).z), $3, $4)";
                case $(SLANG_TEXTURE_CUBE):
                    __intrinsic_asm "textureSampleCompare($0, $1, ($2).xyz, i32(($2).w), $3, $4)";
                }
            }
            __intrinsic_asm "textureSampleCompare($0, $1, $2, $3, $4)";
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl_spirv, sm_5_0)]
    float SampleCmp(SamplerComparisonState s, vector<float, Shape.dimensions+isArray> location, float compareValue, constexpr vector<int, Shape.planeDimensions> offset, float clamp, out uint status)
    {
        __requireComputeDerivative();
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            __intrinsic_asm ".SampleCmp";
        case spirv:
            return spirv_asm
            {
                OpCapability MinLod;
                OpCapability SparseResidency;
                %sparseResultType = OpTypeStruct $$uint $$float;
                %sampledImage:__sampledImageType(this) = OpSampledImage $this $s;

                %sparseResult:%sparseResultType = OpImageSparseSampleDrefImplicitLod %sampledImage $location $compareValue ConstOffset|MinLod $offset $clamp;

                %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                OpStore &status %residentCode;
                result:$$float = OpCompositeExtract %sparseResult 1;
            };
        }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv_wgsl, texture_shadowlod)]
    float SampleCmpLevelZero(SamplerComparisonState s, vector<float, Shape.dimensions+isArray> location, float compareValue, constexpr vector<int, Shape.planeDimensions> offset)
    {
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            __intrinsic_asm ".SampleCmpLevelZero";
        case wgsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                , "WGSL supports only f32 type textures");
            if (isArray == 1)
            {
                switch (Shape.flavor)
                {
                case $(SLANG_TEXTURE_1D):
                    __intrinsic_asm "textureSampleCompareLevel($0, $1, ($2).x, i32(($2).y), $3, $4)";
                case $(SLANG_TEXTURE_2D):
                    __intrinsic_asm "textureSampleCompareLevel($0, $1, ($2).xy, i32(($2).z), $3, $4)";
                case $(SLANG_TEXTURE_CUBE):
                    __intrinsic_asm "textureSampleCompareLevel($0, $1, ($2).xyz, i32(($2).w), $3, $4)";
                }
            }
            __intrinsic_asm "textureSampleCompareLevel($0, $1, $2, $3, $4)";
        default:
            return SampleCmpLevel(s, location, compareValue, 0.0, offset);
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl_spirv, sm_5_0)]
    float SampleCmpLevelZero(SamplerComparisonState s, vector<float, Shape.dimensions+isArray> location, float compareValue, constexpr vector<int, Shape.planeDimensions> offset, out uint status)
    {
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            __intrinsic_asm ".SampleCmpLevelZero";
        case spirv:
            return SampleCmpLevel(s, location, compareValue, 0.0, offset, status);
        }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv, texture_shadowlod)]
    float SampleCmpLevel(SamplerComparisonState s, vector<float, Shape.dimensions+isArray> location, float compareValue, float level)
    {
        __target_switch
        {
        case glsl:
            if (Shape.dimensions == 1 && isArray == 0)
            {
                return __glsl_texture_level_1d_shadow(this, s, __makeVector(__makeVector(location, 0.0), compareValue), level);
            }
            else
            {
                return __glsl_texture_level(this, s, __makeVector(location,compareValue), level);
            }
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            __intrinsic_asm ".SampleCmpLevel";
        case metal:
            return __metal_SampleCmpLevel(__metal_asDepthTexture(this), s, location, compareValue, level);
        case spirv:
            return spirv_asm
            {
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                result:$$float = OpImageSampleDrefExplicitLod %sampledImage $location $compareValue Lod $level;
            };
        }
    }

    [__readNone]
    [ForceInline]
    [require(glsl_hlsl_metal_spirv, texture_shadowlod)]
    float SampleCmpLevel(SamplerComparisonState s, vector<float, Shape.dimensions+isArray> location, float compareValue, float level, constexpr vector<int, Shape.planeDimensions> offset)
    {
        __target_switch
        {
        case glsl:
            if (Shape.dimensions == 1 && isArray == 0)
            {
                return __glsl_texture_level_offset_1d_shadow(this, s, __makeVector(__makeVector(location,0.0),compareValue), level, offset);
            }
            else
            {
                return __glsl_texture_level_offset(this, s, __makeVector(location,compareValue), level, offset);
            }
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            __intrinsic_asm ".SampleCmpLevel";
        case metal:
            return __metal_SampleCmpLevel(__metal_asDepthTexture(this), s, location, compareValue, level, offset);
        case spirv:
            return spirv_asm
            {
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                result:$$float = OpImageSampleDrefExplicitLod %sampledImage $location $compareValue Lod|ConstOffset $level $offset;
            };
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl_spirv, texture_shadowlod)]
    float SampleCmpLevel(SamplerComparisonState s, vector<float, Shape.dimensions+isArray> location, float compareValue, float level, constexpr vector<int, Shape.planeDimensions> offset, out uint status)
    {
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            __intrinsic_asm ".SampleCmpLevel";
        case spirv:
            return spirv_asm
            {
                OpCapability SparseResidency;
                %sparseResultType = OpTypeStruct $$uint $$float;
                %sampledImage:__sampledImageType(this) = OpSampledImage $this $s;

                %sparseResult:%sparseResultType = OpImageSparseSampleDrefExplicitLod %sampledImage $location $compareValue Lod|ConstOffset $level $offset;

                %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                OpStore &status %residentCode;
                result:$$float = OpCompositeExtract %sparseResult 1;
            };
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0)]
    T SampleGrad(SamplerState s, vector<float, Shape.dimensions+isArray> location, vector<float, Shape.dimensions> gradX, vector<float, Shape.dimensions> gradY)
    {
        __target_switch
        {
            case cpp:
                __intrinsic_asm ".SampleGrad";
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                __intrinsic_asm ".SampleGrad";
            case metal:
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "$c$0.sample($1, ($2).xy, uint(($2).z), gradient2d($3, $4))$z";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "$c$0.sample($1, ($2).xyz, uint(($2).w), gradientcube($3, $4))$z";
                    }
                }
                else
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "$c$0.sample($1, $2, gradient2d($3, $4))$z";
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "$c$0.sample($1, $2, gradient3d($3, $4))$z";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "$c$0.sample($1, $2, gradientcube($3, $4))$z";
                    }
                }
                // TODO: This needs to be handled by the capability system
                __intrinsic_asm "<invalid intrinsic>";
            case glsl:
            __intrinsic_asm "$ctextureGrad($p, $2, $3, $4)$z";
            case spirv:
                return spirv_asm
                {
                    %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                    %sampled : __sampledType(T) = OpImageSampleExplicitLod %sampledImage $location None|Grad $gradX $gradY;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    , "WGSL supports only f32 type textures");
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "textureSampleGrad($0, $1, ($2).x, i32(($2).y), $3, $4)$z";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "textureSampleGrad($0, $1, ($2).xy, i32(($2).z), $3, $4)$z";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "textureSampleGrad($0, $1, ($2).xyz, i32(($2).w), $3, $4)$z";
                    }
                }
                __intrinsic_asm "textureSampleGrad($0, $1, $2, $3, $4)$z";
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0)]
    T SampleGrad(SamplerState s, vector<float, Shape.dimensions+isArray> location, vector<float, Shape.dimensions> gradX, vector<float, Shape.dimensions> gradY, constexpr vector<int, Shape.dimensions> offset)
    {
        __target_switch
        {
            case cpp:
                __intrinsic_asm ".SampleGrad";
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                __intrinsic_asm ".SampleGrad";
            case metal:
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "$c$0.sample($1, ($2).xy, uint(($2).z), gradient2d($3, $4), $5)$z";
                    }
                }
                else
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "$c$0.sample($1, $2, gradient2d($3, $4), $5)$z";
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "$c$0.sample($1, $2, gradient3d($3, $4), $5)$z";
                    }
                }
                // TODO: This needs to be handled by the capability system
                __intrinsic_asm "<invalid intrinsic>";
            case glsl:
            __intrinsic_asm "$ctextureGradOffset($p, $2, $3, $4, $5)$z";
            case spirv:
                return spirv_asm
                {
                    %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                    %sampled : __sampledType(T) = OpImageSampleExplicitLod %sampledImage $location None|Grad|ConstOffset $gradX $gradY $offset;
                    __truncate $$T result __sampledType(T) %sampled;

                };
            case wgsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    , "WGSL supports only f32 type textures");
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "textureSampleGrad($0, $1, ($2).x, i32(($2).y), $3, $4, $5)$z";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "textureSampleGrad($0, $1, ($2).xy, i32(($2).z), $3, $4, $5)$z";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "textureSampleGrad($0, $1, ($2).xyz, i32(($2).w), $3, $4, $5)$z";
                    }
                }
                __intrinsic_asm "textureSampleGrad($0, $1, $2, $3, $4, $5)$z";
        }
    }

    [__readNone]
    [ForceInline]
    __glsl_extension(GL_ARB_sparse_texture_clamp)
    [require(cpp_glsl_hlsl_metal_spirv, texture_sm_4_0)]
    T SampleGrad(SamplerState s, vector<float, Shape.dimensions+isArray> location, vector<float, Shape.dimensions> gradX, vector<float, Shape.dimensions> gradY, constexpr vector<int, Shape.dimensions> offset, float lodClamp)
    {
        __target_switch
        {
            case cpp:
                __intrinsic_asm ".SampleGrad";
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                __intrinsic_asm ".SampleGrad";
            case metal:
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "$c$0.sample($1, ($2).xy, uint(($2).z), gradient2d($3, $4), min_lod_clamp($6), $5)$z";
                    }
                }
                else
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "$c$0.sample($1, $2, gradient2d($3, $4), min_lod_clamp($6), $5)$z";
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "$c$0.sample($1, $2, gradient3d($3, $4), min_lod_clamp($6), $5)$z";
                    }
                }
                // TODO: This needs to be handled by the capability system
                __intrinsic_asm "<invalid intrinsic>";
            case glsl:
            __intrinsic_asm "$ctextureGradOffsetClampARB($p, $2, $3, $4, $5, $6)$z";
            case spirv:
                return spirv_asm
                {
                    OpCapability MinLod;
                    %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                    %sampled : __sampledType(T) = OpImageSampleExplicitLod %sampledImage $location None|Grad|ConstOffset|MinLod $gradX $gradY $offset $lodClamp;
                    __truncate $$T result __sampledType(T) %sampled;
                };
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl_spirv, sm_5_0)]
    T SampleGrad(SamplerState s, vector<float, Shape.dimensions+isArray> location, vector<float, Shape.dimensions> gradX, vector<float, Shape.dimensions> gradY, constexpr vector<int, Shape.dimensions> offset, float lodClamp, out uint status)
    {
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            __intrinsic_asm ".SampleGrad";
        case spirv:
            return spirv_asm
            {
                OpCapability MinLod;
                OpCapability SparseResidency;
                %sparseResultType = OpTypeStruct $$uint __sampledType(T);
                %sampledImage:__sampledImageType(this) = OpSampledImage $this $s;

                %sparseResult:%sparseResultType = OpImageSparseSampleExplicitLod %sampledImage $location Grad|ConstOffset|MinLod $gradX $gradY $offset $lodClamp;
                %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                %sampled:__sampledType(T) = OpCompositeExtract %sparseResult 1;

                OpStore &status %residentCode;
                __truncate $$T result __sampledType(T) %sampled;
            };
       }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0)]
    T SampleLevel(SamplerState s, vector<float, Shape.dimensions+isArray> location, float level)
    {
        __target_switch
        {
            case cpp:
                __intrinsic_asm ".SampleLevel";
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                __intrinsic_asm ".SampleLevel";
            case metal:
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "$c$0.sample($1, ($2).x, uint(($2).y))$z";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "$c$0.sample($1, ($2).xy, uint(($2).z), level($3))$z";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "$c$0.sample($1, ($2).xyz, uint(($2).w), level($3))$z";
                    }
                }
                else
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "$c$0.sample($1, $2)$z";
                    case $(SLANG_TEXTURE_2D):
                    case $(SLANG_TEXTURE_3D):
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "$c$0.sample($1, $2, level($3))$z";
                    }
                }
                // TODO: This needs to be handled by the capability system
                __intrinsic_asm "<invalid intrinsic>";
            case glsl:
                __intrinsic_asm "$ctextureLod($p, $2, $3)$z";
            case cuda:
                if (isArray != 0)
                {
                    switch(Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "tex1DLayeredLod<$T0>($0, ($2).x, int(($2).y), ($3))";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "tex2DLayeredLod<$T0>($0, ($2).x, ($2).y, int(($2).z), ($3))";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "texCubemapLayeredLod<$T0>($0, ($2).x, ($2).y, ($2).z, int(($2).w), ($3))";
                    default:
                        __intrinsic_asm "<invalid intrinsic>";
                    }
                }
                else
                {
                    switch(Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "tex1DLod<$T0>($0, ($2), ($3))";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "tex2DLod<$T0>($0, ($2).x, ($2).y, ($3))";
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "tex3DLod<$T0>($0, ($2).x, ($2).y, ($2).z, ($3))";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "texCubemapLod<$T0>($0, ($2).x, ($2).y, ($2).z, ($3))";
                    default:
                        __intrinsic_asm "<invalid intrinsic>";
                    }
                }
            case spirv:
                return spirv_asm
                {
                    %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                    %sampled : __sampledType(T) = OpImageSampleExplicitLod %sampledImage $location None|Lod $level;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    , "WGSL supports only f32 type textures");
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "textureSampleLevel($0, $1, ($2).x, i32(($2).y), $3)$z";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "textureSampleLevel($0, $1, ($2).xy, i32(($2).z), $3)$z";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "textureSampleLevel($0, $1, ($2).xyz, i32(($2).w), $3)$z";
                    }
                }
                __intrinsic_asm "textureSampleLevel($0, $1, $2, $3)$z";
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_0)]
    T SampleLevel(SamplerState s, vector<float, Shape.dimensions+isArray> location, float level, constexpr vector<int, Shape.planeDimensions> offset)
    {
        __target_switch
        {
            case cpp:
                __intrinsic_asm ".SampleLevel";
            case hlsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                    , "HLSL supports only float and half type textures");
                __intrinsic_asm ".SampleLevel";
            case metal:
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "$c$0.sample($1, ($2).xy, uint(($2).z), level($3), $4)$z";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "$c$0.sample($1, ($2).xyz, uint(($2).w), level($3), $4)$z";
                    }
                }
                else
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                    case $(SLANG_TEXTURE_3D):
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "$c$0.sample($1, $2, level($3), $4)$z";
                    }
                }
                __intrinsic_asm "<invalid intrinsic>";
            case glsl:
                __intrinsic_asm "$ctextureLodOffset($p, $2, $3, $4)$z";
            case spirv:
                return spirv_asm
                {
                    %sampledImage : __sampledImageType(this) = OpSampledImage $this $s;
                    %sampled : __sampledType(T) = OpImageSampleExplicitLod %sampledImage $location None|Lod|ConstOffset $level $offset;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                    , "WGSL supports only f32 type textures");
                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "textureSampleLevel($0, $1, ($2).x, i32(($2).y), $3, $4)$z";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "textureSampleLevel($0, $1, ($2).xy, i32(($2).z), $3, $4)$z";
                    case $(SLANG_TEXTURE_CUBE):
                        __intrinsic_asm "textureSampleLevel($0, $1, ($2).xyz, i32(($2).w), $3, $4)$z";
                    }
                }
                __intrinsic_asm "textureSampleLevel($0, $1, $2, $3, $4)$z";
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl_spirv, sm_5_0)]
    T SampleLevel(SamplerState s, vector<float, Shape.dimensions+isArray> location, float level, constexpr vector<int, Shape.planeDimensions> offset, out uint status)
    {
        __target_switch
        {
        case hlsl:
            static_assert(T is float || T is vector<float,2> || T is vector<float,3> || T is vector<float,4>
                || T is half || T is vector<half,2> || T is vector<half,3> || T is vector<half,4>
                , "HLSL supports only float and half type textures");
            __intrinsic_asm ".SampleLevel";
        case spirv:
            return spirv_asm
            {
                OpCapability SparseResidency;
                %sparseResultType = OpTypeStruct $$uint __sampledType(T);
                %sampledImage:__sampledImageType(this) = OpSampledImage $this $s;

                %sparseResult:%sparseResultType = OpImageSparseSampleExplicitLod %sampledImage $location Lod|ConstOffset $level $offset;
                %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                %sampled:__sampledType(T) = OpCompositeExtract %sparseResult 1;

                OpStore &status %residentCode;
                __truncate $$T result __sampledType(T) %sampled;
            };
        }
    }
}

// Texture.GetDimensions and Sampler.GetDimensions
${{{{
const char* kTextureShapeTypeNames[] = {
    "__Shape1D", "__Shape2D", "__Shape3D", "__ShapeCube"};
for (int shapeIndex = 0; shapeIndex < 4; shapeIndex++)
for (int isArray = 0; isArray <= 1; isArray++)
for (int isMS = 0; isMS <= 1; isMS++) {
    if (isMS)
    {
        if (shapeIndex != kCoreModule_ShapeIndex2D)
            continue;
    }
    if (isArray)
    {
        if (shapeIndex == kCoreModule_ShapeIndex3D)
            continue;
    }
    auto shapeTypeName = kTextureShapeTypeNames[shapeIndex];
    TextureTypeInfo textureTypeInfo(kBaseTextureShapes[shapeIndex], isArray, isMS, 0, sb, path);
}}}}

__generic<T:ITexelElement, let sampleCount:int, let access:int, let isShadow:int, let isCombined:int, let format:int>
extension _Texture<T,$(shapeTypeName),$(isArray),$(isMS),sampleCount,access,isShadow,isCombined,format>
{
    ${{{{
    textureTypeInfo.writeGetDimensionFunctions();
    }}}}
}

${{{{
}
}}}}

// Texture.GetSamplePosition(int s);
__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let access:int, let isShadow:int, let isCombined:int, let format:int>
extension _Texture<T,Shape,isArray,1,sampleCount,access,isShadow,isCombined,format>
{
    [require(cpp_cuda_glsl_hlsl_spirv, texture_sm_4_1_vertex_fragment_geometry)]
    float2 GetSamplePosition(int s);
}

__intrinsic_op($(kIROp_MakeArray))
Array<T,4> __makeArray<T>(T v0, T v1, T v2, T v3);


// Beginning of Texture Gather
__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let access:int, let isShadow:int, let format:int>
[ForceInline]
[require(glsl_metal_spirv_wgsl, texture_gather)]
vector<T.Element,4> __texture_gather(
    _Texture<T, Shape, isArray, 0, sampleCount, access, isShadow, 0, format> texture,
    SamplerState s,
    vector<float, Shape.dimensions+isArray> location,
    int component)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "textureGather($p, $2, $3)";
    case metal:
        if (isArray == 1)
        {
            switch (Shape.flavor)
            {
            case $(SLANG_TEXTURE_2D):
                // Tv gather(sampler s, float2 coord, uint array, int2 offset = int2(0), component c = component::x) const
                __intrinsic_asm "$0.gather($1, ($2).xy, uint(($2).z), int2(0), metal::component($3))";
            case $(SLANG_TEXTURE_CUBE):
                // Tv gather(sampler s, float3 coord, uint array, component c = component::x) const
                __intrinsic_asm "$0.gather($1, ($2).xyz, uint(($2).w), metal::component($3))";
            }
        }
        if (Shape.flavor == $(SLANG_TEXTURE_CUBE))
        {
            // Tv gather(sampler s, float3 coord, component c = component::x) const
            __intrinsic_asm "$0.gather($1, $2, metal::component($3))";
        }
        // Tv gather(sampler s, float2 coord, int2 offset = int2(0), component c = component::x) const
        __intrinsic_asm "$0.gather($1, $2, int2(0), metal::component($3))";
    case spirv:
        return spirv_asm {
            %sampledImage : __sampledImageType(texture) = OpSampledImage $texture $s;
            result:$$vector<T.Element,4> = OpImageGather %sampledImage $location $component;
        };
    case wgsl:
        if (isShadow == 1)
        {
            // If depth texture, `textureGather` doesn't take channel value, `$3`.
            if (isArray == 1)
            {
                switch (Shape.flavor)
                {
                case $(SLANG_TEXTURE_2D):
                    __intrinsic_asm "textureGather($0, $1, ($2).xy, u32(($2).z))";
                case $(SLANG_TEXTURE_CUBE):
                    __intrinsic_asm "textureGather($0, $1, ($2).xyz, u32(($2).w))";
                }
            }
            __intrinsic_asm "textureGather($0, $1, $2)";
        }

        if (isArray == 1)
        {
            switch (Shape.flavor)
            {
            case $(SLANG_TEXTURE_2D):
                __intrinsic_asm "textureGather($3, $0, $1, ($2).xy, u32(($2).z))";
            case $(SLANG_TEXTURE_CUBE):
                __intrinsic_asm "textureGather($3, $0, $1, ($2).xyz, u32(($2).w))";
            }
        }
        __intrinsic_asm "textureGather($3, $0, $1, $2)";
    }
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let access:int, let isShadow:int, let format:int>
[ForceInline]
[require(glsl_spirv, texture_gather)]
vector<T.Element,4> __texture_gather(
    _Texture<T, Shape, isArray, 0, sampleCount, access, isShadow, 1, format> sampler,
    vector<float, Shape.dimensions+isArray> location,
    int component)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "textureGather($0, $1, $2)";
    case spirv:
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageGather $sampler $location $component;
        };
    }
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let access:int, let isShadow:int, let format:int>
[ForceInline]
[require(glsl_metal_spirv_wgsl, texture_gather)]
vector<T.Element,4> __texture_gather_offset(
    _Texture<T, Shape, isArray, 0, sampleCount, access, isShadow, 0, format> texture,
    SamplerState s,
    vector<float, Shape.dimensions+isArray> location,
    constexpr vector<int, Shape.planeDimensions> offset,
    int component)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "textureGatherOffset($p, $2, $3, $4)";
    case metal:
        static_assert(Shape.flavor == $(SLANG_TEXTURE_2D),
            "Metal supports offset variant of Gather only for 2D textures");

        if (isArray == 1)
        {
            // Tv gather(sampler s, float2 coord, uint array, int2 offset = int2(0), component c = component::x) const
            __intrinsic_asm "$0.gather($1, ($2).xy, uint(($2).z), $3, metal::component($4))";
        }
        // Tv gather(sampler s, float2 coord, int2 offset = int2(0), component c = component::x) const
        __intrinsic_asm "$0.gather($1, $2, $3, metal::component($4))";
    case spirv:
        return spirv_asm {
            OpCapability ImageGatherExtended;
            %sampledImage : __sampledImageType(texture) = OpSampledImage $texture $s;
            result:$$vector<T.Element,4> = OpImageGather %sampledImage $location $component Offset $offset;
        };
    case wgsl:
        if (isShadow == 1)
        {
            // If depth texture, `textureGather` doesn't take channel value, `$4`.
            if (isArray == 1)
            {
                switch (Shape.flavor)
                {
                case $(SLANG_TEXTURE_2D):
                    __intrinsic_asm "textureGather($0, $1, ($2).xy, u32(($2).z), $3)";
                case $(SLANG_TEXTURE_CUBE):
                    __intrinsic_asm "textureGather($0, $1, ($2).xyz, u32(($2).w), $3)";
                }
            }
            __intrinsic_asm "textureGather($0, $1, $2, $3)";
        }

        if (isArray == 1)
        {
            switch (Shape.flavor)
            {
            case $(SLANG_TEXTURE_2D):
                __intrinsic_asm "textureGather($4, $0, $1, ($2).xy, u32(($2).z), $3)";
            case $(SLANG_TEXTURE_CUBE):
                __intrinsic_asm "textureGather($4, $0, $1, ($2).xyz, u32(($2).w), $3)";
            }
        }
        __intrinsic_asm "textureGather($4, $0, $1, $2, $3)";
    }
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let access:int, let isShadow:int, let format:int>
[ForceInline]
[require(glsl_spirv, texture_gather)]
vector<T.Element,4> __texture_gather_offset(
    _Texture<T, Shape, isArray, 0, sampleCount, access, isShadow, 1, format> sampler,
    vector<float, Shape.dimensions+isArray> location,
    constexpr vector<int, Shape.planeDimensions> offset,
    int component)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "textureGatherOffset($0, $1, $2, $3)";
    case spirv:
        return spirv_asm {
            OpCapability ImageGatherExtended;
            result:$$vector<T.Element,4> = OpImageGather $sampler $location $component Offset $offset;
        };
    }
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let access:int, let isShadow:int, let format:int>
[ForceInline]
[require(glsl_spirv, texture_gather)]
vector<T.Element,4> __texture_gather_offsets(
    _Texture<T, Shape, isArray, 0, sampleCount, access, isShadow, 0, format> texture,
    SamplerState s,
    vector<float, Shape.dimensions+isArray> location,
    constexpr vector<int, Shape.planeDimensions> offset1,
    constexpr vector<int, Shape.planeDimensions> offset2,
    constexpr vector<int, Shape.planeDimensions> offset3,
    constexpr vector<int, Shape.planeDimensions> offset4,
    int component)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "textureGatherOffsets($p, $2, $T3[]($3, $4, $5, $6)), $7";
    case spirv:
        let offsets = __makeArray(offset1,offset2,offset3,offset4);
        return spirv_asm {
            OpCapability ImageGatherExtended;
            %sampledImage : __sampledImageType(texture) = OpSampledImage $texture $s;
            result:$$vector<T.Element,4> = OpImageGather %sampledImage $location $component ConstOffsets $offsets;
        };
    }
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let access:int, let isShadow:int, let format:int>
[ForceInline]
[require(glsl_spirv, texture_gather)]
vector<T.Element,4> __texture_gather_offsets(
    _Texture<T, Shape, isArray, 0, sampleCount, access, isShadow, 1, format> sampler,
    vector<float, Shape.dimensions+isArray> location,
    constexpr vector<int, Shape.planeDimensions> offset1,
    constexpr vector<int, Shape.planeDimensions> offset2,
    constexpr vector<int, Shape.planeDimensions> offset3,
    constexpr vector<int, Shape.planeDimensions> offset4,
    int component)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "textureGatherOffsets($0, $1, $T2[]($2, $3, $4, $5), $6)";
    case spirv:
        let offsets = __makeArray(offset1,offset2,offset3,offset4);
        return spirv_asm {
            OpCapability ImageGatherExtended;
            result:$$vector<T.Element,4> = OpImageGather $sampler $location $component ConstOffsets $offsets;
        };
    }
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let access:int, let isShadow:int, let format:int>
[ForceInline]
[require(glsl_metal_spirv_wgsl, texture_gather)]
vector<T.Element,4> __texture_gatherCmp(
    _Texture<T, Shape, isArray, 0, sampleCount, access, isShadow, 0, format> texture,
    SamplerComparisonState s,
    vector<float, Shape.dimensions+isArray> location,
    T.Element compareValue)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "textureGather($p, $2, $3)";
    case metal:
        if (isArray == 1)
        {
            switch (Shape.flavor)
            {
            case $(SLANG_TEXTURE_2D):
                // Tv gather_compare(sampler s, float2 coord, uint array, float compare_value, int2 offset = int2(0)) const
                __intrinsic_asm "$0.gather_compare($1, ($2).xy, uint(($2).z), $3)";
            case $(SLANG_TEXTURE_CUBE):
                // Tv gather_compare(sampler s, float3 coord, uint array, float compare_value) const
                __intrinsic_asm "$0.gather_compare($1, ($2).xyz, uint(($2).w), $3)";
            }
        }
        // Tv gather_compare(sampler s, float2 coord, float compare_value, int2 offset = int2(0)) const
        __intrinsic_asm "$0.gather_compare($1, $2, $3)";
    case spirv:
        return spirv_asm {
            %sampledImage : __sampledImageType(texture) = OpSampledImage $texture $s;
            result:$$vector<T.Element,4> = OpImageDrefGather %sampledImage $location $compareValue;
        };
    case wgsl:
        static_assert(isShadow == 1, "WGSL supports textureGatherCompare only for depth textures.");

        if (isArray == 1)
        {
            switch (Shape.flavor)
            {
            case $(SLANG_TEXTURE_2D):
                __intrinsic_asm "textureGatherCompare($0, $1, ($2).xy, u32(($2).z), $3)";
            case $(SLANG_TEXTURE_CUBE):
                __intrinsic_asm "textureGatherCompare($0, $1, ($2).xyz, u32(($2).w), $3)";
            }
        }
        __intrinsic_asm "textureGatherCompare($0, $1, $2, $3)";
    }
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let access:int, let isShadow:int, let format:int>
[ForceInline]
[require(glsl_spirv, texture_gather)]
vector<T.Element,4> __texture_gatherCmp(
    _Texture<T, Shape, isArray, 0, sampleCount, access, isShadow, 1, format> sampler,
    vector<float, Shape.dimensions+isArray> location,
    T.Element compareValue)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "textureGather($0, $1, $2)";
    case spirv:
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageDrefGather $sampler $location $compareValue;
        };
    }
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let access:int, let isShadow:int, let format:int>
[ForceInline]
[require(glsl_metal_spirv_wgsl, texture_gather)]
vector<T.Element,4> __texture_gatherCmp_offset(
    _Texture<T, Shape, isArray, 0, sampleCount, access, isShadow, 0, format> texture,
    SamplerComparisonState s,
    vector<float, Shape.dimensions+isArray> location,
    T.Element compareValue,
    constexpr vector<int, Shape.planeDimensions> offset)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "textureGatherOffset($p, $2, $3, $4)";
    case metal:
        static_assert(Shape.flavor == $(SLANG_TEXTURE_2D),
            "Metal supports depth compare Gather only for 2D texture");

        if (isArray == 1)
        {
            // Tv gather_compare(sampler s, float2 coord, uint array, float compare_value, int2 offset = int2(0)) const
            __intrinsic_asm "$0.gather_compare($1, ($2).xy, uint(($2).z), $3, $4)";
        }
        // Tv gather_compare(sampler s, float2 coord, float compare_value, int2 offset = int2(0)) const
        __intrinsic_asm "$0.gather_compare($1, $2, $3, $4)";
    case spirv:
        return spirv_asm {
            %sampledImage : __sampledImageType(texture) = OpSampledImage $texture $s;
            result:$$vector<T.Element,4> = OpImageDrefGather %sampledImage $location $compareValue ConstOffset $offset;
        };
    case wgsl:
        static_assert(isShadow == 1, "WGSL supports textureGatherCompare only for depth textures.");

        if (isArray == 1)
        {
            switch (Shape.flavor)
            {
            case $(SLANG_TEXTURE_2D):
                __intrinsic_asm "textureGatherCompare($0, $1, ($2).xy, u32(($2).z), $3, $4)";
            case $(SLANG_TEXTURE_CUBE):
                __intrinsic_asm "textureGatherCompare($0, $1, ($2).xyz, u32(($2).w), $3, $4)";
            }
        }
        __intrinsic_asm "textureGatherCompare($0, $1, $2, $3, $4)";
    }
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let access:int, let isShadow:int, let format:int>
[ForceInline]
[require(glsl_spirv, texture_gather)]
vector<T.Element,4> __texture_gatherCmp_offset(
    _Texture<T, Shape, isArray, 0, sampleCount, access, isShadow, 1, format> sampler,
    vector<float, Shape.dimensions+isArray> location,
    T.Element compareValue,
    constexpr vector<int, Shape.planeDimensions> offset)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "textureGatherOffset($0, $1, $2, $3)";
    case spirv:
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageDrefGather $sampler $location $compareValue ConstOffset $offset;
        };
    }
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let access:int, let isShadow:int, let format:int>
[ForceInline]
[require(glsl_spirv, texture_gather)]
vector<T.Element,4> __texture_gatherCmp_offsets(
    _Texture<T, Shape, isArray, 0, sampleCount, access, isShadow, 0, format> texture,
    SamplerComparisonState s,
    vector<float, Shape.dimensions+isArray> location,
    T.Element compareValue,
    vector<int, Shape.planeDimensions> offset1,
    vector<int, Shape.planeDimensions> offset2,
    vector<int, Shape.planeDimensions> offset3,
    vector<int, Shape.planeDimensions> offset4)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "textureGatherOffsets($p, $2, $3, $T4[]($4, $5, $6, $7))";
    case spirv:
        let offsets = __makeArray(offset1,offset2,offset3,offset4);
        return spirv_asm {
            OpCapability ImageGatherExtended;
            %sampledImage : __sampledImageType(texture) = OpSampledImage $texture $s;
            result:$$vector<T.Element,4> = OpImageDrefGather %sampledImage $location $compareValue ConstOffsets $offsets;
        };
    }
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let access:int, let isShadow:int, let format:int>
[ForceInline]
[require(glsl_spirv, texture_gather)]
vector<T.Element,4> __texture_gatherCmp_offsets(
    _Texture<T, Shape, isArray, 0, sampleCount, access, isShadow, 1, format> sampler,
    vector<float, Shape.dimensions+isArray> location,
    T.Element compareValue,
    vector<int, Shape.planeDimensions> offset1,
    vector<int, Shape.planeDimensions> offset2,
    vector<int, Shape.planeDimensions> offset3,
    vector<int, Shape.planeDimensions> offset4)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "textureGatherOffsets($0, $1, $2, $T3[]($3, $4, $5, $6))";
    case spirv:
        let offsets = __makeArray(offset1,offset2,offset3,offset4);
        return spirv_asm {
            OpCapability ImageGatherExtended;
            result:$$vector<T.Element,4> = OpImageDrefGather $sampler $location $compareValue ConstOffsets $offsets;
        };
    }
}

${{{{
for (int isCombined = 0; isCombined < 2; isCombined++)
{
}}}}
// Gather for [isCombined = $(isCombined)]

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let isShadow:int, let format:int>
extension _Texture<T,Shape,isArray,0,sampleCount,0,isShadow,$(isCombined),format>
{
${{{{
    for (int isShadow = 0; isShadow < 2; isShadow++)
    for (auto componentId = 0;  componentId < 5; componentId++)
    {
        const char* compareFunc = isShadow ? "Cmp" : "";
        const char* compareParam = isShadow ? ", T.Element compareValue" : "";
        const char* compareArg = isShadow ? ", compareValue" : "";

        // Some targets support the combined texture natively
        const char* samplerParam = isCombined ? "" : (isShadow ? "SamplerComparisonState s," : "SamplerState s,");
        const char* samplerArg = isCombined ? "" : ", s";
        const char* getTexture = isCombined ? "__getTexture()" : "this";
        const char* getSampler = isCombined ? (isShadow ? ", __getComparisonSampler()" : ", __getSampler()") : samplerArg;

        const char* componentFuncString[] = { "", "Red", "Green", "Blue", "Alpha"};
        const char* componentArgString[] = { ", 0", ", 0", ", 1", ", 2", ", 3" };
        const char* componentFunc = componentFuncString[componentId];
        const char* componentArg = (isShadow ? "" : componentArgString[componentId]);
}}}}
    [ForceInline]
    [require(glsl_hlsl_metal_spirv_wgsl, texture_gather)]
    vector<T.Element,4> Gather$(compareFunc)$(componentFunc)(
        $(samplerParam)
        vector<float, Shape.dimensions+isArray> location
        $(compareParam))
    {
        static_assert(Shape.flavor == $(SLANG_TEXTURE_2D) || Shape.flavor == $(SLANG_TEXTURE_CUBE),
            "Gather is supported only for 2D and 3D textures");

        __target_switch
        {
        case hlsl: __intrinsic_asm ".Gather$(compareFunc)$(componentFunc)";
        case metal:
        case wgsl:
            return __texture_gather$(compareFunc)($(getTexture) $(getSampler), location $(compareArg) $(componentArg));
        case glsl:
        case spirv:
            return __texture_gather$(compareFunc)(this $(samplerArg), location $(compareArg) $(componentArg));
        }
    }

    [ForceInline]
    [require(hlsl, texture_gather)]
    vector<T.Element,4> Gather$(compareFunc)$(componentFunc)(
        $(samplerParam)
        vector<float, Shape.dimensions+isArray> location
        $(compareParam),
        out uint status)
    {
        static_assert(Shape.flavor == $(SLANG_TEXTURE_2D) || Shape.flavor == $(SLANG_TEXTURE_CUBE),
            "Gather is supported only for 2D and 3D textures");

        __target_switch
        {
        case hlsl: __intrinsic_asm ".Gather$(compareFunc)$(componentFunc)";
        }
    }

    [ForceInline]
    [require(glsl_hlsl_metal_spirv_wgsl, texture_gather)]
    vector<T.Element,4> Gather$(compareFunc)$(componentFunc)(
        $(samplerParam)
        vector<float, Shape.dimensions+isArray> location
        $(compareParam),
        constexpr vector<int, Shape.planeDimensions> offset)
    {
        static_assert(Shape.flavor == $(SLANG_TEXTURE_2D) || Shape.flavor == $(SLANG_TEXTURE_CUBE),
            "Gather is supported only for 2D and 3D textures");

        __target_switch
        {
        case hlsl: __intrinsic_asm ".Gather$(compareFunc)$(componentFunc)";
        case metal:
        case wgsl:
            return __texture_gather$(compareFunc)_offset($(getTexture) $(getSampler), location $(compareArg), offset $(componentArg));
        case glsl:
        case spirv:
            return __texture_gather$(compareFunc)_offset(this $(samplerArg), location $(compareArg), offset $(componentArg));
        }
    }

    [ForceInline]
    [require(hlsl, texture_gather)]
    vector<T.Element,4> Gather$(compareFunc)$(componentFunc)(
        $(samplerParam)
        vector<float, Shape.dimensions+isArray> location
        $(compareParam),
        constexpr vector<int, Shape.planeDimensions> offset,
        out uint status)
    {
        static_assert(Shape.flavor == $(SLANG_TEXTURE_2D) || Shape.flavor == $(SLANG_TEXTURE_CUBE),
            "Gather is supported only for 2D and 3D textures");

        __target_switch
        {
        case hlsl: __intrinsic_asm ".Gather$(compareFunc)$(componentFunc)";
        }
    }

    [ForceInline]
    [require(glsl_hlsl_spirv, texture_gather)]
    vector<T.Element,4> Gather$(compareFunc)$(componentFunc)(
        $(samplerParam)
        vector<float, Shape.dimensions+isArray> location
        $(compareParam),
        constexpr vector<int, Shape.planeDimensions> offset1,
        constexpr vector<int, Shape.planeDimensions> offset2,
        constexpr vector<int, Shape.planeDimensions> offset3,
        constexpr vector<int, Shape.planeDimensions> offset4)
    {
        static_assert(Shape.flavor == $(SLANG_TEXTURE_2D) || Shape.flavor == $(SLANG_TEXTURE_CUBE),
            "Gather is supported only for 2D and 3D textures");

        __target_switch
        {
        case hlsl: __intrinsic_asm ".Gather$(compareFunc)$(componentFunc)";
        case glsl:
        case spirv:
            return __texture_gather$(compareFunc)_offsets<T>(this $(samplerArg), location $(compareArg), offset1,offset2,offset3,offset4 $(componentArg));
        }
    }

    [ForceInline]
    [require(hlsl, texture_gather)]
    vector<T.Element,4> Gather$(compareFunc)$(componentFunc)(
        $(samplerParam)
        vector<float, Shape.dimensions+isArray> location
        $(compareParam),
        constexpr vector<int, Shape.planeDimensions> offset1,
        constexpr vector<int, Shape.planeDimensions> offset2,
        constexpr vector<int, Shape.planeDimensions> offset3,
        constexpr vector<int, Shape.planeDimensions> offset4,
        out uint status)
    {
        static_assert(Shape.flavor == $(SLANG_TEXTURE_2D) || Shape.flavor == $(SLANG_TEXTURE_CUBE),
            "Gather is supported only for 2D and 3D textures");

        __target_switch
        {
        case hlsl: __intrinsic_asm ".Gather$(compareFunc)$(componentFunc)";
        }
    }

${{{{
    } // for (componentId)
}}}}
} // End of: Gather for [isCombined = $(isCombined)]

${{{{
} // for (isScalarTexture)
}}}}
// End of all Texture Gather


// Load/Subscript for readonly, no MS textures

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let isShadow:int, let isCombined:int, let format:int>
extension _Texture<T,Shape,isArray,0,sampleCount,0,isShadow,isCombined,format>
{
//@hidden:
    static const int isMS = 0;
    static const int access = $(kCoreModule_ResourceAccessReadOnly);
//@public:
    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_1_samplerless)]
    T Load(vector<int, Shape.dimensions+isArray+1> location)
    {
        __target_switch
        {
        case cpp:
        case hlsl:
            __intrinsic_asm ".Load";
		case cuda:
            if (isArray != 0)
                {
     				static_assert(Shape.flavor == $(SLANG_TEXTURE_2D) || Shape.flavor == $(SLANG_TEXTURE_3D),
                              "Integer coordinates are supported for texture reads only for 2D and 3D textures and 2D array textures.");

                    if (Shape.flavor == $(SLANG_TEXTURE_2D))
                    {
						__intrinsic_asm "tex2DArrayfetch_int<$T0>($0, ($1).x, ($1).y, ($1).z)";
				    }
                    else
					{
                        __intrinsic_asm "<invalid intrinsic>";
                    }
                }
                else
                {
                    switch(Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "tex2Dfetch_int<$T0>($0, ($1).x, ($1).y)";
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "tex3Dfetch_int<$T0>($0, ($1).x, ($1).y, ($1).z)";
					case $(SLANG_TEXTURE_CUBE):
                    default:
                        __intrinsic_asm "<invalid intrinsic>";
                    }
                }
        case metal:
            switch (Shape.flavor)
            {
            case $(SLANG_TEXTURE_1D):
                // lod is not supported for 1D texture
                if (isArray == 1)
                    // Tv read(uint coord, uint array, uint lod = 0) const
                    __intrinsic_asm "$c$0.read(uint(($1).x), uint(($1).y))$z";
                else
                    // Tv read(uint coord, uint lod = 0) const
                    __intrinsic_asm "$c$0.read(uint(($1).x))$z";
                break;
            case $(SLANG_TEXTURE_2D):
                if (isShadow == 1)
                {
                    if (isArray == 1)
                        // T read(uint2 coord, uint array, uint lod = 0) const
                        __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy), uint(($1).z), uint(($1).w))$z";
                    else
                        // T read(uint2 coord, uint lod = 0) const
                        __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy), uint(($1).z))$z";
                }
                else
                {
                    if (isArray == 1)
                        // Tv read(uint2 coord, uint array, uint lod = 0) const
                        __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy), uint(($1).z), uint(($1).w))$z";
                    else
                        // Tv read(uint2 coord, uint lod = 0) const
                        __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy), uint(($1).z))$z";
                }
                break;
            case $(SLANG_TEXTURE_3D):
                if (isShadow == 0 && isArray == 0)
                    // Tv read(uint3 coord, uint lod = 0) const
                    __intrinsic_asm "$c$0.read(vec<uint,3>(($1).xyz), uint(($1).w))$z";
                break;
            case $(SLANG_TEXTURE_CUBE):
                static_assert(isArray == 0, "Unsupported 'Load' of 'texture cube array' for 'metal' target");
                if (isShadow == 1)
                {
                    if (isArray == 1)
                        // T read(uint2 coord, uint face, uint array, uint lod = 0) const
                        __intrinsic_asm "<invalid intrinsics>";
                    else
                        // T read(uint2 coord, uint face, uint lod = 0) const
                        __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy), uint(($1).z), uint(($1).w))$z";
                }
                else
                {
                    if (isArray == 1)
                        // Tv read(uint2 coord, uint face, uint array, uint lod = 0) const
                        __intrinsic_asm "<invalid intrinsics>";
                    else
                        // Tv read(uint2 coord, uint face, uint lod = 0) const
                        __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy), uint(($1).z), uint(($1).w))$z";
                }
                break;
            }
            static_assert(false, "Unsupported 'Load' of 'texture' for 'metal' target");
            __intrinsic_asm "<invalid intrinsics>";
        case glsl:
            if (isCombined == 0)
                __requireTargetExtension("GL_EXT_samplerless_texture_functions");
            __intrinsic_asm "$ctexelFetch($0, ($1).$w1b, ($1).$w1e)$z";
        case spirv:
            const int lodLoc = Shape.dimensions+isArray;
            let coord = __vectorReshape<Shape.dimensions+isArray>(location);
            let lod = location[lodLoc];
            if (isCombined != 0)
            {
                return spirv_asm
                {
                    %image:__imageType(this) = OpImage $this;
                    %sampled:__sampledType(T) = OpImageFetch %image $coord Lod $lod;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            }
            else
            {
                return spirv_asm
                {
                    %sampled:__sampledType(T) = OpImageFetch $this $coord Lod $lod;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            }
        case wgsl:
            static_assert(Shape.flavor == $(SLANG_TEXTURE_1D)
                || Shape.flavor == $(SLANG_TEXTURE_2D)
                || Shape.flavor == $(SLANG_TEXTURE_3D)
                , "WGSL doesn't supports textureLoad for Cube texture.");
            static_assert(isArray == 0 || Shape.flavor == $(SLANG_TEXTURE_2D)
                , "WGSL supports textureLoad for texture_2d_array but not for array of 1D, 3D or Cube.");

            if (isArray == 1)
            {
                switch (Shape.flavor)
                {
                case $(SLANG_TEXTURE_2D):
                    __intrinsic_asm "textureLoad($0, ($1).xy, i32(($1).z), ($1).w)$z";
                }
            }
            switch (Shape.flavor)
            {
            case $(SLANG_TEXTURE_1D):
                __intrinsic_asm "textureLoad($0, ($1).x, ($1).y)$z";
            case $(SLANG_TEXTURE_2D):
                __intrinsic_asm "textureLoad($0, ($1).xy, ($1).z)$z";
            case $(SLANG_TEXTURE_3D):
                __intrinsic_asm "textureLoad($0, ($1).xyz, ($1).w)$z";
            }
            return __default<T>();
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_spirv, texture_sm_4_1_samplerless)]
    T Load(vector<int, Shape.dimensions+isArray+1> location, constexpr vector<int, Shape.planeDimensions> offset)
    {
        __target_switch
        {
            case cpp:
            case hlsl:
                __intrinsic_asm ".Load";
            case glsl:
                if (isCombined == 0)
                    __requireTargetExtension("GL_EXT_samplerless_texture_functions");
                __intrinsic_asm "$ctexelFetchOffset($0, ($1).$w1b, ($1).$w1e, ($2))$z";
            case spirv:
                const int lodLoc = Shape.dimensions+isArray;
                let coord = __vectorReshape<Shape.dimensions+isArray>(location);
                let lod = location[lodLoc];
                if (isCombined != 0)
                {
                    return spirv_asm
                    {
                        %image:__imageType(this) = OpImage $this;
                        %sampled:__sampledType(T) = OpImageFetch %image $coord Lod|ConstOffset $lod $offset;
                        __truncate $$T result __sampledType(T) %sampled;
                    };
                }
                else
                {
                    return spirv_asm
                    {
                        %sampled:__sampledType(T) = OpImageFetch $this $coord Lod|ConstOffset $lod $offset;
                        __truncate $$T result __sampledType(T) %sampled;
                    };
                }
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl_spirv, texture_sm_4_1_samplerless)]
    T Load(vector<int, Shape.dimensions+isArray+1> location, constexpr vector<int, Shape.planeDimensions> offset, out uint status)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load";
        case spirv:
             const int lodLoc = Shape.dimensions+isArray;
             let coord = __vectorReshape<Shape.dimensions+isArray>(location);
             let lod = location[lodLoc];
             if (isCombined != 0)
             {
                 return spirv_asm
                 {
                     OpCapability SparseResidency;
                     %sparseResultType = OpTypeStruct $$uint __sampledType(T);
                     %image:__imageType(this) = OpImage $this;
 
                     %sparseResult:%sparseResultType = OpImageSparseFetch %image $coord Lod|ConstOffset $lod $offset;
                     %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                     %sampled:__sampledType(T) = OpCompositeExtract %sparseResult 1;
 
                     OpStore &status %residentCode;
                     __truncate $$T result __sampledType(T) %sampled;
                 };
             }
             else
             {
                 return spirv_asm
                 {
                     OpCapability SparseResidency;
                     %sparseResultType = OpTypeStruct $$uint __sampledType(T);
 
                     %sparseResult:%sparseResultType = OpImageSparseFetch $this $coord Lod|ConstOffset $lod $offset;
                     %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                     %sampled:__sampledType(T) = OpCompositeExtract  %sparseResult 1;
 
                     OpStore &status %residentCode;
                     __truncate $$T result __sampledType(T) %sampled;
                 };
             }
        default:
            status = 0;
            return Load(location, offset);
        }
    }

    __subscript(vector<uint, Shape.dimensions+isArray> location) -> T
    {
        [__readNone]
        [ForceInline]
        [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_1_samplerless)]
        get
        {
            __target_switch
            {
                case cpp:
                case hlsl:
                    __intrinsic_asm ".operator[]";
                case metal:
                case cuda:
                    return Load(__makeVector(location, 0));
                case glsl:
                    if (isCombined == 0)
                        __requireTargetExtension("GL_EXT_samplerless_texture_functions");

                    return Load(__makeVector(location, 0));
                case spirv:
                    if (isCombined != 0)
                    {
                        return spirv_asm
                        {
                            %image:__imageType(this) = OpImage $this;
                            %sampled:__sampledType(T) = OpImageFetch %image $location;
                            __truncate $$T result __sampledType(T) %sampled;
                        };
                    }
                    else
                    {
                        return spirv_asm
                        {
                            %sampled:__sampledType(T) = OpImageFetch $this $location;
                            __truncate $$T result __sampledType(T) %sampled;
                        };
                    }
                case wgsl:
                    return Load(__makeVector(location, 0));
            }
        }
    }
}

// Texture Load/Subscript for readonly, MS textures

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let isShadow:int, let isCombined:int, let format:int>
extension _Texture<T,Shape,isArray,1,sampleCount,0,isShadow,isCombined,format>
{
//@hidden:
    static const int access = $(kCoreModule_ResourceAccessReadOnly);
    static const int isMS = 1;
//@public:
    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_1_samplerless)]
    T Load(vector<int, Shape.dimensions+isArray> location, int sampleIndex)
    {
        __target_switch
        {
            case cpp:
            case hlsl:
                __intrinsic_asm ".Load";
            case metal:
                switch (Shape.flavor)
                {
                case $(SLANG_TEXTURE_2D):
                    if (isShadow == 1)
                    {
                        if (isArray == 1)
                            // Document seems to have a typo. `lod` must be `sample`.
                            // Tv read(uint2 coord, uint array, uint lod = 0) const
                            __intrinsic_asm "$c$0.read(($1).xy, ($1).z, uint($2))$z";
                        else
                            // T read(uint2 coord, uint sample) const
                            __intrinsic_asm "$c$0.read($1, uint($2))$z";
                    }
                    else
                    {
                        if (isArray == 1)
                            // Document seems to have a typo. `lod` must be `sample`.
                            // Tv read(uint2 coord, uint array, uint lod = 0) const
                            __intrinsic_asm "$c$0.read(($1).xy, ($1).z, uint($2))$z";
                        else
                            // Tv read(uint2 coord, uint sample) const
                            __intrinsic_asm "$c$0.read($1, uint($2))$z";
                    }
                    break;
                }
                // TODO: This needs to be handled by the capability system
                __intrinsic_asm "<Not supported>";
            case glsl:
                if (isCombined == 0)
                    __requireTargetExtension("GL_EXT_samplerless_texture_functions");
                __intrinsic_asm "$ctexelFetch($0, $1, ($2))$z";
            case spirv:
                if (isCombined != 0)
                {
                    return spirv_asm
                    {
                        %image:__imageType(this) = OpImage $this;
                        %sampled:__sampledType(T) = OpImageFetch %image $location Sample $sampleIndex;
                        __truncate $$T result __sampledType(T) %sampled;
                    };
                }
                else
                {
                    return spirv_asm
                    {
                        %sampled:__sampledType(T) = OpImageFetch $this $location Sample $sampleIndex;
                        __truncate $$T result __sampledType(T) %sampled;
                    };
                }
            case wgsl:
                static_assert(Shape.flavor == $(SLANG_TEXTURE_2D)
                    , "WGSL supports textureLoad for texture_multisampled_2d but not for multisampled of 1D, 3D or Cube.");
                static_assert(isArray == 0
                    , "WGSL doesn't support array variants of multisampled textures for textureLoad.");

                __intrinsic_asm "textureLoad($0, $1, $2)$z";
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_spirv_wgsl, texture_sm_4_1_samplerless)]
    T Load(vector<int, Shape.dimensions + isArray + 1> locationAndSampleIndex)
    {
        return Load(__vectorReshape<Shape.dimensions + isArray>(locationAndSampleIndex), locationAndSampleIndex[Shape.dimensions + isArray]);
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_spirv, texture_sm_4_1_samplerless)]
    T Load(vector<int, Shape.dimensions+isArray> location, int sampleIndex, constexpr vector<int, Shape.planeDimensions> offset)
    {
        __target_switch
        {
            case cpp:
            case hlsl:
                __intrinsic_asm ".Load";
            case glsl:
                if (isCombined == 0)
                    __requireTargetExtension("GL_EXT_samplerless_texture_functions");
                __intrinsic_asm "$ctexelFetchOffset($0, $1, ($2), ($3))$z";
            case spirv:
                if (isCombined != 0)
                {
                    return spirv_asm
                    {
                        %image:__imageType(this) = OpImage $this;
                        %sampled:__sampledType(T) = OpImageFetch %image $location ConstOffset|Sample $offset $sampleIndex;
                        __truncate $$T result __sampledType(T) %sampled;
                    };
                }
                else
                {
                    return spirv_asm
                    {
                        %sampled:__sampledType(T) = OpImageFetch $this $location ConstOffset|Sample $offset  $sampleIndex;
                        __truncate $$T result __sampledType(T) %sampled;
                    };
                }
        }
    }

    [__readNone]
    [ForceInline]
    [require(hlsl_spirv, texture_sm_4_1_samplerless)]
    T Load(vector<int, Shape.dimensions+isArray> location, int sampleIndex, constexpr vector<int, Shape.planeDimensions> offset, out uint status)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load";
        case spirv:
             if (isCombined != 0)
             {
                 return spirv_asm
                 {
                     OpCapability SparseResidency;
                     %sparseResultType = OpTypeStruct $$uint __sampledType(T);
                     %image:__imageType(this) = OpImage $this;
 
                     %sparseResult:%sparseResultType = OpImageSparseFetch %image $location ConstOffset|Sample $offset $sampleIndex;
                     %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                     %sampled:__sampledType(T) = OpCompositeExtract %sparseResult 1;
 
                     OpStore &status %residentCode;
                     __truncate $$T result __sampledType(T) %sampled;
                 };
             }
             else
             {
                 return spirv_asm
                 {
                     OpCapability SparseResidency;
                     %sparseResultType = OpTypeStruct $$uint __sampledType(T);
 
                     %sparseResult:%sparseResultType = OpImageSparseFetch $this $location ConstOffset|Sample $offset $sampleIndex;
                     %residentCode:$$uint = OpCompositeExtract %sparseResult 0;
                     %sampled:__sampledType(T) = OpCompositeExtract %sparseResult 1;
 
                     OpStore &status %residentCode;
                     __truncate $$T result __sampledType(T) %sampled;
                 };
             }
        default:
            status = 0;
            return Load(location, sampleIndex, offset);
        }
    }

    __subscript(vector<uint, Shape.dimensions+isArray> location) -> T
    {
        [__readNone]
        [ForceInline]
        [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_1_samplerless)]
        get
        {
            __target_switch
            {
                case cpp:
                case hlsl:
                    __intrinsic_asm "($0).sample[$1]";
                case metal:
                case spirv:
                case wgsl:
                    return Load(location, 0);
                case glsl:
                    if (isCombined == 0)
                        __requireTargetExtension("GL_EXT_samplerless_texture_functions");
                    return Load(location, 0);
            }
        }
    }
    __subscript(vector<uint, Shape.dimensions+isArray> location, int sampleIndex) -> T
    {
        [__readNone]
        [ForceInline]
        [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_1_samplerless)]
        get
        {
            __target_switch
            {
                case cpp:
                case hlsl:
                    __intrinsic_asm "($0).sample[$2][$1]";
                case metal:
                case spirv:
                case wgsl:
                    return Load(location, sampleIndex);
                case glsl:
                    if (isCombined == 0)
                        __requireTargetExtension("GL_EXT_samplerless_texture_functions");
                    return Load(location, sampleIndex);
            }
        }
    }
}

// Load/Subscript for readwrite textures
${{{{
    for (int access = kCoreModule_ResourceAccessReadWrite; access <= kCoreModule_ResourceAccessRasterizerOrdered; access++) {
        const char* glslIntrinsic = "$cimageLoad($0, $1)$z";
        const char* glslIntrinsicOffset = "$cimageLoad($0, ($1)+($2))$z";
        const char* glslIntrinsicMS = "$cimageLoad($0, $1, $2)$z";
        const char* glslIntrinsicMSOffset = "$cimageLoad($0, ($1)+($3), $2)$z";
}}}}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let isShadow:int, let format:int>
extension _Texture<T,Shape,isArray,0,sampleCount,$(access),isShadow, 0,format>
{
    ${{{{
        if (access != kCoreModule_ResourceAccessWriteOnly)
        {
    }}}}
    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_1)]
    T Load(vector<int, Shape.dimensions+isArray> location)
    {
        __target_switch
        {
            case cpp:
            case hlsl:
                __intrinsic_asm ".Load";
            case glsl:
                __intrinsic_asm "$(glslIntrinsic)";
            case cuda:
                if (isArray != 0)
                {
                    switch(Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "surf1DLayeredread$C<$T0>($0, ($1).x * $E, ($1).y, SLANG_CUDA_BOUNDARY_MODE)";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "surf2DLayeredread$C<$T0>($0, ($1).x * $E, ($1).y, ($1).z, SLANG_CUDA_BOUNDARY_MODE)";
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "surf3DLayeredread$C<$T0>($0, ($1).x * $E, ($1).y, ($1).z, ($1).w, SLANG_CUDA_BOUNDARY_MODE)";
                    default:
                        __intrinsic_asm "<invalid intrinsic>";
                    }
                }
                else
                {
                    switch(Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "surf1Dread$C<$T0>($0, ($1) * $E, SLANG_CUDA_BOUNDARY_MODE)";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "surf2Dread$C<$T0>($0, ($1).x * $E, ($1).y, SLANG_CUDA_BOUNDARY_MODE)";
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "surf3Dread$C<$T0>($0, ($1).x * $E, ($1).y, ($1).z, SLANG_CUDA_BOUNDARY_MODE)";
                    default:
                        __intrinsic_asm "<invalid intrinsic>";
                    }
                }
            case spirv:
                return spirv_asm
                {
                    %sampled:__sampledType(T) = OpImageRead $this $location;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case metal:
                switch (Shape.flavor)
                {
                case $(SLANG_TEXTURE_1D):
                    // lod is not supported for 1D texture
                    if (isArray == 1)
                        // Tv read(uint coord, uint array, uint lod = 0) const
                        __intrinsic_asm "$c$0.read(uint(($1).x), uint(($1).y))$z";
                    else
                        // Tv read(uint coord, uint lod = 0) const
                        __intrinsic_asm "$c$0.read(uint($1))$z";
                    break;
                case $(SLANG_TEXTURE_2D):
                    if (isShadow == 1)
                    {
                        if (isArray == 1)
                            // T read(uint2 coord, uint array, uint lod = 0) const
                            __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy), uint(($1).z))$z";
                        else
                            // T read(uint2 coord, uint lod = 0) const
                            __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy))$z";
                    }
                    else
                    {
                        if (isArray == 1)
                            // Tv read(uint2 coord, uint array, uint lod = 0) const
                            __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy), uint(($1).z))$z";
                        else
                            // Tv read(uint2 coord, uint lod = 0) const
                            __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy))$z";
                    }
                    break;
                case $(SLANG_TEXTURE_3D):
                    if (isShadow == 0 && isArray == 0)
                        // Tv read(uint3 coord, uint lod = 0) const
                        __intrinsic_asm "$c$0.read(vec<uint,3>(($1).xyz))$z";
                    break;
                case $(SLANG_TEXTURE_CUBE):
                    static_assert(isArray == 0, "Unsupported 'Load' of 'texture cube array' for 'metal' target");
                    if (isShadow == 1)
                    {
                        if (isArray == 1)
                            // T read(uint2 coord, uint face, uint array, uint lod = 0) const
                            __intrinsic_asm "$0.read(vec<uint,2>(($1).xy), uint(($1).z)%6, uint(($1).z)/6, uint(($1).w))";
                        else
                            // T read(uint2 coord, uint face, uint lod = 0) const
                            __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy), uint(($1).z), uint(($1).w))$z";
                    }
                    else
                    {
                        if (isArray == 1)
                            // Tv read(uint2 coord, uint face, uint array, uint lod = 0) const
                            __intrinsic_asm "$0.read(vec<uint,2>(($1).xy), uint(($1).z)%6, uint(($1).z)/6, uint(($1).w))";
                        else
                            // Tv read(uint2 coord, uint face, uint lod = 0) const
                            __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy), uint(($1).z), uint(($1).w))$z";
                    }
                    break;
                }
                static_assert(false, "Unsupported 'Load' of 'texture' for 'metal' target");
                __intrinsic_asm "<invalid intrinsics>";
            case wgsl:
                static_assert(Shape.flavor == $(SLANG_TEXTURE_1D)
                    || Shape.flavor == $(SLANG_TEXTURE_2D)
                    || Shape.flavor == $(SLANG_TEXTURE_3D)
                    , "WGSL doesn't supports textureLoad for Cube texture.");
                static_assert(isArray == 0 || Shape.flavor == $(SLANG_TEXTURE_2D)
                    , "WGSL supports textureLoad for 2d_array but not for array of 1D, 3D or Cube.");
                static_assert(isShadow == 0 || T is float
                    , "WGSL supports only f32 depth textures");

                if (isArray == 1)
                {
                    switch (Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "textureLoad($0, ($1).xy, i32(($1).z))$z";
                    }
                }
                __intrinsic_asm "textureLoad($0, $1)$z";
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
    T Load(vector<int, Shape.dimensions+isArray> location, vector<int, Shape.dimensions+isArray> offset)
    {
        __target_switch
        {
            case cpp:
            case hlsl:
                __intrinsic_asm ".Load";
            case glsl:
                __intrinsic_asm "$(glslIntrinsicOffset)";
            case spirv:
                return spirv_asm
                {
                    %sampled:__sampledType(T) = OpImageRead $this $location ConstOffset $offset;
                    __truncate $$T result __sampledType(T) %sampled;
                };
        }
    }

    [__readNone]
    [ForceInline]
    T Load(vector<int, Shape.dimensions+isArray> location, vector<int, Shape.dimensions+isArray> offset, out uint status)
    {
        __target_switch
        {
        case hlsl:
        case cpp:
            __intrinsic_asm ".Load";
        default:
            status = 0;
            return Load(location, offset);
        }
    }

    ${{{{
        } // if (access != kCoreModule_ResourceAccessWriteOnly)
    }}}}
    [require(glsl, texture_sm_4_1)]
    void __glslImageStore(vector<int, Shape.dimensions+isArray> location, T value)
    {
        __intrinsic_asm "imageStore($0, $1, $V2)";
    }
    
    [require(metal, texture_sm_4_1)]
    __intrinsic_op($(kIROp_ImageStore))
    static void __metalImageStoreArray(This val, vector<uint, Shape.dimensions> location, T value, uint arrayIndex);

    [require(metal, texture_sm_4_1)]
    __intrinsic_op($(kIROp_ImageStore))
    static void __metalImageStore(This val, vector<uint, Shape.dimensions+isArray> location, T value);

    [ForceInline]
    void Store(vector<uint, Shape.dimensions+isArray> location, T newValue)
    {
        __target_switch
        {
            case cpp:
            case hlsl:
                __intrinsic_asm ".operator[]";
            case glsl:
                __glslImageStore(location, newValue);
            case cuda:
                if (isArray != 0)
                {
                    switch(Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "surf1DLayeredwrite$C<$T0>($2, $0, ($1).x * $E, ($1).y, SLANG_CUDA_BOUNDARY_MODE)";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "surf2DLayeredwrite$C<$T0>($2, $0, ($1).x * $E, ($1).y, ($1).z, SLANG_CUDA_BOUNDARY_MODE)";
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "surf3DLayeredwrite$C<$T0>($2, $0, ($1).x * $E, ($1).y, ($1).z, ($1).w, SLANG_CUDA_BOUNDARY_MODE)";
                    default:
                        __intrinsic_asm "<invalid intrinsic>";
                    }
                }
                else
                {
                    switch(Shape.flavor)
                    {
                    case $(SLANG_TEXTURE_1D):
                        __intrinsic_asm "surf1Dwrite$C<$T0>($2, $0, ($1) * $E, SLANG_CUDA_BOUNDARY_MODE)";
                    case $(SLANG_TEXTURE_2D):
                        __intrinsic_asm "surf2Dwrite$C<$T0>($2, $0, ($1).x * $E, ($1).y, SLANG_CUDA_BOUNDARY_MODE)";
                    case $(SLANG_TEXTURE_3D):
                        __intrinsic_asm "surf3Dwrite$C<$T0>($2, $0, ($1).x * $E, ($1).y, ($1).z, SLANG_CUDA_BOUNDARY_MODE)";
                    default:
                        __intrinsic_asm "<invalid intrinsic>";
                    }
                }
            case spirv:
                return spirv_asm
                {
                    OpImageWrite $this $location __convertTexel(newValue);
                };
            case metal:
                if (isArray != 0)
                {
                    // last arg will be replaced with the split off array index
                    __metalImageStoreArray(this, __vectorReshape<Shape.dimensions>(location), newValue, location[Shape.dimensions + isArray - 1]);
                }
                else
                {
                    __metalImageStore(this, location, newValue);
                }
        case wgsl:
            static_assert(Shape.flavor == $(SLANG_TEXTURE_1D)
                || Shape.flavor == $(SLANG_TEXTURE_2D)
                || Shape.flavor == $(SLANG_TEXTURE_3D)
                , "WGSL doesn't supports textureStore for Cube texture.");
            static_assert(isArray == 0 || Shape.flavor == $(SLANG_TEXTURE_2D)
                , "WGSL supports textureStore for texture_store_2d_array but not for array of 1D, 3D or Cube.");

            // WGSL requires the value type to be always `vec4`
            if (isArray == 1)
            {
                if (T is int32_t   || T is int16_t  || T is int8_t)  __intrinsic_asm "textureStore($0, ($1).xy, i32(($1).z), vec4<i32>($V2)";
                if (T is int32_t2  || T is int16_t2  || T is int8_t2)  __intrinsic_asm "textureStore($0, ($1).xy, i32(($1).z), vec4<i32>($2, 0, 1))";
                if (T is int32_t3  || T is int16_t3  || T is int8_t3)  __intrinsic_asm "textureStore($0, ($1).xy, i32(($1).z), vec4<i32>($2, 1))";
                if (T is uint32_t  || T is uint16_t || T is uint8_t) __intrinsic_asm "textureStore($0, ($1).xy, i32(($1).z), vec4<u32>($2, 0, 0, 1))";
                if (T is uint32_t2 || T is uint16_t2 || T is uint8_t2) __intrinsic_asm "textureStore($0, ($1).xy, i32(($1).z), vec4<u32>($2, 0, 1))";
                if (T is uint32_t3 || T is uint16_t3 || T is uint8_t3) __intrinsic_asm "textureStore($0, ($1).xy, i32(($1).z), vec4<u32>($2, 1))";
                if (T is half)   __intrinsic_asm "textureStore($0, ($1).xy, i32(($1).z), vec4<f16>($2, 0, 0, 1))";
                if (T is half2)  __intrinsic_asm "textureStore($0, ($1).xy, i32(($1).z), vec4<f16>($2, 0, 1))";
                if (T is half3)  __intrinsic_asm "textureStore($0, ($1).xy, i32(($1).z), vec4<f16>($2, 1))";
                if (T is float)  __intrinsic_asm "textureStore($0, ($1).xy, i32(($1).z), vec4<f32>($2, 0, 0, 1))";
                if (T is float2) __intrinsic_asm "textureStore($0, ($1).xy, i32(($1).z), vec4<f32>($2, 0, 1))";
                if (T is float3) __intrinsic_asm "textureStore($0, ($1).xy, i32(($1).z), vec4<f32>($2, 1))";
                __intrinsic_asm "textureStore($0, ($1).xy, i32(($1).z), $2)";
            }
            if (T is int32_t   || T is int16_t  || T is int8_t)  __intrinsic_asm "textureStore($0, $1, vec4<i32>($2, 0, 0, 1))";
            if (T is int32_t2  || T is int16_t2 || T is int8_t2)  __intrinsic_asm "textureStore($0, $1, vec4<i32>($2, 0, 1))";
            if (T is int32_t3  || T is int16_t3 || T is int8_t3)  __intrinsic_asm "textureStore($0, $1, vec4<i32>($2, 1))";
            if (T is uint32_t  || T is uint16_t || T is uint8_t) __intrinsic_asm "textureStore($0, $1, vec4<u32>($2, 0, 0, 1))";
            if (T is uint32_t2 || T is uint16_t2 || T is uint8_t2) __intrinsic_asm "textureStore($0, $1, vec4<u32>($2, 0, 1))";
            if (T is uint32_t3 || T is uint16_t3 || T is uint8_t3) __intrinsic_asm "textureStore($0, $1, vec4<u32>($2, 1))";
            if (T is half)   __intrinsic_asm "textureStore($0, $1, vec4<f16>($2, 0, 0, 1))";
            if (T is half2)  __intrinsic_asm "textureStore($0, $1, vec4<f16>($2, 0, 1))";
            if (T is half3)  __intrinsic_asm "textureStore($0, $1, vec4<f16>($2, 1))";
            if (T is float)  __intrinsic_asm "textureStore($0, $1, vec4<f32>($2, 0, 0, 1))";
            if (T is float2) __intrinsic_asm "textureStore($0, $1, vec4<f32>($2, 0, 1))";
            if (T is float3) __intrinsic_asm "textureStore($0, $1, vec4<f32>($2, 1))";
            __intrinsic_asm "textureStore($0, $1, $2)";
        }
    }

    ${{{{
        if (access != kCoreModule_ResourceAccessWriteOnly)
        {
    }}}}
    __subscript(vector<uint, Shape.dimensions+isArray> location) -> T
    {
        [__readNone]
        [ForceInline]
        [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_1)]
        get
        {
            __target_switch
            {
                case cpp:
                case hlsl:
                    __intrinsic_asm ".operator[]";
                case glsl:
                case spirv:
                case cuda:
                case metal:
                case wgsl:
                    return Load(location);
            }
        }

        [nonmutating]
        [ForceInline]
        [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_1)]
        set(T newValue)
        {
            Store(location, newValue);
        }

        // If a 'Texture[location]' is referred to by a '__ref', call 'kIROp_ImageSubscript(location)'.
        // This allows call's to stay aware that the input is from a 'Texture'.
        __intrinsic_op($(kIROp_ImageSubscript)) 
        ref;
    }
    ${{{{
        } // if (access != kCoreModule_ResourceAccessWriteOnly)
    }}}}

}

${{{{
if (access == kCoreModule_ResourceAccessReadWrite) {
}}}}

// RW MS textures.

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let isShadow:int, let format:int>
extension _Texture<T,Shape,isArray,1,sampleCount,$(access),isShadow, 0,format>
{
    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_metal_spirv_wgsl, texture_sm_4_1_compute_fragment)]
    T Load(vector<int, Shape.dimensions+isArray> location, int sampleIndex)
    {
        __target_switch
        {
            case cpp:
            case hlsl:
                __intrinsic_asm ".Load";
            case metal:
                switch (Shape.flavor)
                {
                case $(SLANG_TEXTURE_2D):
                    if (isShadow == 1)
                    {
                        if (isArray == 1)
                            // The document seems to have a typo. `lod` must mean `sample`.
                            // Tv read(uint2 coord, uint array, uint lod = 0) const
                            __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy), uint(($1).z), $2)$z";
                        else
                            // T read(uint2 coord, uint sample) const
                            __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy), $2)$z";
                    }
                    else
                    {
                        if (isArray == 1)
                            // The document seems to have a typo. `lod` must mean `sample`.
                            // Tv read(uint2 coord, uint array, uint lod = 0) const
                            __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy), uint(($1).z), $2)$z";
                        else
                            // Tv read(uint2 coord, uint sample) const
                            __intrinsic_asm "$c$0.read(vec<uint,2>(($1).xy), $2)$z";
                    }
                    break;
                }
                // TODO: This needs to be handled by the capability system
                __intrinsic_asm "<Not supported>";
            case glsl:
                __intrinsic_asm "$(glslIntrinsicMS)";
            case spirv:
                return spirv_asm
                {
                    %sampled:__sampledType(T) = OpImageRead $this $location Sample $sampleIndex;
                    __truncate $$T result __sampledType(T) %sampled;
                };
            case wgsl:
                static_assert(Shape.flavor == $(SLANG_TEXTURE_2D)
                    , "WGSL supports textureLoad for texture_multisampled_2d but not for multisampled of 1D, 3D or Cube.");
                static_assert(isArray == 0
                    , "WGSL doesn't support array variants of multisampled textures for textureLoad.");

                __intrinsic_asm "textureLoad($0, $1, $2)$z";
        }
    }

    [__readNone]
    [ForceInline]
    [require(cpp_glsl_hlsl_spirv, texture_sm_4_1_compute_fragment)]
    T Load(vector<int, Shape.dimensions+isArray> location, int sampleIndex, vector<int, Shape.dimensions+isArray> offset)
    {
        __target_switch
        {
            case cpp:
            case hlsl:
                __intrinsic_asm ".Load";
            case glsl:
                __intrinsic_asm "$(glslIntrinsicMSOffset)";
            case spirv:
                return spirv_asm
                {
                    %sampled:__sampledType(T) = OpImageRead $this $location ConstOffset|Sample $offset $sampleIndex;
                    __truncate $$T result __sampledType(T) %sampled;
                };
        }
    }

    [__readNone]
    [ForceInline]
    T Load(vector<int, Shape.dimensions+isArray> location, int sampleIndex, vector<int, Shape.dimensions+isArray> offset, out uint status)
    {
        __target_switch
        {
        case hlsl:
        case cpp:
            __intrinsic_asm ".Load";
        default:
            status = 0;
            return Load(location, sampleIndex, offset);
        }
    }

    [require(glsl, texture_sm_4_1_compute_fragment)]
    void __glslImageStore(vector<int, Shape.dimensions+isArray> location, int sampleIndex, T value)
    {
        __intrinsic_asm "imageStore($0, $1, $2, $V3)";
    }

    __subscript(vector<uint, Shape.dimensions+isArray> location, int sampleIndex) -> T
    {
        [__readNone]
        [ForceInline]
        [require(cpp_glsl_hlsl_spirv_wgsl, texture_sm_4_1_compute_fragment)]
        get
        {
            __target_switch
            {
                case cpp:
                case hlsl:
                    __intrinsic_asm "$0.sample[$2][$1]";
                case glsl:
                case spirv:
                case wgsl:
                    return Load(location, sampleIndex);
            }
        }

        [nonmutating]
        [ForceInline]
        [require(cpp_glsl_hlsl_spirv, texture_sm_4_1_compute_fragment)]
        set(T newValue)
        {
            __target_switch
            {
                case cpp:
                case hlsl:
                    __intrinsic_asm "$0.sample[$2][$1]";
                case glsl:
                    __glslImageStore(location, sampleIndex, newValue);
                case spirv:
                    return spirv_asm
                    {
                        OpImageWrite $this $location __convertTexel(newValue) Sample $sampleIndex;
                    };
            }
        }

        // If a 'Texture[location, sampleIndex]' is referred to by a '__ref', call 'kIROp_ImageSubscript(location, sampleIndex)'.
        // This allows call's to stay aware that the input is from a 'Texture'.
        __intrinsic_op($(kIROp_ImageSubscript))
        ref;
    }
}

${{{{
} // if (access == kCoreModule_ResourceAccessReadWrite) // for RW MS textures.
} // for (access).
}}}}

// Definitions to support the legacy texture .mips[][] operator.
struct __TextureMip<T:ITexelElement, Shape : __ITextureShape, let isArray : int, let isCombined : int, let format : int>
{
    _Texture<T, Shape, isArray, 0 /*isMS*/, 0 /*sampleCount*/, 0 /*access*/, 0 /*isShadow*/, isCombined, format> tex;
    int mip;
    __subscript(vector<int, isArray + Shape.dimensions> pos)->T
    {
        [__unsafeForceInlineEarly]
        get { return tex.Load(__makeVector(pos, mip)); }
    }
}

struct __TextureMips<T:ITexelElement, Shape : __ITextureShape, let isArray : int, let isCombined : int, let format : int>
{
    _Texture<T, Shape, isArray, 0 /*isMS*/, 0 /*sampleCount*/, 0 /*access*/, 0 /*isShadow*/, isCombined, format> tex;
    __subscript(int mip)->__TextureMip<T, Shape, isArray, isCombined, format>
    {
        [__unsafeForceInlineEarly]
        get { return { tex, mip }; }
    }
}

//@hidden:
__generic<T:ITexelElement, Shape : __ITextureShape, let isArray : int, let isCombined : int, let format : int>
extension _Texture<T, Shape, isArray, 0 /*isMS*/, 0 /*sampleCount*/, 0 /*access*/, 0 /*isShadow*/, isCombined, format>
{
    property __TextureMips<T, Shape, isArray, isCombined, format> mips
    {
        [__unsafeForceInlineEarly]
        get { return { this }; }
    }
}

// Definitions to support the .sample[][] operator.
struct __TextureSample<T:ITexelElement, Shape : __ITextureShape, let isArray : int, let isCombined : int, let format : int>
{
    _Texture<T, Shape, isArray, 1 /*isMS*/, 0 /*sampleCount*/, 0 /*access*/, 0 /*isShadow*/, isCombined, format> tex;
    int sample;
    __subscript(vector<int, isArray + Shape.dimensions> pos)->T
    {
        [__unsafeForceInlineEarly]
        get { return tex[pos, sample]; }
    }
}

struct __TextureSampleMS<T:ITexelElement, Shape : __ITextureShape, let isArray : int, let isCombined : int, let format : int>
{
    _Texture<T, Shape, isArray, 1 /*isMS*/, 0 /*sampleCount*/, 0 /*access*/, 0 /*isShadow*/, isCombined, format> tex;
    __subscript(int sample)->__TextureSample<T, Shape, isArray, isCombined, format>
    {
        [__unsafeForceInlineEarly]
        get { return { tex, sample }; }
    }
}

__generic<T:ITexelElement, Shape : __ITextureShape, let isArray : int, let isCombined : int, let format : int>
extension _Texture<T, Shape, isArray, 1 /*isMS*/, 0 /*sampleCount*/, 0 /*access*/, 0 /*isShadow*/, isCombined, format>
{
    property __TextureSampleMS<T, Shape, isArray, isCombined, format> sample
    {
        [__unsafeForceInlineEarly]
        get { return { this }; }
    }
}

//@public:

// Texture type aliases.
// T, Shape: __ITextureShape, let isArray:int, let isMS:int, let sampleCount:int, let access:int, let isShadow:int, let isCombined:int, let format:int
${{{{
    const char* shapeTypeNames[] = {"1D", "2D", "3D", "Cube"};
    const char* accessPrefix[] = {"", "RW", "W", "RasterizerOrdered", "Feedback"};
    const char* accessDocumentation[] = {"read-only", "read-write", "write-only", "rasterizer-ordered", "feedback"};
    const char* arrayPostFix[] = {"", "Array"};
    const char* msPostFix[] = {"", "MS"};
    for (int shape = 0; shape < 4; shape++)
    for (int isArray = 0; isArray<=1; isArray++)
    for (int isMS = 0; isMS<=1; isMS++)
    for (int isCombined = 0; isCombined<=1; isCombined++)
    for (int access = kCoreModule_ResourceAccessReadOnly; access <= kCoreModule_ResourceAccessFeedback; access++) {
        if (access != kCoreModule_ResourceAccessReadOnly)
        {
            // No RW Cube.
            if (shape == kCoreModule_ShapeIndexCube) continue;
        }
        if (access == kCoreModule_ResourceAccessFeedback)
        {
            // Feedback only defined for Texture2D and Texture2DArray.
            if (shape != 1) continue;
            if (isMS) continue;
            if (isCombined) continue;
        }
        if (isMS)
        {
            // Only Texture2DMS.
            if (shape != kCoreModule_ShapeIndex2D)
                continue;
            // Only Texture2DMS or RWTexture2DMS.
            if (access >= kCoreModule_ShapeIndex3D)
                continue;
        }
        // No 3D Array.
        if (shape == kCoreModule_ShapeIndex3D && isArray == 1)
            continue;
        const char* textureTypeName = isCombined ? "Sampler" : "Texture";
}}}}
/// Represents a handle to a $(accessDocumentation[access])$(isMS?", multisampled": "") $(shapeTypeNames[shape]) $(isCombined?"combined texture-sampler": "texture")$(isArray?" array":"").
/// @param T The texel type of the texture.
/// @param sampleCount The number of samples in the texture, when the texture is multisampled.
/// @param format The storage format of the texture.
/// @see Please refer to `_Texture` for more information about texture types.
/// @category texture_types
typealias $(accessPrefix[access])$(textureTypeName)$(shapeTypeNames[shape])$(msPostFix[isMS])$(arrayPostFix[isArray])<T:ITexelElement=float4, let sampleCount:int=0, let format:int=0> = _Texture<T, __Shape$(shapeTypeNames[shape]), $(isArray), $(isMS), sampleCount, $(access), 0, $(isCombined), format>;
${{{{
}
}}}}

// Declare Sampler*Shadow type aliases.
${{{{
    for (int shape = 0; shape < 4; shape++)
    for (int isArray = 0; isArray<=1; isArray++)
    {
}}}}
/// Represents a handle to a $(shapeTypeNames[shape]) combined texture-sampler for shadow comparison.
/// @param format The storage format of the texture.
/// @see Please refer to `_Texture` for more information about texture types.
/// @category texture_types
typealias Sampler$(shapeTypeNames[shape])$(arrayPostFix[isArray])Shadow<int format=0> = _Texture<float, __Shape$(shapeTypeNames[shape]), $(isArray), 0, 0, 0, 1, 1, format>;
${{{{
    }
}}}}

// Atomic intrinsic insts.

__intrinsic_op($(kIROp_AtomicExchange))
T __atomic_exchange<T>(__ref T val, T newValue, MemoryOrder order = MemoryOrder.Relaxed);
__intrinsic_op($(kIROp_AtomicCompareExchange))
T __atomic_compare_exchange<T>(
    __ref T val,
    T compareValue,
    T newValue,
    MemoryOrder successOrder = MemoryOrder.Relaxed,
    MemoryOrder failOrder = MemoryOrder.Relaxed);
__intrinsic_op($(kIROp_AtomicAdd))
T __atomic_add<T>(__ref T val, T value, MemoryOrder order = MemoryOrder.Relaxed);
__intrinsic_op($(kIROp_AtomicSub))
T __atomic_sub<T>(__ref T val, T value, MemoryOrder order = MemoryOrder.Relaxed);
__intrinsic_op($(kIROp_AtomicMax))
T __atomic_max<T>(__ref T val, T value, MemoryOrder order = MemoryOrder.Relaxed);
__intrinsic_op($(kIROp_AtomicMin))
T __atomic_min<T>(__ref T val, T value, MemoryOrder order = MemoryOrder.Relaxed);
__intrinsic_op($(kIROp_AtomicAnd))
T __atomic_and<T>(__ref T val, T value, MemoryOrder order = MemoryOrder.Relaxed);
__intrinsic_op($(kIROp_AtomicOr))
T __atomic_or<T>(__ref T val, T value, MemoryOrder order = MemoryOrder.Relaxed);
__intrinsic_op($(kIROp_AtomicXor))
T __atomic_xor<T>(__ref T val, T value, MemoryOrder order = MemoryOrder.Relaxed);
__intrinsic_op($(kIROp_AtomicInc))
T __atomic_increment<T>(__ref T val, MemoryOrder order = MemoryOrder.Relaxed);
__intrinsic_op($(kIROp_AtomicDec))
T __atomic_decrement<T>(__ref T val, MemoryOrder order = MemoryOrder.Relaxed);

// Conversion between uint64_t and uint2

[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
uint2 __asuint2(uint64_t i)
{
    return uint2(uint(i), uint(uint64_t(i) >> 32));
}

[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
uint64_t __asuint64(uint2 i)
{
    return (uint64_t(i.y) << 32) | i.x;
}

//

__intrinsic_op($(kIROp_ByteAddressBufferLoad))
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, byteaddressbuffer)]
T __byteAddressBufferLoad<T>(ByteAddressBuffer buffer, int offset, int alignment);

__intrinsic_op($(kIROp_ByteAddressBufferLoad))
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, byteaddressbuffer_rw)]
T __byteAddressBufferLoad<T>(RWByteAddressBuffer buffer, int offset, int alignment);

__intrinsic_op($(kIROp_ByteAddressBufferLoad))
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, byteaddressbuffer_rw)]
T __byteAddressBufferLoad<T>(RasterizerOrderedByteAddressBuffer buffer, int offset, int alignment);

__intrinsic_op($(kIROp_ByteAddressBufferStore))
[require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
void __byteAddressBufferStore<T>(RWByteAddressBuffer buffer, int offset, int alignment, T value);

__intrinsic_op($(kIROp_ByteAddressBufferStore))
[require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
void __byteAddressBufferStore<T>(RasterizerOrderedByteAddressBuffer buffer, int offset, int alignment, T value);

__intrinsic_op($(kIROp_GetUntypedBufferPtr))
[require(spirv, byteaddressbuffer)]
Ptr<uint[]> __getByteAddressBufferPtr(ByteAddressBuffer buffer);

__intrinsic_op($(kIROp_GetUntypedBufferPtr))
[require(spirv, byteaddressbuffer_rw)]
Ptr<uint[]> __getByteAddressBufferPtr(RWByteAddressBuffer buffer);

__intrinsic_op($(kIROp_GetStructuredBufferPtr))
[require(spirv, structuredbuffer)]
Ptr<T[]> __getStructuredBufferPtr<T>(StructuredBuffer<T> buffer);

__intrinsic_op($(kIROp_GetStructuredBufferPtr))
[require(spirv, structuredbuffer_rw)]
Ptr<T[]> __getStructuredBufferPtr<T>(RWStructuredBuffer<T> buffer);

/**
Represents an opaque handle to a read-only structured buffer allocated in global memory.
A structured buffer can be viewed as an array of the specified element type.
@param T The element type of the buffer.
@param L The memory layout of the buffer.
@remarks
The `L` generic parameter is used to specify the memory layout of the buffer when
generating SPIRV.
`L` must be one of `DefaultDataLayout`, `Std140DataLayout`, `Std430DataLayout` or `ScalarDataLayout`.
The default value is `DefaultDataLayout`.
When generating code for other targets, this parameter is ignored and has no effect on the generated code.
@see `RWStructuredBuffer`, `AppendStructuredBuffer`, `ConsumeStructuredBuffer`, `RasterizerOrderedStructuredBuffer`.
@category buffer_types Buffer types
**/
__generic<T, L:IBufferDataLayout=DefaultDataLayout>
__magic_type(HLSLStructuredBufferType)
__intrinsic_type($(kIROp_HLSLStructuredBufferType))
struct StructuredBuffer
{

    /// Get the dimensions of the buffer.
    /// @param numStructs The number of structures in the buffer.
    /// @param stride The stride, in bytes, of each structure element.
    [__readNone]
    [ForceInline]
    void GetDimensions(
        out uint numStructs,
        out uint stride)
    {
        let rs = __structuredBufferGetDimensions(this);
        numStructs = rs.x;
        stride = rs.y;
    }

    /// Load a element from the buffer at the specified location.
    /// @param TIndex Type of the index.
    /// @param location The index of buffer.
    /// @param[out] status The status of the operation.
    /// @return The element at the specified index.
    ///
    /// @remarks
    /// You can't access the output parameter `status` directly; instead,
    /// pass the status to the `CheckAccessFullyMapped` intrinsic function.
    /// `CheckAccessFullyMapped` returns TRUE if all values from the corresponding Sample,
    /// Gather, or Load operation accessed mapped tiles in a tiled resource.
    /// If any values were taken from an unmapped tile, `CheckAccessFullyMapped` returns FALSE.
    __intrinsic_op($(kIROp_StructuredBufferLoad))
    [__readNone]
    [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, structuredbuffer)]
    T Load<TIndex : __BuiltinIntegerType>(TIndex location);

    __intrinsic_op($(kIROp_StructuredBufferLoadStatus))
    [require(hlsl, structuredbuffer)]
    T Load<TIndex : __BuiltinIntegerType>(TIndex location, out uint status);

    /// Load a element from the buffer at the specified location.
    /// @param TIndex Type of the index.
    /// @param index The index of buffer.
    /// @return The element at the specified index.
    __generic<TIndex : __BuiltinIntegerType>
    __subscript(TIndex index) -> T
    {
        [__readNone]
        __intrinsic_op($(kIROp_StructuredBufferLoad))
        [require(cpp_cuda_glsl_hlsl_spirv, structuredbuffer)]
        get;
    };
};

/**
Represents an opaque handle to a consume structured buffer allocated in global memory.
A structured buffer can be viewed as an array of the specified element type.
An append structure buffer internally maintains an atomic counter to keep track of the number of elements in the buffer,
and provide an atomic operation to append a new element to the buffer.
@param T The element type of the buffer.
@param L The memory layout of the buffer.
@remarks
This type is supported natively when targeting HLSL.
When generating code for other targets, this type is translated into a pair or an ordinary `StructuredBuffer` and
a separate `RWStructuredBuffer` that holds the atomic counter.
The `L` generic parameter is used to specify the memory layout of the buffer when
generating SPIRV.
`L` must be one of `DefaultDataLayout`, `Std140DataLayout`, `Std430DataLayout` or `ScalarDataLayout`.
The default value is `DefaultDataLayout`.
When generating code for other targets, this parameter is ignored and has no effect on the generated code.
@see `StructuredBuffer`, `AppendStructuredBuffer`, `RWStructuredBuffer`, `RasterizerOrderedStructuredBuffer`.
@category buffer_types
*/
__generic<T, L:IBufferDataLayout=DefaultDataLayout>
__magic_type(HLSLConsumeStructuredBufferType)
__intrinsic_type($(kIROp_HLSLConsumeStructuredBufferType))
[require(cpp_cuda_glsl_hlsl_spirv, consumestructuredbuffer)]
struct ConsumeStructuredBuffer
{
    /// Reading the element at the end of the buffer indicated by the associated atomic counter
    /// and decrement the builtin atomic counter by 1.
    ///@return The element read from the buffer, it can be a structure.
    __intrinsic_op($(kIROp_StructuredBufferConsume))
    T Consume();

    ///Gets the dimensions of the resource.
    ///@param[out] numStructs The number of structures in the buffer.
    ///@param[out] stride The stride, in bytes, of each element

    [ForceInline]
    void GetDimensions(
        out uint numStructs,
        out uint stride)
    {
        let result = __structuredBufferGetDimensions(this);
        numStructs = result.x;
        stride = result.y;
    }
};

__intrinsic_op($(kIROp_GetElement))
T __getElement<T, U, I>(U collection, I index);

/// @category stage_io Stage IO types
__generic<T, let N : int>
[require(glsl_hlsl_spirv, hull)]
__magic_type(HLSLInputPatchType)
__intrinsic_type($(kIROp_HLSLInputPatchType))
struct InputPatch
{
    __generic<TIndex : __BuiltinIntegerType>
    __subscript(TIndex index)->T
    {
        [__unsafeForceInlineEarly]
        get
        {
            __target_switch
            {
            case hlsl:
                __intrinsic_asm ".operator[]";
            default:
                return __getElement<T>(this, index);
            }
        }
    }
};

/// @category stage_io
__generic<T, let N : int>
[require(glsl_hlsl_spirv, domain_hull)]
__magic_type(HLSLOutputPatchType)
__intrinsic_type($(kIROp_HLSLOutputPatchType))
struct OutputPatch
{
    __generic<TIndex : __BuiltinIntegerType>
    __subscript(TIndex index)->T
    {
        [__unsafeForceInlineEarly]
        get
        {
            __target_switch
            {
            case hlsl:
                __intrinsic_asm ".operator[]";
            default:
                return __getElement<T>(this, index);
            }
        }
    }
};

${{{{
static const struct {
    IROp op;
    char const* name;
} kMutableByteAddressBufferCases[] =
{
    { kIROp_HLSLRWByteAddressBufferType,                "RWByteAddressBuffer" },
    { kIROp_HLSLRasterizerOrderedByteAddressBufferType, "RasterizerOrderedByteAddressBuffer" },
};
for(auto item : kMutableByteAddressBufferCases) {
}}}}

//@public:
/**
Represents an opaque handle to a read-write buffer allocated in global memory that is indexed in bytes.
This type can be used when working with raw buffers. Raw buffer can be viewed as a bag of bits to
which you want raw access, that is, a buffer that you can conveniently access through chunks of one to
four 32-bit typeless address values.
 @remarks
This type is supported natively when targeting HLSL.
 @category buffer_types
*/
__magic_type(HLSL$(item.name)Type)
__intrinsic_type($(item.op))
struct $(item.name)
{
    // Note(tfoley): supports all operations from `ByteAddressBuffer`
    // TODO(tfoley): can this be made a sub-type?

    /// Get the number of bytes in the buffer.
    ///@param[out] dim The number of bytes in the buffer.
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_spirv_wgsl, structuredbuffer_rw)]
    void GetDimensions(out uint dim)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm ".GetDimensions";
        case cuda: __intrinsic_asm ".GetDimensions";
        case hlsl: __intrinsic_asm ".GetDimensions";
        default:
            dim = __structuredBufferGetDimensions(__getEquivalentStructuredBuffer<uint>(this)).x*4;
        }
    }

    /// Load a 32-bit unsigned integer or value with type of `T` from the buffer at the specified location.
    ///@param T The type of the value to load from the buffer.
    ///@param location The input address in bytes, which must be a multiple of 4.
    ///@param alignment Specifies the alignment of the location, which must be a multiple of 4.
    ///@param[out] status The status of the operation.
    ///@return The value loaded from the buffer.
    ///
    ///@remarks
    /// You can't access the output parameter `status` directly; instead,
    /// pass the status to the `CheckAccessFullyMapped` intrinsic function.
    /// `CheckAccessFullyMapped` returns TRUE if all values from the corresponding Sample,
    /// Gather, or Load operation accessed mapped tiles in a tiled resource.
    /// If any values were taken from an unmapped tile, `CheckAccessFullyMapped` returns FALSE.
    /// When targeting non-HLSL, the status is always 0.
    [__NoSideEffect]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, byteaddressbuffer_rw)]
    uint Load(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load";
        default:
            return __byteAddressBufferLoad<uint>(this, location, 0);
        }
    }

    [__NoSideEffect]
    [ForceInline]
    [require(hlsl, byteaddressbuffer_rw)]
    uint Load(int location, out uint status)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load";
        }
    }

    /// Load two 32-bit unsigned integers from the buffer at the specified location
    /// with additional alignment.
    ///@param location The input address in bytes.
    ///@param alignment Specifies the alignment of the location, which must be a multiple of 4.
    ///@param[out] status The status of the operation.
    ///@return Two 32-bit unsigned integers loaded from the buffer.
    ///
    ///@remarks
    /// This function only supports when targeting HLSL.
    /// You can't access the output parameter `status` directly; instead,
    /// pass the status to the `CheckAccessFullyMapped` intrinsic function.
    /// `CheckAccessFullyMapped` returns TRUE if all values from the corresponding Sample,
    /// Gather, or Load operation accessed mapped tiles in a tiled resource.
    /// If any values were taken from an unmapped tile, `CheckAccessFullyMapped` returns FALSE.
    /// When targeting non-HLSL, the status is always 0.
    [__NoSideEffect]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    uint2 Load2(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load2";
        default:
            return __byteAddressBufferLoad<uint2>(this, location, 0);
        }
    }

    [__NoSideEffect]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    uint2 Load2(int location, int alignment)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load2";
        default:
            return __byteAddressBufferLoad<uint2>(this, location, alignment);
        }
    }

    /// Load two 32-bit unsigned integers from the buffer at the specified location with alignment
    /// of `uint2`, which is 8.
    ///@param location The input address in bytes, which must be a multiple of alignment of 8.
    ///@return `uint2` Two 32-bit unsigned integers loaded from the buffer.
    [__NoSideEffect]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    uint2 Load2Aligned(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load2";
        default:
            return __byteAddressBufferLoad<uint2>(this, location, __naturalStrideOf<uint2>());
        }
    }

    [__NoSideEffect]
    [ForceInline]
    [require(hlsl, byteaddressbuffer_rw)]
    uint2 Load2(int location, out uint status)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load2";
        }
    }

    /// Load three 32-bit unsigned integers from the buffer at the specified location.
    ///@param location The input address in bytes, which must be a multiple of 4.
    ///@param alignment Specifies the alignment of the location, which must be a multiple of 4.
    ///@param[out] status The status of the operation.
    ///@return `uint3` Three 32-bit unsigned integer value loaded from the buffer.
    ///
    ///@remarks
    /// This function only supports when targeting HLSL.
    /// You can't access the output parameter `status` directly; instead,
    /// pass the status to the `CheckAccessFullyMapped` intrinsic function.
    /// `CheckAccessFullyMapped` returns TRUE if all values from the corresponding Sample,
    /// Gather, or Load operation accessed mapped tiles in a tiled resource.
    /// If any values were taken from an unmapped tile, `CheckAccessFullyMapped` returns FALSE.
    /// When targeting non-HLSL, the status is always 0.
    [__NoSideEffect]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    uint3 Load3(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load3";
        default:
            return __byteAddressBufferLoad<uint3>(this, location, 0);
        }
    }

    [__NoSideEffect]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    uint3 Load3(int location, int alignment)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load3";
        default:
            return __byteAddressBufferLoad<uint3>(this, location, alignment);
        }
    }

    /// Load three 32-bit unsigned integers from the buffer at the specified location with alignment
    /// of `uint3`, which is 12.
    ///@param location The input address in bytes which must be a multiple of alignment of 12.
    ///@return `uint3` Three 32-bit unsigned integer value loaded from the buffer.
    [__NoSideEffect]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    uint3 Load3Aligned(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load3";
        default:
            return __byteAddressBufferLoad<uint3>(this, location, __naturalStrideOf<uint3>());
        }
    }

    [__NoSideEffect]
    [ForceInline]
    [require(hlsl, byteaddressbuffer_rw)]
    uint3 Load3(int location, out uint status)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load3";
        }
    }

    /// Load four 32-bit unsigned integers from the buffer at the specified location.
    ///@param location The input address in bytes which must be a multiple of alignment of 4.
    ///@param alignment Specifies the alignment of the location, which must be a multiple of 4.
    ///@param[out] status The status of the operation.
    ///@return `uint4` Four 32-bit unsigned integer value loaded from the buffer.
    ///
    ///@remarks
    /// This function only supports when targeting HLSL.
    /// You can't access the output parameter `status` directly; instead,
    /// pass the status to the `CheckAccessFullyMapped` intrinsic function.
    /// `CheckAccessFullyMapped` returns TRUE if all values from the corresponding Sample,
    /// Gather, or Load operation accessed mapped tiles in a tiled resource.
    /// If any values were taken from an unmapped tile, `CheckAccessFullyMapped` returns FALSE.
    [__NoSideEffect]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    uint4 Load4(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load4";
        default:
            return __byteAddressBufferLoad<uint4>(this, location, 0);
        }
    }

    [__NoSideEffect]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    uint4 Load4(int location, int alignment)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load4";
        default:
            return __byteAddressBufferLoad<uint4>(this, location, alignment);
        }
    }

    /// Load four 32-bit unsigned integers from the buffer at the specified location with alignment
    /// of `uint4`, which is 16.
    ///@param location The input address in bytes which must be a multiple of alignment of 16.
    ///@return `uint4` Four 32-bit unsigned integer value loaded from the buffer.
    [__NoSideEffect]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    uint4 Load4Aligned(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load4";
        default:
            return __byteAddressBufferLoad<uint4>(this, location, __naturalStrideOf<uint4>());
        }
    }

    [__NoSideEffect]
    [ForceInline]
    [require(hlsl, byteaddressbuffer_rw)]
    uint4 Load4(int location, out uint status)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load4";
        }
    }

    [__NoSideEffect]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    T Load<T>(int location)
    {
        return __byteAddressBufferLoad<T>(this, location, 0);
    }

    [__NoSideEffect]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    T Load<T>(int location, int alignment)
    {
        return __byteAddressBufferLoad<T>(this, location, alignment);
    }

    /// Load an element with type `T` from the buffer at the specified location with alignment of `T`.
    ///@param location The input address in bytes which must be a multiple of size of `T`.
    ///@return T value with type `T` loaded from the buffer.
    ///@remarks
    ///Currently, this function only supports when `T` is scalar, vector, or matrix type.
    [__NoSideEffect]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    T LoadAligned<T>(int location)
    {
        return __byteAddressBufferLoad<T>(this, location, __naturalStrideOf<T>());
    }

${{{{
    struct BufferAtomicOps
    {
        const char* name;
        const char* internalName;
    };
    const BufferAtomicOps bufferAtomicOps[] = {
        {"Max", "max"},
        {"Min", "min"},
        {"Add", "add"},
        {"And", "and"},
        {"Or", "or"},
        {"Xor", "xor"},
        {"Exchange", "exchange"}
    };
    if (item.op == kIROp_HLSLRWByteAddressBufferType)
    {
}}}}

    // float32 and int64 atomic support. This is a Slang specific extension, it uses
    // GL_EXT_shader_atomic_float on Vulkan
    // NvAPI support on DX
    // NOTE! To use this feature on HLSL based targets the path to 'nvHLSLExtns.h' from the NvAPI SDK must
    // be set. That this include will be added to the *output* that is passed to a downstram compiler.
    // Also note that you *can* include NVAPI headers in your Slang source, and directly use NVAPI functions
    // Directly using NVAPI functions does *not* add the #include on the output
    // Finally note you can *mix* NVAPI direct calls, and use of NVAPI intrinsics below. This doesn't cause
    // any clashes, as Slang will emit any NVAPI function it parsed (say via a include in Slang source) with
    // unique functions.
    //
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#VK_EXT_shader_atomic_float
    // https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/EXT/SPV_EXT_shader_atomic_float_add.html

    // F32 Add

    /// Perform a 32-bit floating point atomic add operation at `byteAddress`.
    /// @param byteAddress The address at which to perform the atomic add operation.
    /// @param valueToAdd The value to add to the value at `byteAddress`.
    /// @param originalValue The original value at `byteAddress` before the add operation.
    /// @remarks For SPIR-V, this function maps to `OpAtomicFAdd`. For HLSL, this function translates to an NVAPI call
    /// due to lack of native HLSL intrinsic for floating point atomic add. For CUDA, this function
    /// maps to `atomicAdd`.
    __cuda_sm_version(2.0)
    [__requiresNVAPI]
    [ForceInline]
    [require(cuda_glsl_hlsl_metal_spirv, atomic_glsl_hlsl_nvapi_cuda_metal_float1)]
    void InterlockedAddF32(uint byteAddress, float valueToAdd, out float originalValue)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "($3 = NvInterlockedAddFp32($0, $1, $2))";
        case cuda: __intrinsic_asm "(*$3 = atomicAdd($0._getPtrAt<float>($1), $2))";
        default:
            {
                let buf = __getEquivalentStructuredBuffer<float>(this);
                originalValue = __atomic_add(buf[byteAddress / 4], valueToAdd);
                return;
            }
        }
    }

    [require(cuda, cuda_sm_6_0)]
    [require(spirv, spvAtomicFloat64AddEXT)]
    void InterlockedAddF64(uint byteAddress, double valueToAdd, out double originalValue)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "(*$3 = atomicAdd($0._getPtrAt<double>($1), $2))";
        default:
            {
                let buf = __getEquivalentStructuredBuffer<double>(this);
                originalValue = __atomic_add(buf[byteAddress / 8], valueToAdd);
                return;
            }
        }
    }
    // FP16x2

    ///@internal
    /// Maps to the `NvInterlockedAddFp16x2` NVAPI function.
    /// Perform 2 16-bit floating point atomic add operations at `byteAddress`.
    /// @param byteAddress The address at which to perform the atomic add operation.
    /// @param fp16x2Value Two 16-bit floating point values are packed into a 32-bit unsigned integer.
    /// @return The 2 16-bit floating point values packed into a 32-bit unsigned integer.
    [__requiresNVAPI]
    [ForceInline]
    [require(cuda_hlsl_spirv)]
    uint _NvInterlockedAddFp16x2(uint byteAddress, uint fp16x2Value)
    {
        __target_switch
        {
        case hlsl:
            __intrinsic_asm "NvInterlockedAddFp16x2($0, $1, $2)";
        default:
            let buf = __getEquivalentStructuredBuffer<half2>(this);
            return bit_cast<uint>(__atomic_add(buf[byteAddress / 4], bit_cast<half2>(fp16x2Value)));
        }
    }


    /// Perform a 16-bit floating point atomic add operation at `byteAddress`.
    /// @param byteAddress The address at which to perform the atomic add operation.
    /// @param value The value to add to the value at `byteAddress`.
    /// @param originalValue The original value at `byteAddress` before the add operation.
    /// @remarks For SPIR-V, this function maps to `OpAtomicFAdd` and requires `SPV_EXT_shader_atomic_float16_add` extension.
    ///
    /// For HLSL, this function translates to an NVAPI call
    /// due to lack of native HLSL intrinsic for floating point atomic add. For CUDA, this function
    /// maps to `atomicAdd`.
    [__requiresNVAPI]
    [ForceInline]
    void InterlockedAddF16(uint byteAddress, half value, out half originalValue)
    {
        __target_switch
        {
        case hlsl:
            if ((byteAddress & 2) == 0)
            {
                uint packedInput = asuint16(value);
                originalValue = asfloat16((uint16_t)_NvInterlockedAddFp16x2(byteAddress, packedInput));
            }
            else
            {
                byteAddress = byteAddress & ~3;
                uint packedInput = ((uint)asuint16(value)) << 16;
                originalValue = asfloat16((uint16_t)(_NvInterlockedAddFp16x2(byteAddress, packedInput) >> 16));
            }
            return;
        default:
            {
                let buf = __getEquivalentStructuredBuffer<half>(this);
                originalValue = __atomic_add(buf[byteAddress/2], value);
                return;
            }
        }
    }

    /// Perform a 16-bit floating point atomic add operation at `byteAddress` through emulation using `half2` atomics.
    /// @param byteAddress The address at which to perform the atomic add operation.
    /// @param value The value to add to the value at `byteAddress`.
    /// @param originalValue The original value at `byteAddress` before the add operation.
    /// @remarks For SPIR-V, this function maps to `OpAtomicFAdd` on a `half2` vector with the correct part set to `value`
    /// and the remaining part set to 0. This requires the `AtomicFloat16VectorNV` capability introduced by the `SPV_NV_shader_atomic_fp16_vector`
    /// extension.
    ///
    /// For HLSL, this function translates to an equivalent NVAPI call
    /// due to lack of native HLSL intrinsic for floating point atomic add. For CUDA, this function
    /// maps to `atomicAdd`.
    [__requiresNVAPI]
    [ForceInline]
    void InterlockedAddF16Emulated(uint byteAddress, half value, out half originalValue)
    {
        __target_switch
        {
        case hlsl:
            if ((byteAddress & 2) == 0)
            {
                uint packedInput = asuint16(value);
                originalValue = asfloat16((uint16_t)_NvInterlockedAddFp16x2(byteAddress, packedInput));
            }
            else
            {
                byteAddress = byteAddress & ~3;
                uint packedInput = ((uint)asuint16(value)) << 16;
                originalValue = asfloat16((uint16_t)(_NvInterlockedAddFp16x2(byteAddress, packedInput) >> 16));
            }
            return;
        default:
            {
                let buf = __getEquivalentStructuredBuffer<half2>(this);
                if ((byteAddress & 2) == 0)
                {
                    originalValue = __atomic_add(buf[byteAddress/4], half2(value, half(0.0))).x;
                }
                else
                {
                    originalValue = __atomic_add(buf[byteAddress/4], half2(half(0.0), value)).y;
                }
                return;
            }
        }
    }

    // Without returning original value

    [__requiresNVAPI]
    [ForceInline]
    __cuda_sm_version(2.0)
    [require(cuda_glsl_hlsl_metal_spirv, atomic_glsl_hlsl_nvapi_cuda_metal_float1)]
    void InterlockedAddF32(uint byteAddress, float valueToAdd)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "(NvInterlockedAddFp32($0, $1, $2))";
        default:
            {
                let buf = __getEquivalentStructuredBuffer<float>(this);
                __atomic_add(buf[byteAddress / 4], valueToAdd);
                return;
            }
        }
    }

    // Int64 Add

    /// Perform a 64-bit integer atomic add operation at `byteAddress`.
    /// @param byteAddress The address at which to perform the atomic add operation.
    /// @param valueToAdd The value to add to the value at `byteAddress`.
    /// @param originalValue The original value at `byteAddress` before the add operation.
    /// @remarks For SPIR-V, this function maps to `OpAtomicAdd`. For HLSL, this function
    /// translates to `InterlockedAdd64` and requires shader model 6.6.
    /// For CUDA, this function maps to `atomicAdd`.
    [ForceInline]
    [require(cuda_glsl_hlsl_spirv, atomic_glsl_hlsl_cuda9_int64)]
    void InterlockedAddI64(uint byteAddress, int64_t valueToAdd, out int64_t originalValue)
    {
        InterlockedAdd64(byteAddress, valueToAdd, originalValue);
    }

    // Without returning original value
    [ForceInline]
    [require(cuda_glsl_hlsl_spirv, atomic_glsl_hlsl_cuda9_int64)]
    void InterlockedAddI64(uint byteAddress, int64_t valueToAdd)
    {
        InterlockedAdd64(byteAddress, valueToAdd);
    }

    // Cas uint64_t

    /// Perform a 64-bit integer atomic compare-and-exchange operation at `byteAddress`.
    /// @param byteAddress The address at which to perform the atomic compare-and-exchange operation.
    /// @param compareValue The value to compare to the value at `byteAddress`.
    /// @param value The value to store at `byteAddress` if the comparison is successful.
    /// @param outOriginalValue The original value at `byteAddress` before the add operation.
    /// @remarks For SPIR-V, this function maps to `OpAtomicCompareExchange`. For HLSL, this function
    /// translates to `InterlockedCompareExchange64` and requires shader model 6.6.
    /// For CUDA, this function maps to `atomicCAS`.
    [require(cuda_glsl_hlsl_spirv, atomic_glsl_hlsl_cuda9_int64)]
    void InterlockedCompareExchangeU64(uint byteAddress, uint64_t compareValue, uint64_t value, out uint64_t outOriginalValue)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "(*$4 = atomicCAS($0._getPtrAt<uint64_t>($1), $2, $3))";
        case hlsl:
            __intrinsic_asm ".InterlockedCompareExchange64";
        default:
            let buf = __getEquivalentStructuredBuffer<uint64_t>(this);
            outOriginalValue = __atomic_compare_exchange(buf[byteAddress / 8], compareValue, value);
        }
    }

    // SM6.6 6 64bit atomics.

    // InterlockedMax64, InterlockedMin64, InterlockedAdd64, InterlockedAnd64, InterlockedOr64, InterlockedXor64, InterlockedExchange64
${{{{
    for (auto op : bufferAtomicOps) {
}}}}

    /// Perform a 64-bit unsigned integer atomic $(op.internalName) operation at `byteAddress`.
    /// @param byteAddress The address at which to perform the atomic $(op.internalName) operation.
    /// @param value The operand for the $(op.internalName) operation.
    [ForceInline]
    [require(cuda_glsl_hlsl_spirv, atomic_glsl_hlsl_cuda9_int64)]
    uint64_t Interlocked$(op.name)U64(uint byteAddress, uint64_t value)
    {
        uint64_t originalValue;
        Interlocked$(op.name)64(byteAddress, value, originalValue);
        return originalValue;
    }

    [ForceInline]
    [require(cuda_glsl_hlsl_spirv, atomic_glsl_hlsl_cuda9_int64)]
    void Interlocked$(op.name)64(uint byteAddress, int64_t value)
    {
        int64_t oldValue;
        Interlocked$(op.name)64(byteAddress, value, oldValue);
    }

    /// Perform a 64-bit integer atomic $(op.internalName) operation at `byteAddress`.
    /// @param byteAddress The address at which to perform the atomic $(op.internalName) operation.
    /// @param value The operand for the $(op.internalName) operation.
    /// @param outOriginalValue The original value at `byteAddress` before the $(op.internalName) operation.
    [ForceInline]
    [require(cuda_glsl_hlsl_spirv, atomic_glsl_hlsl_cuda9_int64)]
    void Interlocked$(op.name)64<T:__BuiltinInt64Type>(uint byteAddress, T value, out T outOriginalValue)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Interlocked$(op.name)64";
        default:
            let buf = __getEquivalentStructuredBuffer<T>(this);
            outOriginalValue = __atomic_$(op.internalName)(buf[byteAddress / 8], value);
            return;
        }
    }
${{{{
} // for (each bufferOps)
}}}}

    /// Perform a 64-bit integer atomic compare-and-exchange operation at `byteAddress`.
    /// @param byteAddress The address at which to perform the atomic compare-and-exchange operation.
    /// @param compareValue The value to compare to the value at `byteAddress`.
    /// @param value The value to store at `byteAddress` if the comparison is successful.
    /// @param outOriginalValue The original value at `byteAddress` before the add operation.
    /// @remarks For SPIR-V, this function maps to `OpAtomicCompareExchange`. For HLSL, this function
    /// translates to `InterlockedCompareExchange64` and requires shader model 6.6.
    /// For CUDA, this function maps to `atomicCAS`.
    [ForceInline]
    [require(cuda_glsl_hlsl_spirv, atomic_glsl_hlsl_cuda9_int64)]
    void InterlockedCompareExchange64<T:__BuiltinInt64Type>(uint byteAddress, T compareValue, T value, out T outOriginalValue)
    {
        __target_switch
        {
        case hlsl:
            __intrinsic_asm ".InterlockedCompareExchange64";
        default:
            let buf = __getEquivalentStructuredBuffer<T>(this);
            outOriginalValue = __atomic_compare_exchange(buf[byteAddress / 8], compareValue, value);
            return;
        }
    }

    /// Perform a floating-point atomic bitwise compare-and-exchange operation at `byteAddress`.
    /// @param byteAddress The address at which to perform the atomic exchange operation.
    /// @param compareValue The value to compare to the value at `byteAddress`.
    /// @param value The value to store at `byteAddress`.
    /// @param [out] outOriginalValue The original value at `byteAddress` before the exchange operation.
    /// @remarks For SPIR-V, this function maps to `OpAtomicCompareExchange`. For HLSL, this function
    /// translates to `InterlockedCompareExchangeFloatBitwise` and requires shader model 6.6.
    /// For CUDA, this function maps to `atomicCAS`.
    [ForceInline]
    [require(cuda_glsl_hlsl_spirv, atomic_glsl_hlsl_cuda9_int64)]
    void InterlockedCompareExchangeFloatBitwise(uint byteAddress, float compareValue, float value, out float outOriginalValue)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".InterlockedCompareExchangeFloatBitwise";
        default:
            let buf = __getEquivalentStructuredBuffer<float>(this);
            outOriginalValue = __atomic_compare_exchange(buf[byteAddress / 4], compareValue, value);
            return;
        }
    }

    /// Perform a floating-point atomic bitwise exchange operation at `byteAddress`.
    /// @param byteAddress The address at which to perform the atomic exchange operation.
    /// @param value The value to store at `byteAddress`.
    /// @param [out] outOriginalValue The original value at `byteAddress` before the exchange operation.
    /// @remarks For SPIR-V, this function maps to `OpAtomicExchange`. For HLSL, this function
    /// translates to `InterlockedExchangeFloat` and requires shader model 6.6.
    /// For CUDA, this function maps to `atomicExch`.
    [ForceInline]
    [require(cuda_glsl_hlsl_spirv, atomic_glsl_hlsl_cuda9_int64)]
    void InterlockedExchangeFloat(uint byteAddress, float value, out float outOriginalValue)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".InterlockedExchangeFloat";
        default:
            let buf = __getEquivalentStructuredBuffer<float>(this);
            outOriginalValue = __atomic_exchange(buf[byteAddress / 4], value);
            return;
        }
    }

    /// Perform a 64-bit integer atomic compare-and-store operation at `byteAddress`.
    /// @param byteAddress The address at which to perform the atomic store operation.
    /// @param compareValue The value to compare to the value at `byteAddress`.
    /// @param value The value to store at `byteAddress` if the the value at address is equal to `compareValue`.
    /// @param [out] outOriginalValue The original value at `byteAddress` before the store operation.
    /// @remarks For SPIR-V, this function maps to `OpAtomicCompareExchange`. For HLSL, this function
    /// translates to `InterlockedCompareStore64` and requires shader model 6.6.
    /// For CUDA, this function maps to `atomicCAS`.
    [ForceInline]
    [ForceInline]
    [require(cuda_glsl_hlsl_spirv, atomic_glsl_hlsl_cuda9_int64)]
    void InterlockedCompareStore64<T:__BuiltinInt64Type>(uint byteAddress, T compareValue, T value)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".InterlockedCompareStore64";
        default:
            let buf = __getEquivalentStructuredBuffer<T>(this);
            __atomic_compare_exchange(buf[byteAddress / 4], compareValue, value);
            return;
        }
    }
   
    /// Perform a floating-point atomic bitwise compare-and-store operation at `byteAddress`.
    /// @param byteAddress The address at which to perform the atomic compare-and-exchange  operation.
    /// @param compareValue The value to perform bitwise comparison to the value at `byteAddress`.
    /// @param value The value to store at `byteAddress` if the comparison is successful.
    /// @param [out] outOriginalValue The original value at `byteAddress` before the compare-and-exchange operation.
    /// @remarks For SPIR-V, this function maps to `OpAtomicCompareExchange`. For HLSL, this function
    /// translates to `InterlockedCompareStoreFloatBitwise` and requires shader model 6.6.
    /// For CUDA, this function maps to `atomicCAS`.
    [ForceInline]
    [require(cuda_glsl_hlsl_spirv, atomic_glsl_hlsl_cuda9_int64)]
    void InterlockedCompareStoreFloatBitwise(uint byteAddress, float compareValue, float value)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".InterlockedCompareStoreFloatBitwise";
        default:
            let buf = __getEquivalentStructuredBuffer<float>(this);
            __atomic_compare_exchange(buf[byteAddress / 4], compareValue, value);
            return;
        }
    }

${{{{
    } // endif (type == RWByteAddressBuffer)
}}}}

    // 32-bit atomic operations:
    // InterlockedMax, InterlockedMin, InterlockedAdd, InterlockedAnd, InterlockedOr, InterlockedXor, InterlockedExchange
${{{{
    for (auto op : bufferAtomicOps) {
}}}}

    /// Perform an atomic $(op.internalName) operation at the specified byte
    /// location of the byte address buffer.
    /// @param dest The byte address at which to perform the atomic $(op.internalName) operation.
    /// @param value The operand of the atomic operation.
    /// @param original_value The original value at `dest` before the $(op.internalName) operation.
    [ForceInline]
    [require(cuda_glsl_hlsl_metal_spirv, atomic_glsl_hlsl_cuda_metal, byteaddressbuffer_rw)]
    void Interlocked$(op.name)(
        UINT dest,
        UINT value,
        out UINT original_value)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Interlocked$(op.name)";
        default:
            let buf = __getEquivalentStructuredBuffer<uint>(this);
            ::Interlocked$(op.name)(buf[dest / 4], value, original_value);
        }
    }

    [ForceInline]
    [require(cuda_glsl_hlsl_metal_spirv, atomic_glsl_hlsl_cuda_metal, byteaddressbuffer_rw)]
    void Interlocked$(op.name)(
        UINT dest,
        UINT value)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Interlocked$(op.name)";
        default:
            let buf = __getEquivalentStructuredBuffer<uint>(this);
            ::Interlocked$(op.name)(buf[dest / 4], value);
        }
    }
${{{{
} // for (buffer atomic ops)
}}}}

    /// Perform a 32-bit integer atomic compare-and-exchange operation at
    /// the specified byte address within the `RWByteAddressBuffer`.
    /// @param dest The address at which to perform the atomic compare-and-exchange operation.
    /// @param compare_value The value to perform bitwise comparison to the value at `byteAddress`.
    /// @param value The value to store at `byteAddress` if the comparison is successful.
    /// @param original_value The original value at `byteAddress` before the compare-and-exchange operation.
    /// @remarks For SPIR-V, this function maps to `OpAtomicCompareExchange`. For HLSL, this function
    /// translates to `InterlockedCompareExchange`.
    /// For CUDA, this function maps to `atomicCAS`.
    [ForceInline]
    [require(cuda_glsl_hlsl_metal_spirv, atomic_glsl_hlsl_cuda_metal, byteaddressbuffer_rw)]
    void InterlockedCompareExchange(
        UINT dest,
        UINT compare_value,
        UINT value,
        out UINT original_value)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".InterlockedCompareExchange";
        default:
            let buf = __getEquivalentStructuredBuffer<uint>(this);
            ::InterlockedCompareExchange(buf[dest / 4], compare_value, value, original_value);
        }
    }

    /// Perform a 32-bit integer atomic compare-and-store operation at
    /// the specified byte address within the `RWByteAddressBuffer`.
    /// @param dest The address at which to perform the atomic add operation.
    /// @param compare_value The value to perform comparison to the value at `byteAddress`.
    /// @param value The value to store at `byteAddress` if the comparison is successful.
    /// @remarks For SPIR-V, this function maps to `OpAtomicCompareExchange`. For HLSL, this function
    /// translates to `InterlockedCompareStore`.
    /// For CUDA, this function maps to `atomicCAS`.
    [ForceInline]
    [require(cuda_glsl_hlsl_metal_spirv, atomic_glsl_hlsl_cuda_metal, byteaddressbuffer_rw)]
    void InterlockedCompareStore(
        UINT dest,
        UINT compare_value,
        UINT value)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".InterlockedCompareStore";
        default:
            let buf = __getEquivalentStructuredBuffer<uint>(this);
            ::InterlockedCompareStore(buf[dest / 4], compare_value, value);
        }
    }

    /// Set one value to the buffer at the specified location.
    ///@param T The type of the value to load from the buffer.
    ///@param value The input value.
    ///@param address The input address in bytes, which must be a multiple of 4.
    ///@param alignment Specifies the alignment of the location, which must be a multiple of 4.
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    void Store(uint address, uint value)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Store";
        default:
            __byteAddressBufferStore(this, address, 0, value);
        }
    }


    /// Set two values to the buffer at the specified location.
    ///@param address The input address in bytes, which must be a multiple of 4.
    ///@param value Two input values.
    ///@param alignment Specifies the alignment of the location, which must be a multiple of 4.
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    void Store2(uint address, uint2 value)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Store2";
        default:
            __byteAddressBufferStore(this, address, 0, value);
        }
    }


    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    void Store2(uint address, uint2 value, uint alignment)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Store2";
        default:
            __byteAddressBufferStore(this, address, alignment, value);
        }
    }

    /// Set two values to the buffer at the specified location, the address will be aligned
    /// to the alignment of  `uint2`, which is 8.
    ///@param address The input address in bytes, which must be a multiple of 8.
    ///@param value Two input values.
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    void Store2Aligned(uint address, uint2 value)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Store2";
        default:
            __byteAddressBufferStore(this, address, __naturalStrideOf<uint2>(), value);
        }
    }

    /// Set three values to the buffer at the specified location.
    ///@param address The input address in bytes, which must be a multiple of 4.
    ///@param value Three input values.
    ///@param alignment Specifies the alignment of the location, which must be a multiple of 4.
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    void Store3(uint address, uint3 value)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Store3";
        default:
            __byteAddressBufferStore(this, address, 0, value);
        }
    }

    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    void Store3(uint address, uint3 value, uint alignment)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Store3";
        default:
            __byteAddressBufferStore(this, address, alignment, value);
        }
    }

    /// Set three values to the buffer at the specified location, the address will be aligned
    /// to the alignment of `uint3`, which is 12.
    ///@param address The input address in bytes, which must be a multiple of 12.
    ///@param value Three input values.
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_spirv, byteaddressbuffer_rw)]
    void Store3Aligned(uint address, uint3 value)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Store3";
        default:
            __byteAddressBufferStore(this, address, __naturalStrideOf<uint3>(), value);
        }
    }

    /// Set four values to the buffer at the specified location.
    ///@param address The input address in bytes, which must be a multiple of 4.
    ///@param value Four input values.
    ///@param alignment Specifies the alignment of the location, which must be a multiple of 4.
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_spirv, byteaddressbuffer_rw)]
    void Store4(uint address, uint4 value)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Store4";
        default:
            __byteAddressBufferStore(this, address, 0, value);
        }
    }


    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    void Store4(uint address, uint4 value, uint alignment)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Store4";
        default:
            __byteAddressBufferStore(this, address, alignment, value);
        }
    }

    /// Set four values to the buffer at the specified location, the address will be aligned
    /// to the alignment of `uint4`, which is 16.
    ///@param address The input address in bytes, which must be a multiple of 16.
    ///@param value Four input values.
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv, byteaddressbuffer_rw)]
    void Store4Aligned(uint address, uint4 value)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Store4";
        default:
            __byteAddressBufferStore(this, address, __naturalStrideOf<uint4>(), value);
        }
    }

    [ForceInline]
    void Store<T>(uint address, T value)
    {
        __byteAddressBufferStore(this, address, 0, value);
    }

    [ForceInline]
    void Store<T>(uint address, T value, uint alignment)
    {
        __byteAddressBufferStore(this, address, alignment, value);
    }

    /// Set four values to the buffer at the specified location, the address will be aligned
    /// to the alignment of `T`.
    ///@param T The type of the input value.
    ///@param address The input address in bytes, which must be a multiple of size of `T`.
    ///@param value The input value.
    [ForceInline]
    void StoreAligned<T>(uint address, T value)
    {
        __byteAddressBufferStore(this, address, __naturalStrideOf<T>(), value);
    }
};

${{{{
}
}}}}

${{{{
static const struct {
    IROp op;
    char const* name;
} kMutableStructuredBufferCases[] =
{
    { kIROp_HLSLRWStructuredBufferType,                "RWStructuredBuffer" },
    { kIROp_HLSLRasterizerOrderedStructuredBufferType, "RasterizerOrderedStructuredBuffer" },
};
for(auto item : kMutableStructuredBufferCases) {
}}}}

__generic<T, L:IBufferDataLayout=DefaultDataLayout>
__magic_type(HLSL$(item.name)Type)
__intrinsic_type($(item.op))
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, structuredbuffer_rw)]
/**
Represents an opaque handle to a mutable structured buffer allocated in global memory.
A structured buffer can be viewed as an array of the specified element type.
 @param T The element type of the buffer.
 @param L The memory layout of the buffer.
 @remarks
The `L` generic parameter is used to specify the memory layout of the buffer when
generating SPIRV.
`L` must be one of `DefaultDataLayout`, `Std140DataLayout`, `Std430DataLayout` or `ScalarDataLayout`.
The default value is `DefaultDataLayout`.
When generating code for other targets, this parameter is ignored and has no effect on the generated code.
 @see `StructuredBuffer`, `AppendStructuredBuffer`, `ConsumeStructuredBuffer`
 @category buffer_types
**/
struct $(item.name)
{
    /// Decrements the object's hidden counter.
    /// @return The post-decremented counter value.
    /// @remarks
    /// This function is not implemented when targeting non-HLSL.
    uint DecrementCounter();

    /// Get the dimensions of the buffer.
    /// @param numStructs The number of structures in the buffer.
    /// @param stride The stride, in bytes, of each structure element.
    [__readNone]
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, structuredbuffer_rw)]
    void GetDimensions(
        out uint numStructs,
        out uint stride)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".GetDimensions";
        default:
            let rs = __structuredBufferGetDimensions(this);
            numStructs = rs.x;
            stride = rs.y;
        }
    }

    /// Increment the object's hidden counter.
    /// @return The pre-incremented counter value.
    /// @remarks
    /// This function is not implemented when targeting non-HLSL.
    uint IncrementCounter();

    /// Load a element from the buffer at the specified location.
    /// @param TIndex Type of the index.
    /// @param location The index of buffer.
    /// @param[out] status The status of the operation.
    /// @return The element at the specified index.
    ///
    /// @remarks
    /// You can't access the output parameter `status` directly; instead,
    /// pass the status to the `CheckAccessFullyMapped` intrinsic function.
    /// `CheckAccessFullyMapped` returns TRUE if all values from the corresponding Sample,
    /// Gather, or Load operation accessed mapped tiles in a tiled resource.
    /// If any values were taken from an unmapped tile, `CheckAccessFullyMapped` returns FALSE.
    /// When targeting non-HLSL, the status is always 0.
    [__NoSideEffect]
    __intrinsic_op($(kIROp_RWStructuredBufferLoad))
    T Load<TIndex : __BuiltinIntegerType>(TIndex location);

    [__NoSideEffect]
    __intrinsic_op($(kIROp_RWStructuredBufferLoadStatus))
    T Load<TIndex : __BuiltinIntegerType>(TIndex location, out uint status);

    /// Load a element from the buffer at the specified location.
    /// @param TIndex Type of the index.
    /// @param index The index of buffer.
    /// @return The element at the specified index.
    __generic<TIndex : __BuiltinIntegerType>
    __subscript(TIndex index) -> T
    {
        // If a 'Buffer[index]' is referred to by a '__ref', call 'kIROp_RWStructuredBufferGetElementPtr(index)'.
        //
        // This allows call's to stay aware that the input is from a 'Buffer'.
        [__NoSideEffect]
        [nonmutating]
        __intrinsic_op($(kIROp_RWStructuredBufferGetElementPtr))
        ref;
    }
};

${{{{
}
}}}}

/// @category stage_io
__generic<T>
[require(glsl_hlsl_spirv, geometry)]
__magic_type(HLSLPointStreamType)
__intrinsic_type($(kIROp_HLSLPointStreamType))
struct PointStream
{
    [KnownBuiltin("GeometryStreamAppend")]
    void Append(T value)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "EmitVertex()";
        case hlsl: __intrinsic_asm ".Append";
        case spirv: spirv_asm { OpEmitVertex; };
        }
    }

    [KnownBuiltin("GeometryStreamRestart")]
    void RestartStrip()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "EndPrimitive()";
        case hlsl: __intrinsic_asm ".RestartStrip";
        case spirv: spirv_asm { OpEndPrimitive; };
        }
    }
};

/// @category stage_io
__generic<T>
[require(glsl_hlsl_spirv, geometry)]
__magic_type(HLSLLineStreamType)
__intrinsic_type($(kIROp_HLSLLineStreamType))
struct LineStream
{
    [KnownBuiltin("GeometryStreamAppend")]
    void Append(T value)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "EmitVertex()";
        case hlsl: __intrinsic_asm ".Append";
        case spirv: spirv_asm { OpEmitVertex; };
        }
    }

    [KnownBuiltin("GeometryStreamRestart")]
    void RestartStrip()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "EndPrimitive()";
        case hlsl: __intrinsic_asm ".RestartStrip";
        case spirv: spirv_asm { OpEndPrimitive; };
        }
    }
};

/// @category stage_io
__generic<T>
[require(glsl_hlsl_spirv, geometry)]
__magic_type(HLSLTriangleStreamType)
__intrinsic_type($(kIROp_HLSLTriangleStreamType))
struct TriangleStream
{
    [KnownBuiltin("GeometryStreamAppend")]
    void Append(T value)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "EmitVertex()";
        case hlsl: __intrinsic_asm ".Append";
        case spirv: spirv_asm { OpEmitVertex; };
        }
    }

    [KnownBuiltin("GeometryStreamRestart")]
    void RestartStrip()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "EndPrimitive()";
        case hlsl: __intrinsic_asm ".RestartStrip";
        case spirv: spirv_asm { OpEndPrimitive; };
        }
    }
};

#define VECTOR_MAP_UNARY(TYPE, COUNT, FUNC, VALUE) \
    vector<TYPE,COUNT> result; for(int i = 0; i < COUNT; ++i) { result[i] = FUNC(VALUE[i]); } return result

#define MATRIX_MAP_UNARY(TYPE, ROWS, COLS, FUNC, VALUE) \
    matrix<TYPE,ROWS,COLS> result; for(int i = 0; i < ROWS; ++i) { result[i] = FUNC(VALUE[i]); } return result

#define VECTOR_MAP_BINARY(TYPE, COUNT, FUNC, LEFT, RIGHT) \
    vector<TYPE,COUNT> result; for(int i = 0; i < COUNT; ++i) { result[i] = FUNC(LEFT[i], RIGHT[i]); } return result

#define MATRIX_MAP_BINARY(TYPE, ROWS, COLS, FUNC, LEFT, RIGHT) \
    matrix<TYPE,ROWS,COLS> result; for(int i = 0; i < ROWS; ++i) { result[i] = FUNC(LEFT[i], RIGHT[i]); } return result

#define VECTOR_MAP_TRINARY(TYPE, COUNT, FUNC, A, B, C) \
    vector<TYPE,COUNT> result; for(int i = 0; i < COUNT; ++i) { result[i] = FUNC(A[i], B[i], C[i]); } return result

#define MATRIX_MAP_TRINARY(TYPE, ROWS, COLS, FUNC, A, B, C) \
    matrix<TYPE,ROWS,COLS> result; for(int i = 0; i < ROWS; ++i) { result[i] = FUNC(A[i], B[i], C[i]); } return result

//@public:

/// Try to terminate the current draw or dispatch call (HLSL SM 4.0)
void abort();

/// The abs function returns the absolute value of x.
/// @param x The input value.
/// @return The absolute value of x.
/// @category math
__generic<T : __BuiltinIntegerType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T abs(T x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "abs";
    case glsl: __intrinsic_asm "abs";
    case metal: __intrinsic_asm "abs";
    case cuda: __intrinsic_asm "$P_abs($0)";
    case cpp: __intrinsic_asm "$P_abs($0)";
    case spirv: return spirv_asm {
        result:$$T = OpExtInst glsl450 SAbs $x
    };
    case wgsl: __intrinsic_asm "abs";
    //default:
    // Note: this simple definition may not be appropriate for floating-point inputs
    // return x < 0 ? -x : x;
    }
}

__generic<T : __BuiltinIntegerType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> abs(vector<T, N> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "abs";
    case glsl: __intrinsic_asm "abs";
    case metal: __intrinsic_asm "abs";
    case spirv: return spirv_asm {
        result:$$vector<T,N> = OpExtInst glsl450 SAbs $x;
    };
    case wgsl: __intrinsic_asm "abs";
    default:
        VECTOR_MAP_UNARY(T, N, abs, x);
    }
}

__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> abs(matrix<T,N,M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "abs";
    default:
        MATRIX_MAP_UNARY(T, N, M, abs, x);
    }
}

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T abs(T x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "abs";
    case metal: __intrinsic_asm "abs";
    case glsl: __intrinsic_asm "abs";
    case cuda: __intrinsic_asm "$P_abs($0)";
    case cpp: __intrinsic_asm "$P_abs($0)";
    case spirv: return spirv_asm {
        result:$$T = OpExtInst glsl450 FAbs $x;
    };
    case wgsl: __intrinsic_asm "abs";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> abs(vector<T, N> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "abs";
    case metal: __intrinsic_asm "abs";
    case glsl: __intrinsic_asm "abs";
    case spirv: return spirv_asm {
        result:$$vector<T,N> = OpExtInst glsl450 FAbs $x;
    };
    case wgsl: __intrinsic_asm "abs";
    default:
        VECTOR_MAP_UNARY(T, N, abs, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> abs(matrix<T,N,M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "abs";
    default:
        MATRIX_MAP_UNARY(T, N, M, abs, x);
    }
}

/// Float-point absolute value.
/// @param x The input value.
/// @return The absolute value of `x`.
/// @remarks For metal targets, this function is equivalent to the `fabs` metal intrinsic.
///          For other targets, this function is equivalent to the `abs` slang function.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T fabs(T x)
{
    __target_switch
    {
    case metal: __intrinsic_asm "fabs";
    default:
        return abs(x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> fabs(vector<T, N> x)
{
    __target_switch
    {
    case metal: __intrinsic_asm "fabs";
    default:
        return abs(x);
    }
}


/// Arc cosine. Returns the angle whose cosine is the specified number.
/// @param x The cosine value.
/// @return The angle in radians, in the range of [0, pi].
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T acos(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_acos($0)";
    case cuda: __intrinsic_asm "$P_acos($0)";
    case glsl: __intrinsic_asm "acos";
    case hlsl: __intrinsic_asm "acos";
    case metal: __intrinsic_asm "acos";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Acos $x
    };
    case wgsl: __intrinsic_asm "acos";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> acos(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "acos";
    case hlsl: __intrinsic_asm "acos";
    case metal: __intrinsic_asm "acos";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Acos $x
    };
    case wgsl: __intrinsic_asm "acos";
    default:
        VECTOR_MAP_UNARY(T, N, acos, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> acos(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "acos";
    default:
        MATRIX_MAP_UNARY(T, N, M, acos, x);
    }
}

/// Arc hyperbolic cosine. Returns the arc hyperbolic cosine of the specified value.
/// @param x The value.
/// @return The arc hyperbolic cosine of the specified value.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T acosh(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_acosh($0)";
    case cuda: __intrinsic_asm "$P_acosh($0)";
    case glsl: __intrinsic_asm "acosh";
    case metal: __intrinsic_asm "acosh";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Acosh $x
    };
    case wgsl: __intrinsic_asm "acosh";
    default:
        return log(x + sqrt( x * x - T(1)));
    }
}

__generic<T : __BuiltinFloatingPointType, let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> acosh(vector<T,N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "acosh";
    case metal: __intrinsic_asm "acosh";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,N> result glsl450 Acosh $x
    };
    case wgsl: __intrinsic_asm "acosh";
    default:
        VECTOR_MAP_UNARY(T, N, acosh, x);
    }
}


// Test if all components are non-zero.
__generic<T : __BuiltinType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl)]
bool all(T x)
{
    __target_switch
    {
    default:
        __intrinsic_asm "bool($0)";
    case hlsl:
        __intrinsic_asm "all";
    case metal:
        __intrinsic_asm "all";
    case wgsl:
        __intrinsic_asm "all";
    case spirv:
        let zero = __default<T>();
        if (__isInt<T>())
            return spirv_asm
            {
                OpINotEqual $$bool result $x $zero
            };
        else if (__isFloat<T>())
            return spirv_asm
            {
                OpFUnordNotEqual $$bool result $x $zero
            };
        else if (__isBool<T>())
            return __slang_noop_cast<bool>(x);
        else
            return false;
    }
}

__generic<T : __BuiltinType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl)]
bool all(vector<T,N> x)
{
    if(N == 1)
        return all(x[0]);
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "all";
    case metal:
        if (__isBool<T>())
            __intrinsic_asm "all";
        __intrinsic_asm "all(bool$N0($0))";
    case glsl:
        __intrinsic_asm "all(bvec$N0($0))";
    case spirv:
        if (__isBool<T>())
            return spirv_asm
            {
                OpAll $$bool result $x
            };
        else if (__isInt<T>())
        {
            let zero = __default<vector<T,N>>();
            return spirv_asm
            {
                OpINotEqual $$vector<bool,N> %castResult $x $zero;
                OpAll $$bool result %castResult
            };
        }
        else
        {
            let zero = __default<vector<T,N>>();
            return spirv_asm
            {
                OpFUnordNotEqual $$vector<bool,N> %castResult $x $zero;
                OpAll $$bool result %castResult
            };
        }
    case wgsl:
        if (__isBool<T>())
            __intrinsic_asm "all";
        __intrinsic_asm "all(vec$N0<bool>($0))";
    default:
        bool result = true;
        for(int i = 0; i < N; ++i)
            result = result && all(x[i]);
        return result;
    }
}

__generic<T : __BuiltinType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv)]
bool all(matrix<T,N,M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "all";
    default:
        bool result = true;
        for(int i = 0; i < N; ++i)
            result = result && all(x[i]);
        return result;
    }
}

/// Barrier for writes to all memory spaces.
/// @category barrier Memory and control barriers
__glsl_extension(GL_KHR_memory_scope_semantics)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, memorybarrier)]
void AllMemoryBarrier()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "AllMemoryBarrier";
    case glsl: __intrinsic_asm "memoryBarrier(gl_ScopeDevice, (gl_StorageSemanticsShared|gl_StorageSemanticsImage|gl_StorageSemanticsBuffer), gl_SemanticsAcquireRelease)";
    case cuda: __intrinsic_asm "__threadfence()";
    case metal: __intrinsic_asm "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock)";
    case spirv: spirv_asm
        {
            OpMemoryBarrier Device AcquireRelease|UniformMemory|WorkgroupMemory|ImageMemory;
        };
    case wgsl: __intrinsic_asm "storageBarrier(); textureBarrier(); workgroupBarrier();";
    }
}

/// Thread-group sync and barrier for writes to all memory spaces.
/// @category barrier
__glsl_extension(GL_KHR_memory_scope_semantics)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, memorybarrier)]
void AllMemoryBarrierWithGroupSync()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "AllMemoryBarrierWithGroupSync";
    case glsl: __intrinsic_asm "controlBarrier(gl_ScopeWorkgroup, gl_ScopeDevice, (gl_StorageSemanticsShared|gl_StorageSemanticsImage|gl_StorageSemanticsBuffer), gl_SemanticsAcquireRelease)";
    case cuda: __intrinsic_asm "__syncthreads()";
    case metal: __intrinsic_asm "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock)";
    case spirv: spirv_asm
        {
            OpControlBarrier Workgroup Device AcquireRelease|UniformMemory|WorkgroupMemory|ImageMemory;
        };
    case wgsl: __intrinsic_asm "storageBarrier(); textureBarrier(); workgroupBarrier();";
    }
}

// Returns the workgroup size of the calling entry point.
[require(compute)]
__intrinsic_op($(kIROp_GetWorkGroupSize))
int3 WorkgroupSize();

// Test if any components is non-zero.

__generic<T : __BuiltinType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl)]
bool any(T x)
{
    __target_switch
    {
    default:
        __intrinsic_asm "bool($0)";
    case hlsl:
        __intrinsic_asm "any";
    case metal:
        __intrinsic_asm "any";
    case wgsl:
        __intrinsic_asm "any";
    case spirv:
        let zero = __default<T>();
        if (__isInt<T>())
            return spirv_asm
            {
                OpINotEqual $$bool result $x $zero
            };
        else if (__isFloat<T>())
            return spirv_asm
            {
                OpFUnordNotEqual $$bool result $x $zero
            };
        else if (__isBool<T>())
            return __slang_noop_cast<bool>(x);
        return false;
    }
}

__generic<T : __BuiltinType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl)]
bool any(vector<T, N> x)
{
    if(N == 1)
        return any(x[0]);
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "any";
    case metal:
        __intrinsic_asm "any";
    case glsl:
        __intrinsic_asm "any(bvec$N0($0))";
    case spirv:
        if (__isBool<T>())
            return spirv_asm
            {
                OpAny $$bool result $x
            };
        else if (__isInt<T>())
        {
            let zero = __default<vector<T,N>>();
            return spirv_asm
            {
                OpINotEqual $$vector<bool,N> %castResult $x $zero;
                OpAny $$bool result %castResult
            };
        }
        else
        {
            let zero = __default<vector<T,N>>();
            return spirv_asm
            {
                OpFUnordNotEqual $$vector<bool,N> %castResult $x $zero;
                OpAny $$bool result %castResult
            };
        }
    case wgsl:
        __intrinsic_asm "any";
    default:
        bool result = false;
        for(int i = 0; i < N; ++i)
            result = result || any(x[i]);
        return result;
    }
}

__generic<T : __BuiltinType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv)]
bool any(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "any";
    default:
        bool result = false;
        for(int i = 0; i < N; ++i)
            result = result || any(x[i]);
        return result;
    }
}


/// Reinterpret bits as a double.
/// @category conversion
__glsl_extension(GL_ARB_gpu_shader5)
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
double asdouble(uint lowbits, uint highbits)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asdouble";
    case glsl: __intrinsic_asm "packDouble2x32(uvec2($0, $1))";
    case cpp: __intrinsic_asm "$P_asdouble($0, $1)";
    case cuda: __intrinsic_asm "$P_asdouble($0, $1)";
    case spirv: return spirv_asm {
        %v:$$uint2 = OpCompositeConstruct $lowbits $highbits;
        result:$$double = OpExtInst glsl450 59 %v
    };
    }
}

__glsl_extension(GL_ARB_gpu_shader5)
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
double2 asdouble(uint2 lowbits, uint2 highbits)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "asdouble($0, $1)";
    default:
        return double2(asdouble(lowbits.x, highbits.x), asdouble(lowbits.y, highbits.y));
    }
}

/// Reinterpret bits as a float.
/// @category conversion
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_4_0)]
float asfloat(int x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_asfloat($0)";
    case cuda: __intrinsic_asm "$P_asfloat($0)";
    case glsl: __intrinsic_asm "intBitsToFloat";
    case hlsl: __intrinsic_asm "asfloat";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$float result $x
    };
    case wgsl: __intrinsic_asm "bitcast<$TR>($0)";
    }
}

[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_4_0)]
float asfloat(uint x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_asfloat($0)";
    case cuda: __intrinsic_asm "$P_asfloat($0)";
    case glsl: __intrinsic_asm "uintBitsToFloat";
    case hlsl: __intrinsic_asm "asfloat";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$float result $x
    };
    case wgsl: __intrinsic_asm "bitcast<$TR>($0)";
    }
}

__generic<let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_4_0)]
vector<float, N> asfloat(vector< int, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "intBitsToFloat";
    case hlsl: __intrinsic_asm "asfloat";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$vector<float, N> result $x
    };
    case wgsl: __intrinsic_asm "bitcast<$TR>($0)";
    default:
        VECTOR_MAP_UNARY(float, N, asfloat, x);
    }
}

__generic<let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_4_0)]
vector<float,N> asfloat(vector<uint,N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "uintBitsToFloat";
    case hlsl: __intrinsic_asm "asfloat";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$vector<float,N> result $x
    };
    case wgsl: __intrinsic_asm "bitcast<$TR>($0)";
    default:
        VECTOR_MAP_UNARY(float, N, asfloat, x);
    }
}

__generic<let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
matrix<float,N,M> asfloat(matrix< int,N,M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asfloat";
    default:
        MATRIX_MAP_UNARY(float, N, M, asfloat, x);
    }
}

__generic<let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
matrix<float,N,M> asfloat(matrix<uint,N,M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asfloat";
    default:
        MATRIX_MAP_UNARY(float, N, M, asfloat, x);
    }
}

[__unsafeForceInlineEarly]
[__readNone]
float asfloat(float x)
{ return x; }

__generic<let N : int>
[__unsafeForceInlineEarly]
[__readNone]
vector<float,N> asfloat(vector<float,N> x)
{ return x; }

__generic<let N : int, let M : int>
[__unsafeForceInlineEarly]
[__readNone]
matrix<float,N,M> asfloat(matrix<float,N,M> x)
{ return x; }

/// Arc sine. Returns the angle whose sine is the specified number.
/// @param x The sine value.
/// @return The angle in radians, in the range of [-pi/2, pi/2].
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T asin(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_asin($0)";
    case cuda: __intrinsic_asm "$P_asin($0)";
    case glsl: __intrinsic_asm "asin";
    case hlsl: __intrinsic_asm "asin";
    case metal: __intrinsic_asm "asin";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Asin $x
    };
    case wgsl: __intrinsic_asm "asin";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> asin(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "asin";
    case hlsl: __intrinsic_asm "asin";
    case metal: __intrinsic_asm "asin";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Asin $x
    };
    case wgsl: __intrinsic_asm "asin";
    default:
        VECTOR_MAP_UNARY(T,N,asin,x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> asin(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asin";
    default:
        MATRIX_MAP_UNARY(T,N,M,asin,x);
    }
}

/// Arc hyperbolic sine. Returns the arc hyperbolic sine of the specified value.
/// @param x The hyperbolic sine value.
/// @return The arc hyperbolic sine of the specified value.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T asinh(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_asinh($0)";
    case cuda: __intrinsic_asm "$P_asinh($0)";
    case glsl: __intrinsic_asm "asinh";
    case metal: __intrinsic_asm "asinh";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Asinh $x
    };
    case wgsl: __intrinsic_asm "asinh";
    default:
        return log(x + sqrt(x * x + T(1)));
    }
}

__generic<T : __BuiltinFloatingPointType, let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> asinh(vector<T,N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "asinh";
    case metal: __intrinsic_asm "asinh";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,N> result glsl450 Asinh $x
    };
    case wgsl: __intrinsic_asm "asinh";
    default:
        VECTOR_MAP_UNARY(T, N, asinh, x);
    }
}

/// Reinterpret bits as an int.
/// @category conversion
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_4_0)]
int asint(float x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_asint($0)";
    case cuda: __intrinsic_asm "$P_asint($0)";
    case glsl: __intrinsic_asm "floatBitsToInt";
    case hlsl: __intrinsic_asm "asint";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case wgsl: __intrinsic_asm "bitcast<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$int result $x
    };
    }
}

[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_4_0)]
int asint(uint x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_asint($0)";
    case cuda: __intrinsic_asm "$P_asint($0)";
    case glsl: __intrinsic_asm "int($0)";
    case hlsl: __intrinsic_asm "asint";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$int result $x
    };
    case wgsl: __intrinsic_asm "bitcast<$TR>($0)";
    }
}

__generic<let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_4_0)]
vector<int, N> asint(vector<float, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "floatBitsToInt";
    case hlsl: __intrinsic_asm "asint";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$vector<int, N> result $x
    };
    case wgsl: __intrinsic_asm "bitcast<$TR>($0)";
    default:
        VECTOR_MAP_UNARY(int, N, asint, x);
    }
}

__generic<let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_4_0)]
vector<int, N> asint(vector<uint, N> x)
{
    if(N == 1)
        return vector<int, N>(asint(x[0]));
    __target_switch
    {
    case glsl: __intrinsic_asm "ivec$N0($0)";
    case hlsl: __intrinsic_asm "asint";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$vector<int, N> result $x
    };
    case wgsl: __intrinsic_asm "bitcast<$TR>($0)";
    default:
        VECTOR_MAP_UNARY(int, N, asint, x);
    }
}

__generic<let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
matrix<int, N, M> asint(matrix<float, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asint";
    default:
        MATRIX_MAP_UNARY(int, N, M, asint, x);
    }
}

__generic<let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
matrix<int, N, M> asint(matrix<uint, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asint";
    default:
        MATRIX_MAP_UNARY(int, N, M, asint, x);
    }
}

// No op
[__unsafeForceInlineEarly]
[__readNone]
int asint(int x)
{ return x; }

__generic<let N : int>
[__unsafeForceInlineEarly]
[__readNone]
vector<int,N> asint(vector<int,N> x)
{ return x; }

__generic<let N : int, let M : int>
[__unsafeForceInlineEarly]
[__readNone]
matrix<int,N,M> asint(matrix<int,N,M> x)
{ return x; }

/// Reinterpret bits of double as a uint.
/// @category conversion
__glsl_extension(GL_ARB_gpu_shader5)
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
void asuint(double value, out uint lowbits, out uint highbits)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asuint";
    case glsl: __intrinsic_asm "{ uvec2 v = unpackDouble2x32($0); $1 = v.x; $2 = v.y; }";
    case cpp:
    case cuda:
        __intrinsic_asm "$P_asuint($0, $1, $2)";
    case spirv:
        let uv = spirv_asm
        {
            result : $$uint2 = OpBitcast $value;
        };
        lowbits = uv.x;
        highbits = uv.y;
        return;
    }
}

// Reinterpret bits as a uint.

[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_4_0)]
uint asuint(float x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_asuint($0)";
    case cuda: __intrinsic_asm "$P_asuint($0)";
    case glsl: __intrinsic_asm "floatBitsToUint";
    case hlsl: __intrinsic_asm "asuint";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case wgsl: __intrinsic_asm "bitcast<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$uint result $x
    };
    }
}

[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_4_0)]
uint asuint(int x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_asuint($0)";
    case cuda: __intrinsic_asm "$P_asuint($0)";
    case glsl: __intrinsic_asm "uint($0)";
    case hlsl: __intrinsic_asm "asuint";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$uint result $x
    };
    case wgsl: __intrinsic_asm "bitcast<$TR>($0)";
    }
}

__generic<let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_4_0)]
vector<uint,N> asuint(vector<float,N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "floatBitsToUint";
    case hlsl: __intrinsic_asm "asuint";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$vector<uint,N> result $x
    };
    default:
        VECTOR_MAP_UNARY(uint, N, asuint, x);
    case wgsl: __intrinsic_asm "bitcast<$TR>($0)";
    }
}

__generic<let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_4_0)]
vector<uint, N> asuint(vector<int, N> x)
{
    if(N == 1)
        return vector<uint, N>(asuint(x[0]));
    __target_switch
    {
    case glsl: __intrinsic_asm "uvec$N0($0)";
    case hlsl: __intrinsic_asm "asuint";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$vector<uint, N> result $x
    };
    case wgsl: __intrinsic_asm "bitcast<$TR>($0)";
    default:
        VECTOR_MAP_UNARY(uint, N, asuint, x);
    }
}

__generic<let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
matrix<uint,N,M> asuint(matrix<float,N,M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asuint";
    default:
        MATRIX_MAP_UNARY(uint, N, M, asuint, x);
    }
}

__generic<let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
matrix<uint, N, M> asuint(matrix<int, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asuint";
    default:
        MATRIX_MAP_UNARY(uint, N, M, asuint, x);
    }
}

[__unsafeForceInlineEarly]
[__readNone]
uint asuint(uint x)
{ return x; }

__generic<let N : int>
[__unsafeForceInlineEarly]
[__readNone]
vector<uint,N> asuint(vector<uint,N> x)
{ return x; }

__generic<let N : int, let M : int>
[__unsafeForceInlineEarly]
[__readNone]
matrix<uint,N,M> asuint(matrix<uint,N,M> x)
{ return x; }


// 16-bit bitcast ops (HLSL SM 6.2)
//
// TODO: We need to map these to GLSL/SPIR-V
// operations that don't require an intermediate
// conversion to fp32.

// Identity cases:

/// Reinterpret bits as a float16 (HLSL SM 6.2).
/// @category conversion
[__unsafeForceInlineEarly][__readNone] float16_t asfloat16(float16_t value) { return value; }
[__unsafeForceInlineEarly][__readNone] vector<float16_t,N> asfloat16<let N : int>(vector<float16_t,N> value) { return value; }
[__unsafeForceInlineEarly][__readNone] matrix<float16_t,R,C> asfloat16<let R : int, let C : int>(matrix<float16_t,R,C> value) { return value; }

/// Reinterpret bits as a int16_t (HLSL SM 6.2).
/// @category conversion
[__unsafeForceInlineEarly][__readNone] int16_t asint16(int16_t value) { return value; }
[__unsafeForceInlineEarly][__readNone] vector<int16_t,N> asint16<let N : int>(vector<int16_t,N> value) { return value; }
[__unsafeForceInlineEarly][__readNone] matrix<int16_t,R,C> asint16<let R : int, let C : int>(matrix<int16_t,R,C> value) { return value; }

/// Reinterpret bits as a uint16_t (HLSL SM 6.2).
/// @category conversion
[__unsafeForceInlineEarly][__readNone] uint16_t asuint16(uint16_t value) { return value; }
[__unsafeForceInlineEarly][__readNone] vector<uint16_t,N> asuint16<let N : int>(vector<uint16_t,N> value) { return value; }
[__unsafeForceInlineEarly][__readNone] matrix<uint16_t,R,C> asuint16<let R : int, let C : int>(matrix<uint16_t,R,C> value) { return value; }

// Signed<->unsigned cases:

[__unsafeForceInlineEarly][__readNone] int16_t asint16(uint16_t value) { return value; }
[__unsafeForceInlineEarly][__readNone] vector<int16_t,N> asint16<let N : int>(vector<uint16_t,N> value) { return value; }
[__unsafeForceInlineEarly][__readNone] matrix<int16_t,R,C> asint16<let R : int, let C : int>(matrix<uint16_t,R,C> value) { return value; }

[__unsafeForceInlineEarly][__readNone] uint16_t asuint16(int16_t value) { return value; }
[__unsafeForceInlineEarly][__readNone] vector<uint16_t,N> asuint16<let N : int>(vector<int16_t,N> value) { return value; }
[__unsafeForceInlineEarly][__readNone] matrix<uint16_t,R,C> asuint16<let R : int, let C : int>(matrix<int16_t,R,C> value) { return value; }

// Float->unsigned cases:

[__readNone]
[require(cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
uint16_t asuint16(float16_t value)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "__half_as_ushort";
    case glsl: __intrinsic_asm "uint16_t(packHalf2x16(vec2($0, 0.0)))";
    case hlsl: __intrinsic_asm "asuint16";
    case spirv: return spirv_asm {
        OpBitcast $$uint16_t result $value
    };
    }
}

[__readNone]
[require(cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
vector<uint16_t,N> asuint16<let N : int>(vector<float16_t,N> value)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asuint16";
    case spirv: return spirv_asm {
        result:$$vector<uint16_t,N> = OpBitcast $value
    };
    default:
        VECTOR_MAP_UNARY(uint16_t, N, asuint16, value);
    }
}

[__readNone]
[require(cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
matrix<uint16_t,R,C> asuint16<let R : int, let C : int>(matrix<float16_t,R,C> value)
{ MATRIX_MAP_UNARY(uint16_t, R, C, asuint16, value); }

// Unsigned->float cases:

[__readNone]
[require(cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
float16_t asfloat16(uint16_t value)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "__ushort_as_half";
    case glsl: __intrinsic_asm "float16_t(unpackHalf2x16($0).x)";
    case hlsl: __intrinsic_asm "asfloat16";
    case spirv: return spirv_asm {
        OpBitcast $$float16_t result $value
    };
    }
}

[__readNone]
[require(cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
vector<float16_t,N> asfloat16<let N : int>(vector<uint16_t,N> value)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asfloat16";
    case spirv: return spirv_asm {
        result:$$vector<float16_t,N> = OpBitcast $value
    };
    default:
        VECTOR_MAP_UNARY(float16_t, N, asfloat16, value);
    }
}

[__readNone]
matrix<float16_t,R,C> asfloat16<let R : int, let C : int>(matrix<uint16_t,R,C> value)
{ MATRIX_MAP_UNARY(float16_t, R, C, asfloat16, value); }

// Float<->signed cases:

[__unsafeForceInlineEarly]
[__readNone]
[require(cuda_hlsl_metal_spirv, shader5_sm_5_0)]
int16_t asint16(float16_t value) 
{
    __target_switch
    {
    case cuda: __intrinsic_asm "__half_as_short";
    case hlsl: __intrinsic_asm "asint16";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$int16_t result $value
    };
    default: return asuint16(value);
    }
}

[__unsafeForceInlineEarly]
[__readNone] 
[require(cuda_hlsl_metal_spirv, shader5_sm_5_0)]
vector<int16_t,N> asint16<let N : int>(vector<float16_t,N> value) 
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asint16";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    default: return asuint16(value);
    }
}

[__unsafeForceInlineEarly]
[__readNone]
[require(cuda_hlsl_spirv, shader5_sm_5_0)]
matrix<int16_t,R,C> asint16<let R : int, let C : int>(matrix<float16_t,R,C> value) 
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asint16";
    default: return asuint16(value);
    }
}

[__readNone]
[__unsafeForceInlineEarly] 
[require(cuda_hlsl_metal_spirv, shader5_sm_5_0)]
float16_t asfloat16(int16_t value) 
{
    __target_switch
    {
    case cuda: __intrinsic_asm "__short_as_half";
    case hlsl: __intrinsic_asm "asfloat16";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$float16_t result $value
    };
    default: return asfloat16(asuint16(value));
    }
}

[__unsafeForceInlineEarly]
[__readNone]
[require(cuda_hlsl_metal_spirv, shader5_sm_5_0)]
vector<float16_t,N> asfloat16<let N : int>(vector<int16_t,N> value) 
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asfloat16";
    case metal: __intrinsic_asm "as_type<$TR>($0)";
    case spirv: return spirv_asm {
        OpBitcast $$vector<float16_t,N> result $value
    };
    default: return asfloat16(asuint16(value));
    }
}

[__unsafeForceInlineEarly]
[__readNone]
[require(cuda_hlsl_spirv, shader5_sm_5_0)]
matrix<float16_t,R,C> asfloat16<let R : int, let C : int>(matrix<int16_t,R,C> value) 
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "asfloat16";
    default: return asfloat16(asuint16(value));
    }
}

/// Arc tangent. Returns the angle whose tangent is the specified number.
/// @param x The tangent value.
/// @return The angle in radians, in the range of [-pi/2, pi/2].
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T atan(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_atan($0)";
    case cuda: __intrinsic_asm "$P_atan($0)";
    case glsl: __intrinsic_asm "atan";
    case hlsl: __intrinsic_asm "atan";
    case metal: __intrinsic_asm "atan";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Atan $x
    };
    case wgsl: __intrinsic_asm "atan";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> atan(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "atan";
    case hlsl: __intrinsic_asm "atan";
    case metal: __intrinsic_asm "atan";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Atan $x
    };
    case wgsl: __intrinsic_asm "atan";
    default:
        VECTOR_MAP_UNARY(T, N, atan, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> atan(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "atan";
    default:
        MATRIX_MAP_UNARY(T, N, M, atan, x);
    }
}

/// Arc tangent of y/x. Returns the angle whose tangent is the quotient of two specified numbers.
/// @param y The numerator.
/// @param x The denominator.
/// @return The angle in radians, in the range of [-pi, pi].
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T atan2(T y, T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_atan2($0, $1)";
    case cuda: __intrinsic_asm "$P_atan2($0, $1)";
    case glsl: __intrinsic_asm "atan($0,$1)";
    case hlsl: __intrinsic_asm "atan2";
    case metal: __intrinsic_asm "atan2";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Atan2 $y $x
    };
    case wgsl: __intrinsic_asm "atan2";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> atan2(vector<T, N> y, vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "atan($0,$1)";
    case hlsl: __intrinsic_asm "atan2";
    case metal: __intrinsic_asm "atan2";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Atan2 $y $x
    };
    case wgsl: __intrinsic_asm "atan2";
    default:
        VECTOR_MAP_BINARY(T, N, atan2, y, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> atan2(matrix<T,N,M> y, matrix<T,N,M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "atan2";
    default:
        MATRIX_MAP_BINARY(T, N, M, atan2, y, x);
    }
}

/// Hyperbolic arc tangent. Returns the hyperbolic arc tangent of the specified value.
/// @param x The value.
/// @return The hyperbolic arc tangent of the specified value.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T atanh(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_atanh($0)";
    case cuda: __intrinsic_asm "$P_atanh($0)";
    case glsl: __intrinsic_asm "atanh";
    case metal: __intrinsic_asm "atanh";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Atanh $x
    };
    case wgsl: __intrinsic_asm "atanh";
    default:
        return T(0.5) * log((T(1) + x) / (T(1) - x));
    }
}

__generic<T : __BuiltinFloatingPointType, let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> atanh(vector<T,N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "atanh";
    case metal: __intrinsic_asm "atanh";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,N> result glsl450 Atanh $x
    };
    case wgsl: __intrinsic_asm "atanh";
    default:
        VECTOR_MAP_UNARY(T, N, atanh, x);
    }
}

/// Ceiling. Returns the smallest integer that is greater than or equal to the specified value.
/// @param x The value.
/// @return The smallest integer that is greater than or equal to the specified value.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T ceil(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_ceil($0)";
    case cuda: __intrinsic_asm "$P_ceil($0)";
    case glsl: __intrinsic_asm "ceil";
    case hlsl: __intrinsic_asm "ceil";
    case metal: __intrinsic_asm "ceil";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Ceil $x
    };
    case wgsl: __intrinsic_asm "ceil";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> ceil(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "ceil";
    case hlsl: __intrinsic_asm "ceil";
    case metal: __intrinsic_asm "ceil";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Ceil $x
    };
    case wgsl: __intrinsic_asm "ceil";
    default:
        VECTOR_MAP_UNARY(T, N, ceil, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> ceil(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "ceil";
    default:
        MATRIX_MAP_UNARY(T, N, M, ceil, x);
    }
}

/// Copy-sign. Returns a value whose magnitude is from one operand and whose sign is from another operand.
/// @param x The value to use as the magnitude.
/// @param y The value to use as the sign.
/// @return A value whose magnitude is from `x` and whose sign is from `y`.
/// @category math
__generic<let N: int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv)]
vector<half,N> copysign_half(vector<half,N> x, vector<half,N> y)
{
    let ux = reinterpret<vector<uint16_t,N>>(x);
    let uy = reinterpret<vector<uint16_t,N>>(y);
    vector<uint16_t,N> signY = (uy & (uint16_t(1) << uint16_t(15)));
    vector<uint16_t,N> newX = (ux & ((uint16_t(1) << uint16_t(15)) - uint16_t(1))) + signY;
    return reinterpret<vector<half,N>>(newX);
}

/// Copy-sign. Returns a value whose magnitude is from one operand and whose sign is from another operand.
/// @param x The value to use as the magnitude.
/// @param y The value to use as the sign.
/// @return A value whose magnitude is from `x` and whose sign is from `y`.
/// @category math
__generic<let N: int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv)]
vector<float,N> copysign_float(vector<float,N> x, vector<float,N> y)
{
    let ux = reinterpret<vector<uint32_t,N>>(x);
    let uy = reinterpret<vector<uint32_t,N>>(y);
    vector<uint32_t,N> signY = (uy & (uint32_t(1) << uint32_t(31)));
    vector<uint32_t,N> newX = (ux & ((uint32_t(1) << uint32_t(31)) - uint32_t(1))) + signY;
    return reinterpret<vector<float,N>>(newX);
}

/// Copy-sign. Returns a value whose magnitude is from one operand and whose sign is from another operand.
/// @param x The value to use as the magnitude.
/// @param y The value to use as the sign.
/// @return A value whose magnitude is from `x` and whose sign is from `y`.
/// @category math
__generic<let N: int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv)]
vector<double,N> copysign_double(vector<double,N> x, vector<double,N> y)
{
    let ux = reinterpret<vector<uint64_t,N>>(x);
    let uy = reinterpret<vector<uint64_t,N>>(y);
    vector<uint64_t,N> signY = (uy & (uint64_t(1) << uint64_t(63)));
    vector<uint64_t,N> newX = (ux & ((uint64_t(1) << uint64_t(63)) - uint64_t(1))) + signY;
    return reinterpret<vector<double,N>>(newX);
}

__generic<T:__BuiltinFloatingPointType, U:__BuiltinFloatingPointType, let N : int>
__intrinsic_op($(kIROp_FloatCast))
vector<T,N> __real_cast(vector<U,N> val);

/// Copy-sign. Returns a value whose magnitude is from one operand and whose sign is from another operand.
/// @param x The value to use as the magnitude.
/// @param y The value to use as the sign.
/// @return A value whose magnitude is from x and whose sign is from y.
/// @category math
__generic<T : __BuiltinFloatingPointType, let N: int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv)]
vector<T,N> copysign(vector<T,N> x, vector<T,N> y)
{
    __target_switch
    {
    case metal: __intrinsic_asm "copysign";
    default:
    {
        // sign of -0.0 needs to be respected.
        if (T is half)
            return __real_cast<T>(copysign_half(
                    __real_cast<half>(x),
                    __real_cast<half>(y)));
        if (T is float)
            return __real_cast<T>(copysign_float(
                    __real_cast<float>(x),
                    __real_cast<float>(y)));
        return __real_cast<T>(copysign_double(
                __real_cast<double>(x),
                __real_cast<double>(y)));
    }
    }
}

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv)]
T copysign(T x, T y)
{
    __target_switch
    {
    case metal: __intrinsic_asm "copysign";
    default:
        return copysign(vector<T,1>(x), vector<T,1>(y))[0];
    }
}


// Check access status to tiled resource
[ForceInline]
[require(hlsl_spirv, sm_5_0)]
bool CheckAccessFullyMapped(uint status)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "CheckAccessFullyMapped";
    case spirv:
        return spirv_asm
        {
            OpCapability SparseResidency;
            result:$$bool = OpImageSparseTexelsResident $status;
        };
    }
}

/// Clamp. Returns the specified value clamped to the specified minimum and maximum bounds.
/// @param x The value to clamp.
/// @param minBound The minimum bound.
/// @param maxBound The maximum bound.
/// @return The clamped value.
/// @category math
__generic<T : __BuiltinIntegerType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T clamp(T x, T minBound, T maxBound)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "clamp";
    case glsl: __intrinsic_asm "clamp";
    case metal: __intrinsic_asm "clamp";
    case spirv:
        if (__isSignedInt<T>())
            return spirv_asm {
                result:$$T = OpExtInst glsl450 SClamp $x $minBound $maxBound
            };
        else
            return spirv_asm {
                result:$$T = OpExtInst glsl450 UClamp $x $minBound $maxBound
            };
    case wgsl: __intrinsic_asm "clamp";
    default:
        return min(max(x, minBound), maxBound);
    }
}

__generic<T : __BuiltinIntegerType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> clamp(vector<T, N> x, vector<T, N> minBound, vector<T, N> maxBound)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "clamp";
    case glsl: __intrinsic_asm "clamp";
    case metal: __intrinsic_asm "clamp";
    case spirv:
        if (__isSignedInt<T>())
            return spirv_asm {
                result:$$vector<T, N> = OpExtInst glsl450 SClamp $x $minBound $maxBound
            };
        else
            return spirv_asm {
                result:$$vector<T, N> = OpExtInst glsl450 UClamp $x $minBound $maxBound
            };
    case wgsl: __intrinsic_asm "clamp";
    default:
        return min(max(x, minBound), maxBound);
    }
}

__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> clamp(matrix<T,N,M> x, matrix<T,N,M> minBound, matrix<T,N,M> maxBound)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "clamp";
    default:
        return min(max(x, minBound), maxBound);
    }
}

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T clamp(T x, T minBound, T maxBound)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "clamp";
    case glsl: __intrinsic_asm "clamp";
    case metal: __intrinsic_asm "clamp";
    case spirv: return spirv_asm {
        result:$$T = OpExtInst glsl450 FClamp $x $minBound $maxBound
    };
    case wgsl: __intrinsic_asm "clamp";
    default:
        return min(max(x, minBound), maxBound);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> clamp(vector<T, N> x, vector<T, N> minBound, vector<T, N> maxBound)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "clamp";
    case glsl: __intrinsic_asm "clamp";
    case metal: __intrinsic_asm "clamp";
    case spirv: return spirv_asm {
        result:$$vector<T,N> = OpExtInst glsl450 FClamp $x $minBound $maxBound
    };
    case wgsl: __intrinsic_asm "clamp";
    default:
        return min(max(x, minBound), maxBound);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> clamp(matrix<T,N,M> x, matrix<T,N,M> minBound, matrix<T,N,M> maxBound)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "clamp";
    default:
        return min(max(x, minBound), maxBound);
    }
}

/// Clip (discard) fragment conditionally
__generic<T : __BuiltinFloatingPointType>
[require(cpp_cuda_glsl_hlsl_spirv, fragment)]
void clip(T x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "clip";
    default:
        if(x < T(0)) discard;
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[require(cpp_cuda_glsl_hlsl_spirv, fragment)]
void clip(vector<T,N> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "clip";
    default:
        if(any(x < T(0))) discard;
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[require(cpp_cuda_glsl_hlsl_spirv, fragment)]
void clip(matrix<T,N,M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "clip";
    default:
        if(any(x < T(0))) discard;
    }
}

/// Cosine. Returns the cosine of the specified angle.
/// @param x The angle in radians.
/// @return The cosine of the specified angle.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T cos(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_cos($0)";
    case cuda: __intrinsic_asm "$P_cos($0)";
    case glsl: __intrinsic_asm "cos";
    case hlsl: __intrinsic_asm "cos";
    case metal: __intrinsic_asm "cos";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Cos $x
    };
    case wgsl: __intrinsic_asm "cos";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> cos(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "cos";
    case hlsl: __intrinsic_asm "cos";
    case metal: __intrinsic_asm "cos";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Cos $x
    };
    case wgsl: __intrinsic_asm "cos";
    default:
        VECTOR_MAP_UNARY(T,N, cos, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> cos(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "cos";
    default:
        MATRIX_MAP_UNARY(T, N, M, cos, x);
    }
}

/// Hyperbolic cosine. Returns the hyperbolic cosine of the specified value.
/// @param x The specified value.
/// @return The hyperbolic cosine of the specified value.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T cosh(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_cosh($0)";
    case cuda: __intrinsic_asm "$P_cosh($0)";
    case glsl: __intrinsic_asm "cosh";
    case hlsl: __intrinsic_asm "cosh";
    case metal: __intrinsic_asm "cosh";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Cosh $x
    };
    case wgsl: __intrinsic_asm "cosh";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> cosh(vector<T,N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "cosh";
    case hlsl: __intrinsic_asm "cosh";
    case metal: __intrinsic_asm "cosh";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,N> result glsl450 Cosh $x
    };
    case wgsl: __intrinsic_asm "cosh";
    default:
        VECTOR_MAP_UNARY(T,N, cosh, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> cosh(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "cosh";
    default:
        MATRIX_MAP_UNARY(T, N, M, cosh, x);
    }
}

/// Compute the cosine of pi times the input.
/// @param x The input value.
/// @return The cosine of pi times the input.
/// @remarks This function is equivalent to `cos(PI * x)`. On Metal, this function is implemented using the `cospi` intrinsic.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T cospi(T x)
{
    __target_switch
    {
    case metal: __intrinsic_asm "cospi";
    default:
        return cos(T.getPi() * x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N: int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> cospi(vector<T,N> x)
{
    __target_switch
    {
    case metal: __intrinsic_asm "cospi";
    default:
        return cos(T.getPi() * x);
    }
}

// emulate 64-bit countbits when not natively supported.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
internal uint __emulatedCountbits64(uint64_t value)
{
    uint2 value_uint2 = bit_cast<uint2>(value);
    uint2 counted_bits_uint2 = countbits(value_uint2);
    return counted_bits_uint2.x + counted_bits_uint2.y;
}

/// Population count.
/// Counts the number of set bits in the binary representation of a value.
/// @param value The value to count bits in.
/// @return The number of bits in the binary representation of `value` that are set to one.
/// @remarks For SPIR-V, this function maps to `OpBitCount`.
/// @category bitops
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
__generic<T : __BuiltinIntegerType>
uint countbits(T value)
{
    // Emulate 8-bit support
    // 8-bit support is not currently supported anywhere natively
    if (T is int8_t || T  is uint8_t)
    {
        return countbits(__intCast<uint32_t>(value));
    }

    __target_switch
    {
    case hlsl:
        // 64-bit support dependent on SM6.0 and dxil
        // 16-bit support dependent on SM6.2 and dxil
        __intrinsic_asm "countbits";
    case glsl:
        // 64-bit support dependent on GL_ARB_gpu_shader_int64
        // 16-bit support dependent on GL_EXT_shader_16bit_storage
        __intrinsic_asm "bitCount";
    case metal:
        if (T is int64_t || T  is uint64_t)
        {
            return __emulatedCountbits64(__intCast<uint64_t>(value));
        }
        else if (T is int16_t || T  is uint16_t)
        {
            // emulate 16-bit
            return countbits(__intCast<uint32_t>(value));
        }
        else
        {
            __intrinsic_asm "popcount";
        }
    case cuda:
    case cpp:
        __intrinsic_asm "$P_countbits($0)";
    case spirv:
        if(T is int64_t || T  is uint64_t)
        {
            return __emulatedCountbits64(__intCast<uint64_t>(value));
        }
        else if (T is int16_t || T  is uint16_t)
        {
            // emulate 16-bit
            return countbits(__intCast<uint32_t>(value));
        }
        else
        {
            // OpBitCount only supports 32-bit
            return spirv_asm {OpBitCount $$uint result $value};
        }
    case wgsl:
            // wgsl only supports 32-bit integers
            if (T is int32_t)
            {
                // wgsl countOneBits returns the same type as the
                // one it was given. Cast signed ints to unsigned
                // so we can provide the correct return value.
                return countbits(__intCast<uint32_t>(value));
            }
            __intrinsic_asm "countOneBits";
    }
}

__generic<T : __BuiltinIntegerType, let N : int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
vector<uint, N> countbits(vector<T, N> value)
{
    // Emulate 8-bit support
    // 8-bit support is not currently supported anywhere natively
    if (T is int8_t || T  is uint8_t)
    {
        VECTOR_MAP_UNARY(uint, N, countbits, value);
    }

    __target_switch
    {
    case hlsl:
        __intrinsic_asm "countbits";
    case glsl:
        __intrinsic_asm "bitCount";
    case metal:
        if(T is int64_t || T  is uint64_t || T is int16_t || T  is uint16_t)
        {
            // Emulate 64-bit and 16-bit
            VECTOR_MAP_UNARY(uint, N, countbits, value);
        }
        else
        {
            __intrinsic_asm "popcount";
        }
    case spirv:
        if(T is int64_t || T  is uint64_t || T is int16_t || T  is uint16_t)
        {
            // Emulate 64-bit and 16-bit
            VECTOR_MAP_UNARY(uint, N, countbits, value);
        }
        else
        {
            return spirv_asm {OpBitCount $$vector<uint, N> result $value};
        }
    case wgsl:
        // wgsl only supports 32-bit integers
        if (T is int32_t)
        {
            vector<uint32_t, N> ret;
            for (int i = 0; i < N; i++)
            {
                ret[i] = countbits(__intCast<uint32_t>(value[i]));
            }
            return ret;
        }
            __intrinsic_asm "countOneBits";
    default:
        VECTOR_MAP_UNARY(uint, N, countbits, value);
    }
}

/// Cross product. Returns the cross product of two 3D vectors.
/// @param left The first vector.
/// @param right The second vector.
/// @return The cross product of `left` and `right`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,3> cross(vector<T,3> left, vector<T,3> right)
{
    // TODO: SPIRV does not support integer vectors.
    __target_switch
    {
    case glsl: __intrinsic_asm "cross";
    case hlsl: __intrinsic_asm "cross";
    case metal: __intrinsic_asm "cross";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,3> result glsl450 Cross $left $right
    };
    case wgsl: __intrinsic_asm "cross";
    default:
        return vector<T,3>(
            left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x);
    }
}

__generic<T : __BuiltinIntegerType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, 3> cross(vector<T, 3> left, vector<T, 3> right)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "cross";
    case hlsl: __intrinsic_asm "cross";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, 3> result glsl450 Cross $left $right
    };
    case wgsl: __intrinsic_asm "cross";
    default:
        return vector<T, 3>(
            left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x);
    }
}

// Convert encoded color
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
int4 D3DCOLORtoUBYTE4(float4 color)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "D3DCOLORtoUBYTE4";
    case wgsl: __intrinsic_asm "bitcast<vec4i>(pack4x8unorm($0)).zyxw";
    default:
        let scaled = color.zyxw * 255.001999f;
        return int4(scaled);
    }
}

// Partial-difference derivatives
${{{{
const char* diffDimensions[2] = {"x", "y"};
for (auto xOrY : diffDimensions) {
}}}}
/// Take the partial derivative of `p` with respect to $(xOrY) in screen space.
/// @param p The value to take partial derivative for.
/// @return The partial derivative of `p`.
/// @remarks For SPIR-V, this function maps to `OpDPd$(xOrY)`.
/// @category derivative Derivative functions
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, fragmentprocessing)]
T dd$(xOrY)(T p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl:
    case cpp:
    case cuda:
        __intrinsic_asm "dd$(xOrY)";
    case glsl:
        __intrinsic_asm "dFd$(xOrY)";
    case metal:
        __intrinsic_asm "dfd$(xOrY)";
    case spirv:
        return spirv_asm {OpDPd$(xOrY) $$T result $p};
    case wgsl:
        __intrinsic_asm "dpd$(xOrY)";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, fragmentprocessing)]
vector<T, N> dd$(xOrY)(vector<T, N> p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl:
    case cpp:
    case cuda:
        __intrinsic_asm "dd$(xOrY)";
    case glsl:
        __intrinsic_asm "dFd$(xOrY)";
    case metal:
        __intrinsic_asm "dfd$(xOrY)";
    case spirv:
        return spirv_asm {OpDPd$(xOrY) $$vector<T, N> result $p};
    case wgsl:
        __intrinsic_asm "dpd$(xOrY)";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, fragmentprocessing)]
matrix<T, N, M> dd$(xOrY)(matrix<T, N, M> p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "dd$(xOrY)";
    default:
        MATRIX_MAP_UNARY(T, N, M, dd$(xOrY), p);
    }
}

/// Take the coarse partial derivative of `p` with respect to $(xOrY) in screen space.
/// @param p The value to take partial derivative for.
/// @return The partial derivative of `p`.
/// @remarks For SPIR-V, this function maps to `OpDPd$(xOrY)Coarse`.
/// @category derivative
__generic<T : __BuiltinFloatingPointType>
__glsl_extension(GL_ARB_derivative_control)
[__readNone]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
T dd$(xOrY)_coarse(T p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl: __intrinsic_asm "dd$(xOrY)_coarse";
    case glsl: __intrinsic_asm "dFd$(xOrY)Coarse";
    case spirv: return spirv_asm {OpCapability DerivativeControl; result:$$T = OpDPd$(xOrY)Coarse $p};
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
__glsl_extension(GL_ARB_derivative_control)
[__readNone]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
vector<T, N> dd$(xOrY)_coarse(vector<T, N> p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl: __intrinsic_asm "dd$(xOrY)_coarse";
    case glsl: __intrinsic_asm "dFd$(xOrY)Coarse";
    case spirv: return spirv_asm {OpCapability DerivativeControl; result:$$vector<T,N> = OpDPd$(xOrY)Coarse $p};
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
matrix<T, N, M> dd$(xOrY)_coarse(matrix<T, N, M> p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "dd$(xOrY)_coarse";
    default:
        MATRIX_MAP_UNARY(T, N, M, dd$(xOrY)_coarse, p);
    }
}

/// Take the fine partial derivative of `p` with respect to $(xOrY) in screen space.
/// @param p The value to take partial derivative for.
/// @return The partial derivative of `p`.
/// @remarks For SPIR-V, this function maps to `OpDPd$(xOrY)Fine`.
/// @category derivative
__generic<T : __BuiltinFloatingPointType>
__glsl_extension(GL_ARB_derivative_control)
[__readNone]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
T dd$(xOrY)_fine(T p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl: __intrinsic_asm "dd$(xOrY)_fine";
    case glsl: __intrinsic_asm "dFd$(xOrY)Fine";
    case spirv: return spirv_asm {OpCapability DerivativeControl; result:$$T = OpDPd$(xOrY)Fine $p};
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
__glsl_extension(GL_ARB_derivative_control)
[__readNone]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
vector<T, N> dd$(xOrY)_fine(vector<T, N> p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl: __intrinsic_asm "dd$(xOrY)_fine";
    case glsl: __intrinsic_asm "dFd$(xOrY)Fine";
    case spirv: return spirv_asm {OpCapability DerivativeControl; result:$$vector<T,N> = OpDPd$(xOrY)Fine $p};
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
matrix<T, N, M> dd$(xOrY)_fine(matrix<T, N, M> p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "dd$(xOrY)_fine";
    default:
        MATRIX_MAP_UNARY(T, N, M, dd$(xOrY)_fine, p);
    }
}

${{{{
} // for (xOrY)
}}}}


/// Convert radians to degrees.
/// @param x The angle in radians.
/// @return The angle in degrees.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl)]
T degrees(T x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "degrees";
    case hlsl: __intrinsic_asm "degrees";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Degrees $x
    };
    case wgsl: __intrinsic_asm "degrees";
    default:
        return x * (T(180) / T.getPi());
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl)]
vector<T, N> degrees(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "degrees";
    case hlsl: __intrinsic_asm "degrees";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Degrees $x
    };
    case wgsl: __intrinsic_asm "degrees";
    default:
        VECTOR_MAP_UNARY(T, N, degrees, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl)]
matrix<T, N, M> degrees(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "degrees";
    default:
        MATRIX_MAP_UNARY(T, N, M, degrees, x);
    }
}

/// Compute matrix determinant.
/// @param m The matrix.
/// @return The determinant of the matrix.
/// @category math
__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[PreferCheckpoint]
[require(glsl_hlsl_metal_spirv_wgsl)]
T determinant(matrix<T,N,N> m)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "determinant";
    case hlsl: __intrinsic_asm "determinant";
    case metal: __intrinsic_asm "determinant";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Determinant $m
    };
    case wgsl: __intrinsic_asm "determinant";
    }
}

/// Barrier for device memory.
/// @category barrier
__glsl_extension(GL_KHR_memory_scope_semantics)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, memorybarrier)]
void DeviceMemoryBarrier()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "DeviceMemoryBarrier";
    case glsl: __intrinsic_asm "memoryBarrier(gl_ScopeDevice, (gl_StorageSemanticsImage|gl_StorageSemanticsBuffer), gl_SemanticsAcquireRelease)";
    case cuda: __intrinsic_asm "__threadfence()";
    case metal: __intrinsic_asm "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock)";
    case spirv: spirv_asm
        {
            OpMemoryBarrier Device AcquireRelease|UniformMemory|ImageMemory;
        };
    case wgsl: __intrinsic_asm "storageBarrier(); textureBarrier(); workgroupBarrier();";
    }
}

/// Barrier for device memory with group synchronization.
/// @category barrier
__glsl_extension(GL_KHR_memory_scope_semantics)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, memorybarrier)]
void DeviceMemoryBarrierWithGroupSync()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "DeviceMemoryBarrierWithGroupSync";
    case glsl: __intrinsic_asm "controlBarrier(gl_ScopeWorkgroup, gl_ScopeDevice, (gl_StorageSemanticsImage|gl_StorageSemanticsBuffer), gl_SemanticsAcquireRelease)";
    case cuda: __intrinsic_asm "__syncthreads()";
    case metal: __intrinsic_asm "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock)";
    case spirv: spirv_asm
        {
            OpControlBarrier Workgroup Device AcquireRelease|UniformMemory|ImageMemory;
        };
    case wgsl: __intrinsic_asm "storageBarrier(); textureBarrier(); workgroupBarrier();";
    }
}

/// Vector distance. Returns the distance between two points.
/// @param x The first point.
/// @param y The second point.
/// @return The distance between `x` and `y`.
/// @remarks This function is equivalent to `length(x - y)`. When `x` and `y` are scalars, this function is equivalent to `abs(x - y)`.
/// @category math
__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T distance(vector<T, N> x, vector<T, N> y)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "distance";
    case hlsl: __intrinsic_asm "distance";
    case metal: __intrinsic_asm "distance";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Distance $x $y
    };
    case wgsl: __intrinsic_asm "distance";
    default:
        return length(x - y);
    }
}

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T distance(T x, T y)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "distance";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Distance $x $y
    };
    case wgsl: __intrinsic_asm "distance";
    default:
        return length(x - y);
    }
}

/// Computes `max(0, x-y)`.
/// @param x The first value.
/// @param y The second value.
/// @return The result of `max(0, x-y)`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
T fdim(T x, T y)
{
    __target_switch
    {
    case metal: __intrinsic_asm "fdim";
    default:
        return max(T(0), x - y);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
vector<T,N> fdim(vector<T,N> x, vector<T,N> y)
{
    __target_switch
    {
    case metal: __intrinsic_asm "fdim";
    default:
        return max(T(0), x - y);
    }
}

/// Divide values.
/// @param x The dividend.
/// @param y The divisor.
/// @return The result of dividing `x` by `y`, element-wise for vector types.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv)]
T divide(T x, T y)
{
    __target_switch
    {
    case metal: __intrinsic_asm "divide";
    default:
        return x / y;
    }
}

__generic<T : __BuiltinFloatingPointType, let N: int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv)]
vector<T,N> divide(vector<T,N> x, vector<T,N> y)
{
    __target_switch
    {
    case metal: __intrinsic_asm "divide";
    default:
        return x / y;
    }
}

/// Vector dot product. Returns the dot product of two vectors.
/// @param x The first vector.
/// @param y The second vector.
/// @return The dot product of `x` and `y`.
/// @remarks When `x` and `y` are scalars, this function is equivalent to `x * y`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T dot(T x, T y)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "dot";
    case hlsl: __intrinsic_asm "dot";
    case wgsl: __intrinsic_asm "dot";
    default:
        return x * y;
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T dot(vector<T, N> x, vector<T, N> y)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "dot";
    case hlsl: __intrinsic_asm "dot";
    case metal: __intrinsic_asm "dot";
    case spirv: return spirv_asm {
        OpDot $$T result $x $y
    };
    case wgsl: __intrinsic_asm "dot";
    default:
        T result = T(0);
        for(int i = 0; i < N; ++i)
            result += x[i] * y[i];
        return result;
    }
}

__generic<T : __BuiltinIntegerType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T dot(vector<T, N> x, vector<T, N> y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "dot";
    case wgsl: __intrinsic_asm "dot";
    case spirv:
    {
        spirv_asm
        {
            OpCapability DotProduct;
            OpCapability DotProductInputAll;
            OpExtension "SPV_KHR_integer_dot_product";
        };

        if (__isSignedInt<T>())
        {
            return spirv_asm
            {
                result:$$T = OpSDot $x $y;
            };
        }
        else
        {
            return spirv_asm
            {
                result:$$T = OpUDot $x $y;
            };
        }
    }
    default:
        T result = T(0);
        for(int i = 0; i < N; ++i)
            result += x[i] * y[i];
        return result;
    }
}

/// Helper for computing distance terms for lighting (obsolete).
/// Use the subtraction operator '-' instead.
/// @deprecated
/// @category math
__generic<T : __BuiltinFloatingPointType> vector<T,4> dst(vector<T,4> x, vector<T,4> y);

// Given a RWByteAddressBuffer allow it to be interpreted as a RWStructuredBuffer
__intrinsic_op($(kIROp_GetEquivalentStructuredBuffer))
RWStructuredBuffer<T> __getEquivalentStructuredBuffer<T>(RWByteAddressBuffer b);

__intrinsic_op($(kIROp_GetEquivalentStructuredBuffer))
StructuredBuffer<T> __getEquivalentStructuredBuffer<T>(ByteAddressBuffer b);

__intrinsic_op($(kIROp_GetEquivalentStructuredBuffer))
RasterizerOrderedStructuredBuffer<T> __getEquivalentStructuredBuffer<T>(RasterizerOrderedByteAddressBuffer b);

// Error message

// void errorf( string format, ... );

// Attribute evaluation

T __EvaluateAttributeAtCentroid<T>(__constref T x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "EvaluateAttributeAtCentroid";
    case glsl: __intrinsic_asm "interpolateAtCentroid";
    }
}

// TODO: The matrix cases of these functions won't actuall work
// when compiled to GLSL, since they only support scalar/vector

// TODO: Should these be constrains to `__BuiltinFloatingPointType`?
// TODO: SPIRV-direct does not support non-floating-point types.

/// Interpolates vertex attribute at centroid position.
/// @param x The vertex attribute to interpolate.
/// @return The interpolated attribute value.
/// @remarks `x` must be a direct reference to a fragment shader varying input.
/// @category interpolation Vertex Interpolation Functions
__generic<T : __BuiltinArithmeticType>
[__readNone]
[__unsafeForceInlineEarly]
[require(glsl_hlsl_spirv, fragmentprocessing)]
T EvaluateAttributeAtCentroid(__constref T x)
{
    __target_switch
    {
    case hlsl:
    case glsl:
        return __EvaluateAttributeAtCentroid(__ResolveVaryingInputRef(x));
    case spirv: return spirv_asm {
        OpCapability InterpolationFunction;
        OpExtInst $$T result glsl450 InterpolateAtCentroid $__ResolveVaryingInputRef(x)
    };
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
[__readNone]
[__unsafeForceInlineEarly]
[require(glsl_hlsl_spirv, fragmentprocessing)]
vector<T,N> EvaluateAttributeAtCentroid(__constref vector<T,N> x)
{
    __target_switch
    {
    case hlsl:
    case glsl:
        return __EvaluateAttributeAtCentroid(__ResolveVaryingInputRef(x));
    case spirv: return spirv_asm {
        OpCapability InterpolationFunction;
        OpExtInst $$vector<T,N> result glsl450 InterpolateAtCentroid $__ResolveVaryingInputRef(x)
    };
    }
}

__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[__readNone]
[__unsafeForceInlineEarly]
[require(glsl_hlsl_spirv, fragmentprocessing)]
matrix<T,N,M> EvaluateAttributeAtCentroid(__constref matrix<T,N,M> x)
{
    __target_switch
    {
    case hlsl:
    case glsl:
        return __EvaluateAttributeAtCentroid(__ResolveVaryingInputRef(x));
    default:
        MATRIX_MAP_UNARY(T, N, M, EvaluateAttributeAtCentroid, x);
    }
}

T __EvaluateAttributeAtSample<T>(__constref T x, uint sampleIndex)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "EvaluateAttributeAtSample";
    case glsl: __intrinsic_asm "interpolateAtSample";
    }
}

/// Interpolates vertex attribute at the current fragment sample position.
/// @param x The vertex attribute to interpolate.
/// @return The interpolated attribute value.
/// @remarks `x` must be a direct reference to a fragment shader varying input.
/// @category interpolation Vertex Interpolation Functions
__generic<T : __BuiltinArithmeticType>
[__readNone]
[__unsafeForceInlineEarly]
[require(glsl_hlsl_spirv, fragmentprocessing)]
T EvaluateAttributeAtSample(__constref T x, uint sampleindex)
{
    __target_switch
    {
    case hlsl:
    case glsl:
        return __EvaluateAttributeAtSample(__ResolveVaryingInputRef(x), sampleindex);
    case spirv: return spirv_asm {
        OpCapability InterpolationFunction;
        OpExtInst $$T result glsl450 InterpolateAtSample $__ResolveVaryingInputRef(x) $sampleindex
    };
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
[__readNone]
[__unsafeForceInlineEarly]
[require(glsl_hlsl_spirv, fragmentprocessing)]
vector<T,N> EvaluateAttributeAtSample(__constref vector<T,N> x, uint sampleindex)
{
    __target_switch
    {
    case hlsl:
    case glsl:
        return __EvaluateAttributeAtSample(__ResolveVaryingInputRef(x), sampleindex);
    case spirv: return spirv_asm {
        OpCapability InterpolationFunction;
        OpExtInst $$vector<T,N> result glsl450 InterpolateAtSample $__ResolveVaryingInputRef(x) $sampleindex
    };
    }
}

__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[__readNone]
[__unsafeForceInlineEarly]
[require(glsl_hlsl_spirv, fragmentprocessing)]
matrix<T,N,M> EvaluateAttributeAtSample(__constref matrix<T,N,M> x, uint sampleindex)
{
    __target_switch
    {
    case hlsl:
    case glsl:
        return __EvaluateAttributeAtSample(__ResolveVaryingInputRef(x), sampleindex);
    default:
        matrix<T,N,M> result;
        for(int i = 0; i < N; ++i)
        {
            result[i] = EvaluateAttributeAtSample(x[i], sampleindex);
        }
        return result;
    }
}

T __EvaluateAttributeSnapped<T>(__constref T x, int2 offset)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "EvaluateAttributeSnapped";
    case glsl: __intrinsic_asm "EvaluateAttributeSnapped";
    }
}

/// Interpolates vertex attribute at the specified subpixel offset.
/// @param x The vertex attribute to interpolate.
/// @param offset The subpixel offset. Each component is a 4-bit signed integer in range [-8, 7].
/// @return The interpolated attribute value.
/// @remarks `x` must be a direct reference to a fragment shader varying input.
///
/// The valid values of each component of `offset` are:
///
/// - 1000 = -0.5f (-8 / 16)
/// - 1001 = -0.4375f (-7 / 16)
/// - 1010 = -0.375f (-6 / 16)
/// - 1011 = -0.3125f (-5 / 16)
/// - 1100 = -0.25f (-4 / 16)
/// - 1101 = -0.1875f (-3 / 16)
/// - 1110 = -0.125f (-2 / 16)
/// - 1111 = -0.0625f (-1 / 16)
/// - 0000 = 0.0f ( 0 / 16)
/// - 0001 = 0.0625f ( 1 / 16)
/// - 0010 = 0.125f ( 2 / 16)
/// - 0011 = 0.1875f ( 3 / 16)
/// - 0100 = 0.25f ( 4 / 16)
/// - 0101 = 0.3125f ( 5 / 16)
/// - 0110 = 0.375f ( 6 / 16)
/// - 0111 = 0.4375f ( 7 / 16)
/// @category interpolation Vertex Interpolation Functions
__generic<T : __BuiltinArithmeticType>
[__readNone]
[__unsafeForceInlineEarly]
[require(glsl_hlsl_spirv, fragmentprocessing)]
T EvaluateAttributeSnapped(__constref T x, int2 offset)
{
    __target_switch
    {
    case hlsl: 
    case glsl:
        return __EvaluateAttributeSnapped(__ResolveVaryingInputRef(x), offset);
    case spirv:
    {
        const float2 tmp = float2(16.f, 16.f);
        return spirv_asm {
            OpCapability InterpolationFunction;
            %foffset:$$float2 = OpConvertSToF $offset;
            %offsetdiv16:$$float2 = OpFDiv %foffset $tmp;
            result:$$T = OpExtInst glsl450 InterpolateAtOffset $__ResolveVaryingInputRef(x) %offsetdiv16
        };
    }
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
[__readNone]
[__unsafeForceInlineEarly]
[require(glsl_hlsl_spirv, fragmentprocessing)]
vector<T,N> EvaluateAttributeSnapped(__constref vector<T,N> x, int2 offset)
{
    __target_switch
    {
    case hlsl: 
    case glsl:
        return __EvaluateAttributeSnapped(__ResolveVaryingInputRef(x), offset);
    case spirv:
    {
        const float2 tmp = float2(16.f, 16.f);
        return spirv_asm {
            OpCapability InterpolationFunction;
            %foffset:$$float2 = OpConvertSToF $offset;
            %offsetdiv16:$$float2 = OpFDiv %foffset $tmp;
            result:$$vector<T,N> = OpExtInst glsl450 InterpolateAtOffset $__ResolveVaryingInputRef(x) %offsetdiv16
        };
    }
    }
}

__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[__readNone]
[__unsafeForceInlineEarly]
[require(glsl_hlsl_spirv, fragmentprocessing)]
matrix<T,N,M> EvaluateAttributeSnapped(__constref matrix<T,N,M> x, int2 offset)
{
    __target_switch
    {
    case hlsl: 
    case glsl:
        return __EvaluateAttributeSnapped(__ResolveVaryingInputRef(x), offset);
    default:
        matrix<T,N,M> result;
        for(int i = 0; i < N; ++i)
        {
            result[i] = EvaluateAttributeSnapped(x[i], offset);
        }
        return result;
    }
}

/// Computes base-e exponent.
/// @param x The input value.
/// @return The base-e exponent of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T exp(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_exp($0)";
    case cuda: __intrinsic_asm "$P_exp($0)";
    case glsl: __intrinsic_asm "exp";
    case hlsl: __intrinsic_asm "exp";
    case metal: __intrinsic_asm "exp";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Exp $x
    };
    case wgsl: __intrinsic_asm "exp";
    }
}

/// @category math
__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> exp(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "exp";
    case hlsl: __intrinsic_asm "exp";
    case metal: __intrinsic_asm "exp";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Exp $x
    };
    case wgsl: __intrinsic_asm "exp";
    default:
        VECTOR_MAP_UNARY(T, N, exp, x);
    }
}

/// @category math
__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> exp(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "exp";
    default:
        MATRIX_MAP_UNARY(T, N, M, exp, x);
    }
}

/// Computes base-2 exponent.
/// @param x The input value.
/// @return The base-2 exponent of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T exp2(T x)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "exp2($0)";
    case spirv:
        if (__isHalf<T>())
        {
            return spirv_asm { OpExtInst $$T result glsl450 Exp2 $x };
        }
        else
        {
            float xf = __realCast<float>(x);
            return T(spirv_asm {
                 result:$$float = OpExtInst glsl450 Exp2 $xf
            });
        }
    case hlsl:
        __intrinsic_asm "exp2($0)";
    case metal: __intrinsic_asm "exp2";
    case cpp: 
        __intrinsic_asm "$P_exp2($0)";
    case cuda: 
        __intrinsic_asm "$P_exp2($0)";
    case wgsl:
        __intrinsic_asm "exp2";
    }
        
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> exp2(vector<T,N> x)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "exp2($0)";
    case hlsl: __intrinsic_asm "exp2";
    case metal: __intrinsic_asm "exp2";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,N> result glsl450 Exp2 $x
    };
    case wgsl: __intrinsic_asm "exp2";
    default:
        VECTOR_MAP_UNARY(T, N, exp2, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> exp2(matrix<T,N,M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "exp2";
    default:
        MATRIX_MAP_UNARY(T, N, M, exp2, x);
    }
}

/// Computes base-10 exponent.
/// @param x The input value.
/// @return The base-10 exponent of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T exp10(T x)
{
    __target_switch
    {
    case metal: __intrinsic_asm "exp10";
    default:
        const T ln10 = T(2.302585092994045901); // ln(10)
        return exp(x * ln10);
    }
}

__generic<T : __BuiltinFloatingPointType, let N: int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> exp10(vector<T,N> x)
{
    __target_switch
    {
    case metal: __intrinsic_asm "exp10";
    default:
        const T ln10 = T(2.30258509299); // ln(10)
        return exp(x * ln10);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> exp10(matrix<T,N,M> x)
{
    __target_switch
    {
    default:
        MATRIX_MAP_UNARY(T, N, M, exp10, x);
    }
}


/// Convert 16-bit float stored in low bits of integer
/// @category conversion Conversion functions
__glsl_version(420)
__cuda_sm_version(6.0)
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
float f16tof32(uint value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "unpackHalf2x16($0).x";
    case hlsl: __intrinsic_asm "f16tof32($0)";
    case cuda: __intrinsic_asm "__half2float(__ushort_as_half($0))";
    case cpp: __intrinsic_asm "f16tof32($0)";
    case metal: __intrinsic_asm "as_type<half>((ushort)($0))";
    case spirv:
    {
        return spirv_asm {
            %lowBits = OpUConvert $$uint16_t $value;
            %half = OpBitcast $$half %lowBits;
            result:$$float = OpFConvert %half
        };
    }
    case wgsl: __intrinsic_asm "unpack2x16float($0).x";
    }
}

__generic<let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
vector<float, N> f16tof32(vector<uint, N> value)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "f16tof32";
    case spirv:
    {
        return spirv_asm {
            %lowBits = OpUConvert $$vector<uint16_t,N> $value;
            %half = OpBitcast $$vector<half,N> %lowBits;
            result:$$vector<float,N> = OpFConvert %half
        };
    }
    default:
        VECTOR_MAP_UNARY(float, N, f16tof32, value);
    }
}

/// Convert to 16-bit float stored in low bits of integer.
/// @category conversion
__glsl_version(420)
__cuda_sm_version(6.0)
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
uint f32tof16(float value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "packHalf2x16(vec2($0,0.0))";
    case hlsl: __intrinsic_asm "f32tof16($0)";
    case cuda: __intrinsic_asm "__half_as_ushort(__float2half($0))";
    case cpp: __intrinsic_asm "f32tof16($0)";
    case metal: __intrinsic_asm "as_type<ushort>((half)($0))";
    case spirv:
    {
        return spirv_asm {
            %half = OpFConvert $$half $value;
            %lowBits = OpBitcast $$uint16_t %half;
            result:$$uint = OpUConvert %lowBits
        };
    }
    case wgsl: __intrinsic_asm "pack2x16float(vec2f($0,0.0))";
    }
}

__generic<let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
vector<uint, N> f32tof16(vector<float, N> value)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "f32tof16";
    case spirv:
    {
        return spirv_asm {
            %half = OpFConvert $$vector<half,N> $value;
            %lowBits = OpBitcast $$vector<uint16_t,N> %half;
            result:$$vector<uint,N> = OpUConvert %lowBits
        };
    }
    default:
        VECTOR_MAP_UNARY(uint, N, f32tof16, value);
    }
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// The following is Slang specific and NOT part of standard HLSL
// It's not clear what happens with float16 time in HLSL -> can the float16 coerce to uint for example? If so that would
// give the wrong result

__glsl_version(420)
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
float f16tof32(float16_t value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "unpackHalf2x16($0).x";
    case hlsl: __intrinsic_asm "f16tof32($0)";
    case cuda: __intrinsic_asm "__half2float($0)";
    case cpp: __intrinsic_asm "f16tof32($0)";
    case metal: __intrinsic_asm "float($0)";
    case spirv:
    {
        return spirv_asm {
            result:$$float = OpFConvert $value
        };
    }
    case wgsl: __intrinsic_asm "f32($0)";
    }
}

__generic<let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
vector<float, N> f16tof32(vector<float16_t, N> value)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "__half2float";
    case hlsl: __intrinsic_asm "f16tof32";
    case metal: __intrinsic_asm "$TR($0)";
    case spirv: return spirv_asm {
        OpFConvert $$vector<float, N> result $value
    };
    default:
        VECTOR_MAP_UNARY(float, N, f16tof32, value);
    }
}

/// Convert to float16_t.
/// @category conversion
__glsl_version(420)
[__readNone]
[require(cuda_glsl_metal_spirv_wgsl, shader5_sm_5_0)]
float16_t f32tof16_(float value)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "__float2half";
    case glsl: __intrinsic_asm "packHalf2x16(vec2($0,0.0))";
    case metal: __intrinsic_asm "half($0)";
    case spirv: return spirv_asm {
        OpFConvert $$float16_t result $value
    };
    case wgsl: __intrinsic_asm "f16($0)";
    }
}

__generic<let N : int>
[__readNone]
[require(cuda_glsl_metal_spirv_wgsl, shader5_sm_5_0)]
vector<float16_t, N> f32tof16_(vector<float, N> value)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "__float2half";
    case metal: __intrinsic_asm "$TR($0)";
    case spirv: return spirv_asm {
        OpFConvert $$vector<float16_t, N> result $value
    };
    default:
        VECTOR_MAP_UNARY(float16_t, N, f32tof16_, value);
    }
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

/// Flip vector to face forward, if needed.
/// @param n The vector to orient.
/// @param i The incident vector.
/// @param ng The geometric normal vector.
/// @return `n` if the dot product of `ng` and `i` is less than 0, otherwise `-n`.
/// @category math
__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> faceforward(vector<T,N> n, vector<T,N> i, vector<T,N> ng)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "faceforward";
    case hlsl: __intrinsic_asm "faceforward";
    case metal: __intrinsic_asm "faceforward";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,N> result glsl450 FaceForward $n $i $ng
    };
    case wgsl: __intrinsic_asm "faceForward";
    default:
        return dot(ng, i) < T(0.0f) ? n : -n;
    }
}

/// Find first set bit starting at high bit and working down.
/// @param value The value to find set bits in.
/// @return The bit index number of the most significant bit,
///         or returns -1 if `value` is either 0 if `value is
///         a signed type and equal to -1.
/// @remarks If `value` is unsigned, or signed with positive value, the bit index returned is the highest 1-bit.
///          If `value` is signed with negative value, the bit index returned is the highest 0-bit.
///          For SPIR-V, this function maps to GLSL extended instruction `FindSMsb` if `value` is signed,
///          or `FindUMsb` if `value` is unsigned.
/// @category bitops Bit operation functions
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
int firstbithigh(int value)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_firstbithigh($0)";
    case cuda: __intrinsic_asm "$P_firstbithigh($0)";
    case glsl: __intrinsic_asm "findMSB";
    case hlsl: __intrinsic_asm "firstbithigh";
    case metal: __intrinsic_asm "clz";
    case spirv: return spirv_asm {
        OpExtInst $$int result glsl450 FindSMsb $value
    };
    case wgsl: __intrinsic_asm "firstLeadingBit";
    }
}

__generic<let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
vector<int, N> firstbithigh(vector<int, N> value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "findMSB";
    case hlsl: __intrinsic_asm "firstbithigh";
    case metal: __intrinsic_asm "clz";
    case spirv: return spirv_asm {
        OpExtInst $$vector<int, N> result glsl450 FindSMsb $value
    };
    case wgsl: __intrinsic_asm "firstLeadingBit";
    default:
        VECTOR_MAP_UNARY(int, N, firstbithigh, value);
    }
}

[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
uint firstbithigh(uint value)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_firstbithigh($0)";
    case cuda: __intrinsic_asm "$P_firstbithigh($0)";
    case glsl: __intrinsic_asm "findMSB";
    case hlsl: __intrinsic_asm "firstbithigh";
    case metal: __intrinsic_asm "clz";
    case spirv: return spirv_asm {
        OpExtInst $$uint result glsl450 FindUMsb $value
    };
    case wgsl: __intrinsic_asm "firstLeadingBit";
    }
}

__generic<let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
vector<uint,N> firstbithigh(vector<uint,N> value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "findMSB";
    case hlsl: __intrinsic_asm "firstbithigh";
    case metal: __intrinsic_asm "clz";
    case spirv: return spirv_asm {
        OpExtInst $$vector<uint,N> result glsl450 FindUMsb $value
    };
    case wgsl: __intrinsic_asm "firstLeadingBit";
    default:
        VECTOR_MAP_UNARY(uint, N, firstbithigh, value);
    }
}

/// Find first set bit starting at low bit and working up.
/// @param value The value to find set bits in.
/// @return The bit index number of the least significant set bit,
///         or all ones (-1 when interpretted as signed) if `value` is 0.
/// @remarks For SPIR-V, this function maps to GLSL extended instruction `FindILsb`.
/// @category bitops
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
int firstbitlow(int value)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_firstbitlow($0)";
    case cuda: __intrinsic_asm "$P_firstbitlow($0)";
    case glsl: __intrinsic_asm "findLSB";
    case hlsl: __intrinsic_asm "firstbitlow";
    case metal: __intrinsic_asm "ctz";
    case spirv: return spirv_asm {
        OpExtInst $$int result glsl450 FindILsb $value
    };
    case wgsl: __intrinsic_asm "firstTrailingBit";
    }
}

__generic<let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
vector<int,N> firstbitlow(vector<int,N> value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "findLSB";
    case hlsl: __intrinsic_asm "firstbitlow";
    case metal: __intrinsic_asm "ctz";
    case spirv: return spirv_asm {
        OpExtInst $$vector<int,N> result glsl450 FindILsb $value
    };
    case wgsl: __intrinsic_asm "firstTrailingBit";
    default:
        VECTOR_MAP_UNARY(int, N, firstbitlow, value);
    }
}

[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
uint firstbitlow(uint value)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_firstbitlow($0)";
    case cuda: __intrinsic_asm "$P_firstbitlow($0)";
    case glsl: __intrinsic_asm "findLSB";
    case hlsl: __intrinsic_asm "firstbitlow";
    case metal: __intrinsic_asm "ctz";
    case spirv: return spirv_asm {
        OpExtInst $$uint result glsl450 FindILsb $value
    };
    case wgsl: __intrinsic_asm "firstTrailingBit";
    }
}

__generic<let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
vector<uint,N> firstbitlow(vector<uint,N> value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "findLSB";
    case hlsl: __intrinsic_asm "firstbitlow";
    case metal: __intrinsic_asm "ctz";
    case spirv: return spirv_asm {
        OpExtInst $$vector<uint,N> result glsl450 FindILsb $value
    };
    case wgsl: __intrinsic_asm "firstTrailingBit";
    default:
        VECTOR_MAP_UNARY(uint, N, firstbitlow, value);
    }
}

/// Floor. Returns the largest integer value not greater than `x`.
/// @param x The input value.
/// @return The largest integer value not greater than `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T floor(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_floor($0)";
    case cuda: __intrinsic_asm "$P_floor($0)";
    case glsl: __intrinsic_asm "floor";
    case hlsl: __intrinsic_asm "floor";
    case metal: __intrinsic_asm "floor";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Floor $x
    };
    case wgsl: __intrinsic_asm "floor";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> floor(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "floor";
    case hlsl: __intrinsic_asm "floor";
    case metal: __intrinsic_asm "floor";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Floor $x
    };
    case wgsl: __intrinsic_asm "floor";
    default:
        VECTOR_MAP_UNARY(T, N, floor, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> floor(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "floor";
    default:
        MATRIX_MAP_UNARY(T, N, M, floor, x);
    }
}

/// Fused multiply-add.
/// @param a The first value to multiply.
/// @param b The second value to multiply.
/// @param c The value to add to the product of `a` and `b`.
/// @return The result of `a * b + c`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
T fma(T a, T b, T c)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_fma($0, $1, $2)";
    case cuda: __intrinsic_asm "$P_fma($0, $1, $2)";
    case glsl: __intrinsic_asm "fma";
    case hlsl:
        if (__isFloat<T>() || __isHalf<T>())
            return mad(a, b, c);
        else
            __intrinsic_asm "fma($0, $1, $2)";
    case metal: __intrinsic_asm "fma";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Fma $a $b $c
    };
    case wgsl: __intrinsic_asm "fma";
    default:
        return a*b + c;
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
vector<T, N> fma(vector<T, N> a, vector<T, N> b, vector<T, N> c)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "fma";
    case hlsl: __intrinsic_asm "fma";
    case metal: __intrinsic_asm "fma";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Fma $a $b $c
    };
    case wgsl: __intrinsic_asm "fma";
    default:
        VECTOR_MAP_TRINARY(T, N, fma, a, b, c);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
matrix<T, N, M> fma(matrix<T, N, M> a, matrix<T, N, M> b, matrix<T, N, M> c)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "fma";
    default:
        MATRIX_MAP_TRINARY(T, N, M, fma, a, b, c);
    }
}

/// Floating point remainder of x/y.
/// The floating-point remainder is calculated such that x = i * y + f,
/// where i is an integer, f has the same sign as x, and the absolute value
/// of f is less than the absolute value of y.
/// @param x The dividend.
/// @param y The divisor.
/// @return The floating-point remainder of x/y.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T fmod(T x, T y)
{
    // In HLSL, `fmod` returns a remainder.
    // Definition of `fmod` in HLSL is,
    // "The floating-point remainder is calculated such that x = i * y + f,
    // where i is an integer, f has the same sign as x, and the absolute value
    // of f is less than the absolute value of y."
    //
    // In GLSL, `mod` is a Modulus function.
    // OpenGL document defines "Modulus" as "Returns x - y * floor(x / y)".
    // The use of "Floor()" makes the difference.
    //
    // In Metal, `fmod` is Modulus function.
    // Metal document defines it as "Returns x - y * trunc(x/y)".
    // Note that the function name is same to HLSL but it behaves differently.
    //
    // The tricky ones are when x or y is a negative value.
    //
    //       | Remainder | Modulus
    //  x  y | x= i*y +f | x-y*floor(x/y)
    // ------+-----------+------------------------------
    //  4  3 | 4= 1*3 +1 | 4-3*floor( 4/3) = 4-3* 1 = 1
    //  3  3 | 3= 1*3 +0 | 3-3*floor( 3/3) = 3-3* 1 = 0
    //  2  3 | 2= 0*3 +2 | 2-3*floor( 2/3) = 2-3* 0 = 2
    //  1  3 | 1= 0*3 +1 | 1-3*floor( 1/3) = 1-3* 0 = 1
    //  0  3 | 0= 0*3 +0 | 0-3*floor( 0/3) = 0-3* 0 = 0
    // -1  3 |-1= 0*3 -1 |-1-3*floor(-1/3) =-1-3*-1 = 2
    // -2  3 |-2= 0*3 -2 |-2-3*floor(-2/3) =-2-3*-1 = 1
    // -3  3 |-3=-1*3  0 |-3-3*floor(-3/3) =-3-3*-1 = 0
    // -4  3 |-4=-1*3 -1 |-4-3*floor(-4/3) =-4-3*-2 = 2
    //
    // When y is a negative value,
    //
    //       | Remainder | Modulus
    //  x  y | x= i*y +f | x-y*floor(x/y)
    // ------+-----------+------------------------------
    //  4 -3 | 4=-1*-3+1 | 4+3*floor( 4/-3) = 4+3*-2 =-2
    //  3 -3 | 3=-1*-3+0 | 3+3*floor( 3/-3) = 3+3*-1 = 0
    //  2 -3 | 2= 0*-3+2 | 2+3*floor( 2/-3) = 2+3*-1 =-1
    //  1 -3 | 1= 0*-3+1 | 1+3*floor( 1/-3) = 1+3*-1 =-2
    //  0 -3 | 0= 0*-3+0 | 0+3*floor( 0/-3) = 0+3* 0 = 0
    // -1 -3 |-1= 0*-3-1 |-1+3*floor(-1/-3) =-1+3* 0 =-1
    // -2 -3 |-2= 0*-3-2 |-2+3*floor(-2/-3) =-2+3* 0 =-2
    // -3 -3 |-3= 1*-3 0 |-3+3*floor(-3/-3) =-3+3* 1 = 0
    // -4 -3 |-4= 1*-3-1 |-4+3*floor(-4/-3) =-4+3* 1 =-1

    __target_switch
    {
    case cpp: __intrinsic_asm "$P_fmod($0, $1)";
    case cuda: __intrinsic_asm "$P_fmod($0, $1)";
    case glsl:
        // GLSL doesn't have a function for remainder.
        __intrinsic_asm "(($0 < 0.0) ? -mod(-$0,abs($1)) : mod($0,abs($1)))";
    case hlsl: __intrinsic_asm "fmod";
    case metal:
        // Metal doesn't have a function for remainder.
        __intrinsic_asm "(($0 < 0.0) ? -fmod(-$0,abs($1)) : fmod($0,abs($1)))";
    case spirv:
        // OpFRem return "The floating-point remainder whose sign
        // matches the sign of Operand 1", where Operand 1 is "x".
        return spirv_asm
        {
            result:$$T = OpFRem $x $y
        };
    case wgsl:
        __intrinsic_asm "(($0) % ($1))";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> fmod(vector<T, N> x, vector<T, N> y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "fmod";
    case spirv: return spirv_asm {
        result:$$vector<T,N> = OpFRem $x $y
    };
    default:
        VECTOR_MAP_BINARY(T, N, fmod, x, y);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> fmod(matrix<T, N, M> x, matrix<T, N, M> y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "fmod";
    default:
        MATRIX_MAP_BINARY(T, N, M, fmod, x, y);
    }
}

/// Extract the fractional part of a floating-point number.
/// @param x The input value.
/// @return The fractional part of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T frac(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_frac($0)";
    case cuda: __intrinsic_asm "$P_frac($0)";
    case glsl: __intrinsic_asm "fract";
    case hlsl: __intrinsic_asm "frac";
    case metal: __intrinsic_asm "fract";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Fract $x
    };
    case wgsl: __intrinsic_asm "fract";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> frac(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "fract";
    case hlsl: __intrinsic_asm "frac";
    case metal: __intrinsic_asm "fract";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Fract $x
    };
    case wgsl: __intrinsic_asm "fract";
    default:
        VECTOR_MAP_UNARY(T, N, frac, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
matrix<T, N, M> frac(matrix<T, N, M> x)
{
    MATRIX_MAP_UNARY(T, N, M, frac, x);
}

/// Extract the fractional part of a floating-point number.
/// @param x The input value.
/// @return The fractional part of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T fract(T x)
{
    return frac(x);
}

__generic<T : __BuiltinFloatingPointType, let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> fract(vector<T, N> x)
{
    return frac(x);
}

/// Split float into mantissa and exponent.
/// @param x The input value.
/// @param[out] exp The output exponent.
/// @return The mantissa of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T frexp(T x, out int exp)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_frexp($0, $1)";
    case cuda: __intrinsic_asm "$P_frexp($0, $1)";
    case glsl: __intrinsic_asm "frexp";
    case hlsl: __intrinsic_asm "frexp";
    case metal: __intrinsic_asm "frexp($0, *($1))";
    case spirv: return spirv_asm {
        result:$$T = OpExtInst glsl450 Frexp $x &exp
    };
    case wgsl:
        T fract;
        __wgsl_frexp<T>(x, fract, exp);
        return fract;
    }
}

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(wgsl)]
void __wgsl_frexp(T x, out T fract, out int exp)
{
    __intrinsic_asm "{ var s = frexp($0); ($1) = s.fract; ($2) = s.exp; }";
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> frexp(vector<T, N> x, out vector<int, N> exp)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "frexp";
    case hlsl: __intrinsic_asm "frexp";
    case metal: __intrinsic_asm "frexp($0, *($1))";
    case spirv: return spirv_asm {
        result:$$vector<T, N> = OpExtInst glsl450 Frexp $x &exp
    };
    case wgsl:
        vector<T,N> fract;
        __wgsl_frexp<T>(x, fract, exp);
        return fract;
    default:
        VECTOR_MAP_BINARY(T, N, frexp, x, exp);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[ForceInline]
[require(wgsl)]
void __wgsl_frexp(vector<T, N> x, out vector<T, N> fract, out vector<int, N> exp)
{
    __intrinsic_asm "{ var s = frexp($0); ($1) = s.fract; ($2) = s.exp; }";
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int, let L : int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> frexp(matrix<T, N, M> x, out matrix<int, N, M, L> exp)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "frexp";
    default:
        MATRIX_MAP_BINARY(T, N, M, frexp, x, exp);
    }
}

/// Texture filter width.
/// Calculates the sum abs(ddx(`p`)) + abs(ddy(`p`)).
/// @param p The value to sum x and y partial derivative magnitudes for.
/// @return The sum of abs(ddx(`p`)) and abs(ddy(`p`)).
/// @remarks For SPIR-V, this function maps to `OpFwidth`.
/// @category derivative
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(glsl_hlsl_metal_spirv_wgsl, fragmentprocessing)]
T fwidth(T p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "fwidth($0)";
    case glsl:
        __intrinsic_asm "fwidth($0)";
    case metal:
        __intrinsic_asm "fwidth($0)";
    case spirv:
        return spirv_asm
        {
            OpFwidth $$T result $p;
        };
    case wgsl:
        __intrinsic_asm "fwidth($0)";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(glsl_hlsl_spirv_wgsl, fragmentprocessing)]
vector<T, N> fwidth(vector<T, N> x)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "fwidth($0)";
    case glsl:
        __intrinsic_asm "fwidth($0)";
    case spirv:
        return spirv_asm
        {
            OpFwidth $$vector<T, N> result $x;
        };
    case wgsl:
        __intrinsic_asm "fwidth($0)";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(glsl_hlsl_spirv, fragmentprocessing)]
matrix<T, N, M> fwidth(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "fwidth($0)";
    default:
        MATRIX_MAP_UNARY(T, N, M, fwidth, x);
    }
}

/// Texture filter width (coarse).
/// Calculates the sum abs(ddx_coarse(`p`)) + abs(ddy_coarse(`p`)).
/// @param p The value to sum x and y partial derivative magnitudes for.
/// @return The sum of abs(ddx_coarse(`p`)) and abs(ddy_coarse(`p`)).
/// @remarks For SPIR-V, this function maps to `OpFwidthCoarse`.
/// @category derivative
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
T fwidth_coarse(T p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "abs(ddx_coarse($0)) + abs(ddy_coarse($0))";
    case glsl:
        __intrinsic_asm "fwidthCoarse($0)";
    case spirv:
        return spirv_asm
        {
            OpCapability DerivativeControl;
            OpFwidthCoarse $$T result $p;
        };
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
vector<T, N> fwidth_coarse(vector<T, N> x)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "abs(ddx_coarse($0)) + abs(ddy_coarse($0))";
    case glsl:
        __intrinsic_asm "fwidthCoarse($0)";
    case spirv:
        return spirv_asm
        {
            OpCapability DerivativeControl;
            OpFwidthCoarse $$vector<T, N> result $x;
        };
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(glsl_hlsl_spirv, fragmentprocessing)]
matrix<T, N, M> fwidth_coarse(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "abs(ddx_coarse($0)) + abs(ddy_coarse($0))";
    default:
        MATRIX_MAP_UNARY(T, N, M, fwidth_coarse, x);
    }
}

/// Texture filter width (fine).
/// Calculates the sum abs(ddx_fine(`p`)) + abs(ddy_fine(`p`)).
/// @param p The value to sum x and y partial derivative magnitudes for.
/// @return The sum of abs(ddx_fine(`p`)) and abs(ddy_fine(`p`)).
/// @remarks For SPIR-V, this function maps to `OpFwidthFine`.
/// @category derivative
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
T fwidth_fine(T p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "abs(ddx_fine($0)) + abs(ddy_fine($0))";
    case glsl:
        __intrinsic_asm "fwidthFine($0)";
    case spirv:
        return spirv_asm
        {
            OpCapability DerivativeControl;
            OpFwidthFine $$T result $p;
        };
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
vector<T, N> fwidth_fine(vector<T, N> x)
{
    __requireComputeDerivative();
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "abs(ddx_fine($0)) + abs(ddy_fine($0))";
    case glsl:
        __intrinsic_asm "fwidthFine($0)";
    case spirv:
        return spirv_asm
        {
            OpCapability DerivativeControl;
            OpFwidthFine $$vector<T, N> result $x;
        };
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(glsl_hlsl_spirv, fragmentprocessing)]
matrix<T, N, M> fwidth_fine(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "abs(ddx_fine($0)) + abs(ddy_fine($0))";
    default:
        MATRIX_MAP_UNARY(T, N, M, fwidth_fine, x);
    }
}

__intrinsic_op($(kIROp_ResolveVaryingInputRef))
Ref<T> __ResolveVaryingInputRef<T>(__constref T attribute);

__intrinsic_op($(kIROp_GetPerVertexInputArray))
Ref<Array<T, 3>> __GetPerVertexInputArray<T>(__constref T attribute);

T __GetAttributeAtVertex<T>(__constref T attribute, uint vertexIndex)
{
    __intrinsic_asm "GetAttributeAtVertex";
}

/// Get the value of a vertex attribute at a specific vertex.
///
/// The `GetAttributeAtVertex()` function can be used in a fragment shader
/// to get the value of the given `attribute` at the vertex of the primitive
/// that corresponds to the given `vertexIndex`.
///
/// Note that the `attribute` must have been a declared varying input to
/// the fragment shader with the `nointerpolation` modifier.
///
/// This function can be applied to scalars, vectors, and matrices of
/// built-in scalar types.
///
__generic<T : __BuiltinType>
[__readNone]
__glsl_version(450)
__glsl_extension(GL_EXT_fragment_shader_barycentric)
[require(glsl_hlsl_spirv, getattributeatvertex)]
[KnownBuiltin("GetAttributeAtVertex")]
[__unsafeForceInlineEarly]
T GetAttributeAtVertex(__constref T attribute, uint vertexIndex)
{
    __target_switch
    {
    case hlsl:
        return __GetAttributeAtVertex(__ResolveVaryingInputRef(attribute), vertexIndex);
    case glsl:
    case spirv:
        return __GetPerVertexInputArray(__ResolveVaryingInputRef(attribute))[vertexIndex];
    }
}

/// Get the value of a vertex attribute at a specific vertex.
///
/// The `GetAttributeAtVertex()` function can be used in a fragment shader
/// to get the value of the given `attribute` at the vertex of the primitive
/// that corresponds to the given `vertexIndex`.
///
/// Note that the `attribute` must have been a declared varying input to
/// the fragment shader with the `nointerpolation` modifier.
///
/// This function can be applied to scalars, vectors, and matrices of
/// built-in scalar types.
///
__generic<T : __BuiltinType, let N : int>
[__readNone]
__glsl_version(450)
__glsl_extension(GL_EXT_fragment_shader_barycentric)
[require(glsl_hlsl_spirv, getattributeatvertex)]
[__unsafeForceInlineEarly]
vector<T,N> GetAttributeAtVertex(__constref vector<T,N> attribute, uint vertexIndex)
{
    __target_switch
    {
    case hlsl:
        return __GetAttributeAtVertex(__ResolveVaryingInputRef(attribute), vertexIndex);
    case glsl:
    case spirv:
        return __GetPerVertexInputArray(__ResolveVaryingInputRef(attribute))[vertexIndex];
    }
}

/// Get the value of a vertex attribute at a specific vertex.
///
/// The `GetAttributeAtVertex()` function can be used in a fragment shader
/// to get the value of the given `attribute` at the vertex of the primitive
/// that corresponds to the given `vertexIndex`.
///
/// Note that the `attribute` must have been a declared varying input to
/// the fragment shader with the `nointerpolation` modifier.
///
/// This function can be applied to scalars, vectors, and matrices of
/// built-in scalar types.
///
__generic<T : __BuiltinType, let N : int, let M : int>
[__readNone]
__glsl_version(450)
__glsl_extension(GL_EXT_fragment_shader_barycentric)
[require(glsl_hlsl_spirv, getattributeatvertex)]
[__unsafeForceInlineEarly]
matrix<T,N,M> GetAttributeAtVertex(__constref matrix<T,N,M> attribute, uint vertexIndex)
{
    __target_switch
    {
    case hlsl:
        return __GetAttributeAtVertex(__ResolveVaryingInputRef(attribute), vertexIndex);
    case glsl:
    case spirv:
        return __GetPerVertexInputArray(__ResolveVaryingInputRef(attribute))[vertexIndex];
    }
}

// Get number of samples in render target
[__readNone]
[require(hlsl, sm_4_0)]
[require(metal)]
uint GetRenderTargetSampleCount()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "GetRenderTargetSampleCount";
    case metal: __intrinsic_asm "get_num_samples";
    }
}

// Get position of given sample
[__readNone]
[require(hlsl, sm_4_0)]
[require(metal)]
float2 GetRenderTargetSamplePosition(int Index)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "GetRenderTargetSamplePosition";
    case metal: __intrinsic_asm "get_sample_position";
    }
}

/// Group memory barrier. Ensures that all memory accesses in the group are visible to all threads in the group.
/// @category barrier
__glsl_extension(GL_KHR_memory_scope_semantics)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, memorybarrier)]
void GroupMemoryBarrier()
{
    __target_switch
    {
    case glsl: __intrinsic_asm "memoryBarrier(gl_ScopeWorkgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease)";
    case hlsl: __intrinsic_asm "GroupMemoryBarrier";
    case cuda: __intrinsic_asm "__threadfence_block";
    case metal: __intrinsic_asm "threadgroup_barrier(mem_flags::mem_threadgroup)";
    case spirv:
        spirv_asm
        {
            OpMemoryBarrier Workgroup AcquireRelease|WorkgroupMemory
        };
    case wgsl: __intrinsic_asm "workgroupBarrier";
    }
}

[require(cuda_glsl_hlsl_metal_spirv, memorybarrier)]
void __subgroupBarrier()
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupBarrier";
    case hlsl: __intrinsic_asm "GroupMemoryBarrierWithGroupSync";
    case cuda: __intrinsic_asm "__syncthreads()";
    case metal: __intrinsic_asm "simdgroup_barrier(mem_flags::none)";
    case spirv:
        spirv_asm
        {
            OpControlBarrier Subgroup Subgroup AcquireRelease|WorkgroupMemory|ImageMemory|UniformMemory
        };
    }
}

/// Group memory barrier. Ensures that all memory accesses in the group are visible to all threads in the group.
/// @category barrier
__glsl_extension(GL_KHR_memory_scope_semantics)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, memorybarrier)]
void GroupMemoryBarrierWithGroupSync()
{
    __target_switch
    {
    case glsl: __intrinsic_asm "controlBarrier(gl_ScopeWorkgroup, gl_ScopeWorkgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease)";
    case hlsl: __intrinsic_asm "GroupMemoryBarrierWithGroupSync";
    case cuda: __intrinsic_asm "__syncthreads()";
    case metal: __intrinsic_asm "threadgroup_barrier(mem_flags::mem_threadgroup)";
    case spirv:
        spirv_asm
        {
            OpControlBarrier Workgroup Workgroup AcquireRelease|WorkgroupMemory
        };
    case wgsl: __intrinsic_asm "workgroupBarrier";
    }
}

// Atomics

// Accepts an ImageSubscript
// Gets Texture used with ImageSubscript.
__generic<TextureAccess>
__intrinsic_op($(kIROp_ExtractTextureFromTextureAccess))
TextureAccess* __extractTextureFromTextureAccess(__ref TextureAccess x);

// Accepts an ImageSubscript
// Gets Coord from ImageSubscript. Swizzles out ArrayCoord if applicable
__generic<TextureAccess>
__intrinsic_op($(kIROp_ExtractCoordFromTextureAccess))
uint __extractCoordFromTextureAccess(__ref TextureAccess x);

// Accepts an ImageSubscript
// Gets ArrayCoord from ImageSubscript
__generic<TextureAccess>
__intrinsic_op($(kIROp_ExtractArrayCoordFromTextureAccess))
uint __extractArrayCoordFromTextureAccess(__ref TextureAccess x);

${{{{
// Generates code for:
// InterlockedAdd, InterlockedAnd, InterlockedOr, InterlockedXor,
// InterlockedMax, InterlockedMin, InterlockedExchange
struct SlangAtomicOperationInfo
{
    const char* slangCallSuffix;
    const char* internalCallSuffix;
    const char* interface;
};

SlangAtomicOperationInfo slangAtomicOperationInfo[7] = {
    { "Add", "add", "IArithmeticAtomicable" },
    { "And", "and", "IArithmeticAtomicable" },
    { "Or", "or", "IArithmeticAtomicable" },
    { "Xor", "xor", "IArithmeticAtomicable" },
    { "Max", "max", "IArithmeticAtomicable" },
    { "Min", "min", "IArithmeticAtomicable" },
    { "Exchange", "exchange", "IAtomicable" },
};

for (SlangAtomicOperationInfo atomicOp : slangAtomicOperationInfo)
{
}}}}

/// Perform an atomic $(atomicOp.internalCallSuffix) operation on `dest`.
/// @param T The type of the value to perform the atomic operation on.
/// @param dest The value to perform the atomic operation on.
/// @param value The operand to the atomic operation.
/// @param original_value The value of `dest` before the operation.
/// @remarks When targeting HLSL, it is invalid to call this function with `T` being a floating-point type, since
/// HLSL does not allow atomic operations on floating point types. For `InterlockedAdd`, consider using
/// `RWByteAddressBuffer.InterlockedAddF32` or `RWByteAddressBuffer.InterlockedAddF16` instead when NVAPI is available.
/// On SPIR-V (Vulkan), all integer and floating point types are supported.
/// On Metal and WGSL, all floating-point types are not supported.
/// @category atomic Atomic functions
[ForceInline]
__glsl_version(430)
[require(cuda_glsl_hlsl_metal_spirv, atomic_glsl_hlsl_cuda_metal)]
void Interlocked$(atomicOp.slangCallSuffix)<T:$(atomicOp.interface)>(__ref T dest,  T value)
{
    __atomic_$(atomicOp.internalCallSuffix)(dest, value);
}

[ForceInline]
__glsl_version(430)
[require(cuda_glsl_hlsl_metal_spirv, atomic_glsl_hlsl_cuda_metal)]
void Interlocked$(atomicOp.slangCallSuffix)<T:$(atomicOp.interface)>(__ref T dest, T value, out T original_value)
{
    original_value = __atomic_$(atomicOp.internalCallSuffix)(dest, value);
}

[ForceInline]
__glsl_version(430)
[require(cuda_glsl_hlsl_metal_spirv, atomic_glsl_hlsl_cuda_metal)]
void Interlocked$(atomicOp.slangCallSuffix)(__ref uint dest, int value)
{
    __atomic_$(atomicOp.internalCallSuffix)(dest, (uint)value);
}

${{{{
} // for (SlangAtomicOperationInfo atomicOp : slangAtomicOperationInfo)
}}}}

/// Perform an atomic compare and exchange operation on `dest`.
/// @param T The type of the value to perform the atomic operation on.
/// @param dest The value to perform the atomic operation on.
/// @param compare_value The value to compare `dest` with.
/// @param value The value to store into `dest` if the compare result is equal.
/// @param original_value The value of `dest` before the operation.
/// @remarks When targeting HLSL, a call to this function with `T` being `float` will translate to a call to
/// `InterlockedCompareExchangeFloatBitwise`, which means the comparison is done as a bitwise comparison.
///
/// On SPIR-V (Vulkan), this function maps to `OpAtomicCompareExchange`.
///
/// On Metal and WGSL, all floating-point types are not supported.
///
/// On CUDA, this function maps to `atomicCAS`.
/// @category atomic
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv, atomic_glsl_hlsl_cuda_metal)]
void InterlockedCompareExchange<T:IAtomicable>(__ref T dest, T compare_value, T value, out T original_value)
{
    original_value = __atomic_compare_exchange(dest, compare_value, value);
}

/// Perform an atomic compare and exchange operation on `dest`.
/// @param T The type of the value to perform the atomic operation on.
/// @param dest The value to perform the atomic operation on.
/// @param compare_value The value to compare `dest` with.
/// @param value The value to store into `dest` if the compare result is equal.
/// @param original_value The value of `dest` before the operation.
/// @remarks When targeting HLSL, a call to this function will translate to a call to
/// `InterlockedCompareExchangeFloatBitwise`, which means the comparison is done as a bitwise comparison.
///
/// On SPIR-V (Vulkan), this function maps to `OpAtomicCompareExchange`.
///
/// On Metal and WGSL, this function is not available.
///
/// On CUDA, this function maps to `atomicCAS`.
/// @category atomic
[ForceInline]
void InterlockedCompareExchangeFloatBitwise(__ref  float dest, float compare_value, float value)
{
    __atomic_compare_exchange(dest, compare_value, value);
}

[ForceInline]
void InterlockedCompareExchangeFloatBitwise(__ref  float dest, float compare_value, float value, out float original_value)
{
    original_value = __atomic_compare_exchange(dest, compare_value, value);
}

/// Perform an atomic compare and store operation on `dest`.
/// @param T The type of the value to perform the atomic operation on.
/// @param dest The value to perform the atomic operation on.
/// @param compare_value The value to compare `dest` with.
/// @param value The value to store into `dest` if the compare result is equal.
/// @remarks When targeting HLSL, a call to this function with `T` being `float` will translate to a call to
/// `InterlockedCompareStoreFloatBitwise`, which means the comparison is done as a bitwise comparison.
///
/// On SPIR-V (Vulkan), this function maps to `OpAtomicCompareExchange`.
///
/// On Metal and WGSL, this function is not available.
///
/// On CUDA, this function maps to `atomicCAS`.
/// @category atomic
[ForceInline]
__glsl_version(430)
[require(cuda_glsl_hlsl_metal_spirv, atomic_glsl_hlsl_cuda_metal)]
void InterlockedCompareStore<T:IAtomicable>(__ref T dest,  T compare_value,  T value)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "InterlockedCompareStore";
    default:
        __atomic_compare_exchange(dest, compare_value, value);
        return;
    }
}

/// Perform an atomic compare and store operation on `dest`.
/// @param T The type of the value to perform the atomic operation on.
/// @param dest The value to perform the atomic operation on.
/// @param compare_value The value to compare `dest` with.
/// @param value The value to store into `dest` if the compare result is equal.
/// @remarks When targeting HLSL, a call to this function will translate to a call to
/// `InterlockedCompareStoreFloatBitwise`, which means the comparison is done as a bitwise comparison.
///
/// On SPIR-V (Vulkan), this function maps to `OpAtomicCompareExchange`.
///
/// On Metal and WGSL, this function is not available.
///
/// On CUDA, this function maps to `atomicCAS`.
/// @category atomic
[ForceInline]
void InterlockedCompareStoreFloatBitwise<T:IAtomicable>(__ref  T dest,  T compare_value, T value)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "InterlockedCompareStoreFloatBitwise";
    default:
        __atomic_compare_exchange(dest, compare_value, value);
        return;
    }
}


/// Test if a floating-point value is finite.
/// @param x The input value.
/// @return `true` if `x` is finite, `false` otherwise.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
bool isfinite(T x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "isfinite";
    case cuda:
    case cpp:
        __intrinsic_asm "$P_isfinite($0)";
    case metal:
        __intrinsic_asm "isfinite";
    default:
        return !(isinf(x) || isnan(x));
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<bool, N> isfinite(vector<T, N> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "isfinite";
    case glsl:
    case spirv:
        return !(isinf(x) || isnan(x));
    case metal:
        __intrinsic_asm "isfinite";
    default:
        VECTOR_MAP_UNARY(bool, N, isfinite, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<bool, N, M> isfinite(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "isfinite";
    default:
        MATRIX_MAP_UNARY(bool, N, M, isfinite, x);
    }
}

/// Test if a floating-point value is infinite.
/// @param x The input value.
/// @return `true` if `x` is infinite, `false` otherwise.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
bool isinf(T x)
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case metal:
        __intrinsic_asm "isinf";
    case cuda:
    case cpp:
        __intrinsic_asm "$P_isinf($0)";
    case spirv:
        return spirv_asm { result:$$bool = OpIsInf $x};
    case wgsl:
        __intrinsic_asm "($0 > 0x1.fffffep+127f) || ($0 < -0x1.fffffep+127f)";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<bool, N> isinf(vector<T, N> x)
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case metal:
        __intrinsic_asm "isinf";
    case spirv:
        return spirv_asm { result:$$vector<bool,N> = OpIsInf $x};
    default:
        VECTOR_MAP_UNARY(bool, N, isinf, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<bool, N, M> isinf(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "isinf";
    default:
        MATRIX_MAP_UNARY(bool, N, M, isinf, x);
    }
}

/// Test if a floating-point value is not-a-number.
/// @param x The input value.
/// @return `true` if `x` is not-a-number, `false` otherwise.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
bool isnan(T x)
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case metal:
        __intrinsic_asm "isnan";
    case cuda:
    case cpp:
        __intrinsic_asm "$P_isnan($0)";
    case spirv:
        return spirv_asm { result:$$bool = OpIsNan $x};
    case wgsl:
         __intrinsic_asm "$0 != $0";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<bool, N> isnan(vector<T, N> x)
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case metal:
        __intrinsic_asm "isnan";
    case spirv:
        return spirv_asm { result:$$vector<bool, N> = OpIsNan $x};
    default:
        VECTOR_MAP_UNARY(bool, N, isnan, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<bool, N, M> isnan(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "isnan";
    default:
        MATRIX_MAP_UNARY(bool, N, M, isnan, x);
    }
}

/// Construct float from mantissa and exponent.
/// @param x The significand.
/// @param exp The exponent.
/// @return The floating-point number constructed from `x` and `exp`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T ldexp(T x, T exp)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "ldexp";
    // In WGSL spec, ldexp can only take integer as the exponent.
    case wgsl: __intrinsic_asm "($0 * exp2($1))";
    default:
        return x * exp2(exp);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> ldexp(vector<T, N> x, vector<T, N> exp)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "ldexp";
    // In WGSL spec, ldexp can only take integer as the exponent.
    case wgsl: __intrinsic_asm "($0 * exp2($1))";
    default:
        return x * exp2(exp);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> ldexp(matrix<T, N, M> x, matrix<T, N, M> exp)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "ldexp";
    default:
        MATRIX_MAP_BINARY(T, N, M, ldexp, x, exp);
    }
}

__generic<T : __BuiltinFloatingPointType, E : __BuiltinIntegerType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T ldexp(T x, E exp)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "ldexp";
    case hlsl: __intrinsic_asm "ldexp";
    case metal: __intrinsic_asm "ldexp";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Ldexp $x $exp
    };
    case wgsl: __intrinsic_asm "ldexp";
    default:
        return ldexp(x, __realCast<T>(exp));
    }
}

__generic<T : __BuiltinFloatingPointType, E : __BuiltinIntegerType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> ldexp(vector<T, N> x, vector<E, N> exp)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "ldexp";
    case hlsl: __intrinsic_asm "ldexp";
    case metal: __intrinsic_asm "ldexp";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,N> result glsl450 Ldexp $x $exp
    };
    case wgsl: __intrinsic_asm "ldexp";
    default:
        vector<T,N> temp;
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
            temp[i] = __realCast<T>(exp[i]);
        return ldexp(x, temp);
    }
}


/// Compute the length of a vector.
/// @param x The input vector.
/// @return The length of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T length(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "length";
    case hlsl: __intrinsic_asm "length";
    case metal: __intrinsic_asm "length";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Length $x
    };
    case wgsl: __intrinsic_asm "length";
    default:
        return sqrt(dot(x, x));
    }
}

__generic<T : __BuiltinFloatingPointType>
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T length(T x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "length"; 
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Length $x
    };
    case wgsl: __intrinsic_asm "length"; 
    default:
        return abs(x);
    }
}

/// Computes linear interpolation.
/// @param x The starting value.
/// @param y The ending value.
/// @param s The interpolation factor.
/// @return Returns `x+(y-x)*s`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
T lerp(T x, T y, T s)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "mix";
    case wgsl: __intrinsic_asm "mix";
    case metal: __intrinsic_asm "mix";
    case hlsl: __intrinsic_asm "lerp";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 FMix $x $y $s
    };
    default:
        return x + (y - x) * s;
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
vector<T, N> lerp(vector<T, N> x, vector<T, N> y, vector<T, N> s)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "mix";
    case wgsl: __intrinsic_asm "mix";
    case metal: __intrinsic_asm "mix";
    case hlsl: __intrinsic_asm "lerp";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 FMix $x $y $s
    };
    default:
        return x + (y - x) * s;
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
matrix<T,N,M> lerp(matrix<T,N,M> x, matrix<T,N,M> y, matrix<T,N,M> s)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "lerp";
    default:
        MATRIX_MAP_TRINARY(T, N, M, lerp, x, y, s);
    }
}

/// Legacy lighting function (obsolete).
/// @param n_dot_l The dot product of the normal and light vectors.
/// @param n_dot_h The dot product of the normal and half-angle vectors.
/// @param m The material shininess factor.
/// @return The lighting coefficients, (ambient, diffuse, specular, 1.0).
/// @remarks In HLSL, this function is implemented as an intrinsic. It is emulated for other targets.
/// @deprecated
/// @category math
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
float4 lit(float n_dot_l, float n_dot_h, float m)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "lit";
    default:
        let ambient = 1.0f;
        let diffuse = max(n_dot_l, 0.0f);
        let specular = step(0.0f, n_dot_l) * max(pow(n_dot_h, m), 0.0f);
        return float4(ambient, diffuse, specular, 1.0f);
    }
}

/// Compute base-e logarithm.
/// @param x The input value.
/// @return The natural logarithm of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T log(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_log($0)";
    case cuda: __intrinsic_asm "$P_log($0)";
    case glsl: __intrinsic_asm "log";
    case hlsl: __intrinsic_asm "log";
    case metal: __intrinsic_asm "log";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Log $x
    };
    case wgsl: __intrinsic_asm "log";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> log(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "log";
    case hlsl: __intrinsic_asm "log";
    case metal: __intrinsic_asm "log";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Log $x
    };
    case wgsl: __intrinsic_asm "log";
    default:
        VECTOR_MAP_UNARY(T, N, log, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> log(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "log";
    default:
        MATRIX_MAP_UNARY(T, N, M, log, x);
    }
}

/// Compute base-10 logarithm.
/// @param x The input value.
/// @return The base-10 logarithm of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T log10(T x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "log10";
    case metal: __intrinsic_asm "log10";
    case wgsl: __intrinsic_asm "(log( $0 ) * $S0( 0.43429448190325182765112891891661) )";
    case glsl: __intrinsic_asm "(log( $0 ) * $S0( 0.43429448190325182765112891891661) )";
    case cuda: __intrinsic_asm "$P_log10($0)";
    case cpp: __intrinsic_asm "$P_log10($0)";
    case spirv:
    {
        const T tmp = T(0.43429448190325182765112891891661);
        return spirv_asm {
            %baseElog:$$T = OpExtInst glsl450 Log $x;
            result:$$T = OpFMul %baseElog $tmp
        };
    }
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> log10(vector<T,N> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "log10";
    case metal: __intrinsic_asm "log10";
    case wgsl: __intrinsic_asm "(log( $0 ) * $S0(0.43429448190325182765112891891661) )";
    case glsl: __intrinsic_asm "(log( $0 ) * $S0(0.43429448190325182765112891891661) )";
    case spirv:
    {
        const T tmp = T(0.43429448190325182765112891891661);
        return spirv_asm {
            %baseElog:$$vector<T,N> = OpExtInst glsl450 Log $x;
            result:$$vector<T,N> = OpVectorTimesScalar %baseElog $tmp
        };
    }
    default:
        VECTOR_MAP_UNARY(T, N, log10, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
matrix<T,N,M> log10(matrix<T,N,M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "log10";
    default:
        MATRIX_MAP_UNARY(T, N, M, log10, x);
    }
}

/// Compute base-2 logarithm.
/// @param x The input value.
/// @return The base-2 logarithm of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T log2(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_log2($0)";
    case cuda: __intrinsic_asm "$P_log2($0)";
    case glsl: __intrinsic_asm "log2";
    case hlsl: __intrinsic_asm "log2";
    case metal: __intrinsic_asm "log2";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Log2 $x
    };
    case wgsl: __intrinsic_asm "log2";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> log2(vector<T,N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "log2";
    case hlsl: __intrinsic_asm "log2";
    case metal: __intrinsic_asm "log2";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,N> result glsl450 Log2 $x
    };
    case wgsl: __intrinsic_asm "log2";
    default:
        VECTOR_MAP_UNARY(T, N, log2, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> log2(matrix<T,N,M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "log2";
    default:
        MATRIX_MAP_UNARY(T, N, M, log2, x);
    }
}

/// Computes multiply-add.
/// @param mvalue The multiplier.
/// @param avalue The multiplicand.
/// @param bvalue The addend.
/// @return The result of `mvalue * avalue + bvalue`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, shader5_sm_5_0)]
T mad(T mvalue, T avalue, T bvalue)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_fma($0, $1, $2)";
    case cuda: __intrinsic_asm "$P_fma($0, $1, $2)";
    case glsl: __intrinsic_asm "fma";
    case hlsl: __intrinsic_asm "mad";
    case metal: __intrinsic_asm "fma";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Fma $mvalue $avalue $bvalue
    };
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, shader5_sm_5_0)]
vector<T, N> mad(vector<T, N> mvalue, vector<T, N> avalue, vector<T, N> bvalue)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "fma";
    case hlsl: __intrinsic_asm "mad";
    case metal: __intrinsic_asm "fma";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Fma $mvalue $avalue $bvalue
    };
    default:
        VECTOR_MAP_TRINARY(T, N, mad, mvalue, avalue, bvalue);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, shader5_sm_5_0)]
matrix<T, N, M> mad(matrix<T, N, M> mvalue, matrix<T, N, M> avalue, matrix<T, N, M> bvalue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "mad";
    default:
        MATRIX_MAP_TRINARY(T, N, M, mad, mvalue, avalue, bvalue);
    }
}

__generic<T : __BuiltinIntegerType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
T mad(T mvalue, T avalue, T bvalue)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_fma($0, $1, $2)";
    case cuda: __intrinsic_asm "$P_fma($0, $1, $2)";
    case glsl: __intrinsic_asm "fma";
    case hlsl: __intrinsic_asm "mad";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Fma $mvalue $avalue $bvalue
    };
    }
}

__generic<T : __BuiltinIntegerType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
vector<T, N> mad(vector<T, N> mvalue, vector<T, N> avalue, vector<T, N> bvalue)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "fma";
    case hlsl: __intrinsic_asm "mad";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Fma $mvalue $avalue $bvalue
    };
    default:
        VECTOR_MAP_TRINARY(T, N, mad, mvalue, avalue, bvalue);
    }
}

__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
matrix<T, N, M> mad(matrix<T, N, M> mvalue, matrix<T, N, M> avalue, matrix<T, N, M> bvalue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "mad";
    default:
        MATRIX_MAP_TRINARY(T, N, M, mad, mvalue, avalue, bvalue);
    }
}

/// Maximum.
/// @param x The first value.
/// @param y The second value.
/// @return The maximum of `x` and `y`.
/// @category math
__generic<T : __BuiltinIntegerType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T max(T x, T y)
{
    // Note: a core module implementation of `max` (or `min`) will require splitting
    // floating-point and integer cases apart, because the floating-point
    // version needs to correctly handle the case where one of the inputs
    // is not-a-number.

    __target_switch
    {
    case hlsl: __intrinsic_asm "max";
    case glsl: __intrinsic_asm "max";
    case metal: __intrinsic_asm "max";
    case cuda: __intrinsic_asm "$P_max($0, $1)";
    case cpp: __intrinsic_asm "$P_max($0, $1)";
    case spirv:
    {
        if (__isSignedInt<T>())
        {
            return spirv_asm {
                result:$$T = OpExtInst glsl450 SMax $x $y
            };
        }
        else
        {
            return spirv_asm {
                result:$$T = OpExtInst glsl450 UMax $x $y
            };
        }
    }
    case wgsl: __intrinsic_asm "max";
    }
}

/// @category math
__generic<T : __BuiltinIntegerType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> max(vector<T, N> x, vector<T, N> y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "max";
    case glsl: __intrinsic_asm "max";
    case metal: __intrinsic_asm "max";
    case spirv:
    {
        if (__isSignedInt<T>())
        {
            return spirv_asm {
                result:$$vector<T,N> = OpExtInst glsl450 SMax $x $y
            };
        }
        else
        {
            return spirv_asm {
                result:$$vector<T,N> = OpExtInst glsl450 UMax $x $y
            };
        }
    }
    case wgsl: __intrinsic_asm "max";
    default:
        VECTOR_MAP_BINARY(T, N, max, x, y);
    }
}

/// @category math
__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> max(matrix<T, N, M> x, matrix<T, N, M> y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "max";
    default:
        MATRIX_MAP_BINARY(T, N, M, max, x, y);
    }
}

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T max(T x, T y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "max";
    case metal: __intrinsic_asm "max";
    case glsl: __intrinsic_asm "max";
    case cuda: __intrinsic_asm "$P_max($0, $1)";
    case cpp: __intrinsic_asm "$P_max($0, $1)";
    case spirv: return spirv_asm {
        result:$$T = OpExtInst glsl450 FMax $x $y
    };
    case wgsl: __intrinsic_asm "max";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> max(vector<T, N> x, vector<T, N> y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "max";
    case metal: __intrinsic_asm "max";
    case glsl: __intrinsic_asm "max";
    case spirv: return spirv_asm {
        result:$$vector<T, N> = OpExtInst glsl450 FMax $x $y
    };
    case wgsl: __intrinsic_asm "max";
    default:
        VECTOR_MAP_BINARY(T, N, max, x, y);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> max(matrix<T, N, M> x, matrix<T, N, M> y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "max";
    default:
        MATRIX_MAP_BINARY(T, N, M, max, x, y);
    }
}

/// Maximum of 3 inputs.
/// @param x The first value to compare.
/// @param y The second value to compare.
/// @param z The third value to compare.
/// @return The largest of the three values, element-wise if vector typed.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
T max3(T x, T y, T z)
{
    __target_switch
    {
    case metal: __intrinsic_asm "max3";
    default:
        return max(x, max(y, z));
    }
}

__generic<T : __BuiltinFloatingPointType, let N: int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
vector<T,N> max3(vector<T,N> x, vector<T,N> y, vector<T,N> z)
{
    __target_switch
    {
    case metal: __intrinsic_asm "max3";
    default:
        return max(x, max(y, z));
    }
}

/// Floating-point maximum.
/// @param x The first value to compare.
/// @param y The second value to compare.
/// @return The larger of the two values, element-wise if vector typed.
/// @remarks Result is `y` if `x` < `y`, either `x` or `y` if both `x` and `y` are zeros, otherwise `x`. Which operand is the result is undefined if one of the operands is a NaN.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
T fmax(T x, T y)
{
    __target_switch
    {
    case metal: __intrinsic_asm "fmax";
    default:
        return max(x, y);
    }
}

__generic<T : __BuiltinFloatingPointType, let N: int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
vector<T,N> fmax(vector<T,N> x, vector<T,N> y)
{
    __target_switch
    {
    case metal: __intrinsic_asm "fmax";
    default:
        VECTOR_MAP_BINARY(T, N, fmax, x, y);
    }
}

/// Floating-point maximum of 3 inputs.
/// @param x The first value to compare.
/// @param y The second value to compare.
/// @param z The third value to compare.
/// @return The largest of the three values, element-wise if vector typed.
/// @remarks If any operand in the 3-way comparison is NaN, it is undefined which operand is returned.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
T fmax3(T x, T y, T z)
{
    __target_switch
    {
    case metal: __intrinsic_asm "fmax3";
    default:
    {
        return max(y, max(x, z));
    }
    }
}

__generic<T : __BuiltinFloatingPointType, let N: int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
vector<T,N> fmax3(vector<T,N> x, vector<T,N> y, vector<T,N> z)
{
    __target_switch
    {
    case metal: __intrinsic_asm "fmax3";
    default:
        VECTOR_MAP_TRINARY(T, N, fmax3, x, y, z);
    }
}

/// Minimum.
/// @param x The first value to compare.
/// @param y The second value to compare.
/// @return The smaller of the two values, element-wise if vector typed.
/// @remarks For HLSL, GLSL, and metal targets, this is implemented with the min() intrinsic.
/// For SPIR-V, it is implemented with the UMin or SMin instruction, depending on the signedness of the type.
/// @category math
__generic<T : __BuiltinIntegerType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T min(T x, T y)
{
    __target_switch
    {
    case hlsl:
    case glsl:
    case metal:
        __intrinsic_asm "min";
    case cuda:
    case cpp:
        __intrinsic_asm "$P_min($0, $1)";
    case spirv:
    {
        if (__isSignedInt<T>())
            return spirv_asm {
                result:$$T = OpExtInst glsl450 SMin $x $y
            };
        else
            return spirv_asm {
                result:$$T = OpExtInst glsl450 UMin $x $y
            };
    }
    case wgsl: __intrinsic_asm "min";
    }
}

__generic<T : __BuiltinIntegerType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> min(vector<T,N> x, vector<T,N> y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "min";
    case glsl: __intrinsic_asm "min";
    case metal: __intrinsic_asm "min";
    case spirv:
    {
        if (__isSignedInt<T>())
            return spirv_asm {
                result:$$vector<T,N> = OpExtInst glsl450 SMin $x $y
            };
        else
            return spirv_asm {
                result:$$vector<T,N> = OpExtInst glsl450 UMin $x $y
            };
    }
    case wgsl: __intrinsic_asm "min";
    default:
        VECTOR_MAP_BINARY(T, N, min, x, y);
    }
}

__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> min(matrix<T,N,M> x, matrix<T,N,M> y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "min";
    default:
        MATRIX_MAP_BINARY(T, N, M, min, x, y);
    }
}

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T min(T x, T y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "min";
    case metal: __intrinsic_asm "min";
    case glsl: __intrinsic_asm "min";
    case cuda: __intrinsic_asm "$P_min($0, $1)";
    case cpp: __intrinsic_asm "$P_min($0, $1)";
    case spirv: return spirv_asm {
        result:$$T = OpExtInst glsl450 FMin $x $y
    };
    case wgsl: __intrinsic_asm "min";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> min(vector<T,N> x, vector<T,N> y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "min";
    case metal: __intrinsic_asm "min";
    case glsl: __intrinsic_asm "min";
    case spirv: return spirv_asm {
        result:$$vector<T,N> = OpExtInst glsl450 FMin $x $y
    };
    case wgsl: __intrinsic_asm "min";
    default:
        VECTOR_MAP_BINARY(T, N, min, x, y);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> min(matrix<T,N,M> x, matrix<T,N,M> y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "min";
    default:
        MATRIX_MAP_BINARY(T, N, M, min, x, y);
    }
}

/// Minimum of 3 inputs.
/// @param x The first value to compare.
/// @param y The second value to compare.
/// @param z The third value to compare.
/// @return The smallest of the three values, element-wise if vector typed.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
T min3(T x, T y, T z)
{
    __target_switch
    {
    case metal: __intrinsic_asm "min3";
    default:
        return min(x, min(y, z));
    }
}

/// @category math
__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
vector<T,N> min3(vector<T,N> x, vector<T,N> y, vector<T,N> z)
{
    __target_switch
    {
    case metal: __intrinsic_asm "min3";
    default:
        return min(x, min(y, z));
    }
}

/// Floating-point minimum.
/// @param x The first value to compare.
/// @param y The second value to compare.
/// @return The smaller of the two values, element-wise if vector typed.
/// @remarks Result is `x` if `x` < `y`, either `x` or `y` if both `x` and `y` are zeros, otherwise `y`. Which operand is the result is undefined if one of the operands is a NaN.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
T fmin(T x, T y)
{
    __target_switch
    {
    case metal: __intrinsic_asm "fmin";
    default:
        return min(x, y);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
vector<T,N> fmin(vector<T,N> x, vector<T,N> y)
{
    __target_switch
    {
    case metal: __intrinsic_asm "fmin";
    default:
        VECTOR_MAP_BINARY(T, N, fmin, x, y);
    }
}

/// Floating-point minimum of 3 inputs.
/// @param x The first value to compare.
/// @param y The second value to compare.
/// @param z The third value to compare.
/// @return The smallest of the three values, element-wise if vector typed.
/// @remarks If any operand in the 3-way comparison is NaN, it is undefined which operand is returned.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
T fmin3(T x, T y, T z)
{
    __target_switch
    {
    case metal: __intrinsic_asm "fmin3";
    default:
    {
        return min(x, min(y, z));
    }
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
vector<T,N> fmin3(vector<T,N> x, vector<T,N> y, vector<T,N> z)
{
    __target_switch
    {
    case metal: __intrinsic_asm "fmin3";
    default:
        VECTOR_MAP_TRINARY(T, N, fmin3, x, y, z);
    }
}

/// Median of 3 values.
/// @param x The first value to compare.
/// @param y The second value to compare.
/// @param z The third value to compare.
/// @return The median of the three values, element-wise if vector typed.
/// @remarks For metal, this is implemented with the median3 intrinsic which has special handling for NaN.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
T median3(T x, T y, T z)
{
    __target_switch
    {
    case metal: __intrinsic_asm "median3";
    default:
    {
        //           | a | b | c | m |
        // ----------+---+---+---+---+
        // x > y > z | z | y | x | y |
        // x > z > y | y | z | x | z |
        // y > x > z | z | y | x | x |
        // y > z > x | z | y | z | z |
        // z > x > y | y | z | x | x |
        // z > y > x | y | z | y | y |

        T a = min(y, z);
        T b = max(y, z);
        T c = max(x, a);
        T m = min(b, c);
        return m;
    }
    }
}

/// @category math
__generic<T : __BuiltinFloatingPointType, let N: int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
vector<T,N> median3(vector<T,N> x, vector<T,N> y, vector<T,N> z)
{
    __target_switch
    {
    case metal: __intrinsic_asm "median3";
    default:
    {
        vector<T,N> a = min(y, z);
        vector<T,N> b = max(y, z);
        vector<T,N> c = max(x, a);
        vector<T,N> m = min(b, c);
        return m;
    }
    }
}

/// Floating-point median.
/// @param x The first value to compare.
/// @param y The second value to compare.
/// @param z The third value to compare.
/// @return The median of the three values, element-wise if vector typed.
/// @remarks For metal, this is implemented with the fmedian3 intrinsic.
/// If any value is NaN, it is unspecified which operand is returned.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
T fmedian3(T x, T y, T z)
{
    __target_switch
    {
    case metal: __intrinsic_asm "fmedian3";
    default:
    {
        return median3(x, y, z);
    }
    }
}

__generic<T : __BuiltinFloatingPointType, let N: int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
vector<T,N> fmedian3(vector<T,N> x, vector<T,N> y, vector<T,N> z)
{
    __target_switch
    {
    case metal: __intrinsic_asm "fmedian3";
    default:
        VECTOR_MAP_TRINARY(T, N, fmedian3, x, y, z);
    }
}

/// Split into integer and fractional parts (both with same sign).
/// @param x The input value.
/// @param[out] ip The integer part of `x`.
/// @return The fractional part of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T modf(T x, out T ip)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_modf($0, $1)";
    case cuda: __intrinsic_asm "$P_modf($0, $1)";
    case hlsl: __intrinsic_asm "modf";
    case glsl: __intrinsic_asm "modf";
    case metal: __intrinsic_asm "modf($0, *($1))";
    case spirv: return spirv_asm {
        result:$$T = OpExtInst glsl450 Modf $x &ip
    };
    case wgsl:
        T fract;
        __wgsl_modf<T>(x, fract, ip);
        return fract;
    }
}

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(wgsl)]
void __wgsl_modf(T x, out T fract, out T whole)
{
    __intrinsic_asm "{ var s = modf($0); ($1) = s.fract; ($2) = s.whole; }";
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> modf(vector<T,N> x, out vector<T,N> ip)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "modf";
    case glsl: __intrinsic_asm "modf";
    case metal: __intrinsic_asm "modf($0, *($1))";
    case spirv: return spirv_asm {
        result:$$vector<T,N> = OpExtInst glsl450 Modf $x &ip
    };
    case wgsl:
        vector<T,N> fract;
        __wgsl_modf<T>(x, fract, ip);
        return fract;
    default:
        VECTOR_MAP_BINARY(T, N, modf, x, ip);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[ForceInline]
[require(wgsl)]
void __wgsl_modf(vector<T,N> x, out vector<T,N> fract, out vector<T,N> whole)
{
    __intrinsic_asm "{ var s = modf($0); ($1) = s.fract; ($2) = s.whole; }";
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int, let L : int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> modf(matrix<T,N,M> x, out matrix<T,N,M,L> ip)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "modf";
    default:
        MATRIX_MAP_BINARY(T, N, M, modf, x, ip);
    }
}

/// Masked sum of absolute differences of byte alignments.
/// This function computes the absolute differences of the byte alignments of the reference and source values, and adds them to the accumulated differences.
/// @param reference The reference 4 bytes packed in a uint.
/// @param source The source 2 uints packed in a uint2.
/// @param accum The accumulated differences.
/// @return The updated accumulated differences.
/// @remarks In HLSL, this is implemented with the msad4 intrinsic.
/// @category math
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
uint4 msad4(uint reference, uint2 source, uint4 accum)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "msad4";
    default:
        int4 bytesRef = (reference >> uint4(24, 16, 8, 0)) & 0xFF;
        int4 bytesX   = (source.x  >> uint4(24, 16, 8, 0)) & 0xFF;
        int4 bytesY   = (source.y  >> uint4(24, 16, 8, 0)) & 0xFF;
    
        uint4 mask = select(bytesRef == 0, 0, 0xFFFFFFFFu);
    
        uint4 result = accum;
        result += mask.x & abs(bytesRef - int4(bytesX.x,           bytesY.y, bytesY.z, bytesY.w));
        result += mask.y & abs(bytesRef - int4(bytesX.x, bytesX.y,           bytesY.z, bytesY.w));
        result += mask.z & abs(bytesRef - int4(bytesX.x, bytesX.y, bytesX.z,           bytesY.w));
        result += mask.w & abs(bytesRef - int4(bytesX.x, bytesX.y, bytesX.z, bytesX.w));
        return result;
    }
}

// General inner products

// scalar-scalar
/// Multiply.
/// @param x The first value.
/// @param y The second value.
/// @return The inner product of `x` and `y`.
/// @category math
__generic<T : __BuiltinArithmeticType>
__intrinsic_op($(kIROp_Mul))
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
T mul(T x, T y);

// scalar-vector and vector-scalar
__generic<T : __BuiltinArithmeticType, let N : int>
__intrinsic_op($(kIROp_Mul))
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
vector<T, N> mul(vector<T, N> x, T y);

__generic<T : __BuiltinArithmeticType, let N : int>
__intrinsic_op($(kIROp_Mul))
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
vector<T, N> mul(T x, vector<T, N> y);

// scalar-matrix and matrix-scalar
__generic<T : __BuiltinArithmeticType, let N : int, let M :int>
__intrinsic_op($(kIROp_Mul))
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
matrix<T, N, M> mul(matrix<T, N, M> x, T y);

__generic<T : __BuiltinArithmeticType, let N : int, let M :int>
__intrinsic_op($(kIROp_Mul))
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
matrix<T, N, M> mul(T x, matrix<T, N, M> y);

// vector-vector (dot product)
/// @category math
__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T mul(vector<T, N> x, vector<T, N> y)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "dot";
    case metal: __intrinsic_asm "dot";
    case hlsl: __intrinsic_asm "mul";
    case wgsl: __intrinsic_asm "dot";
    default:
        return dot(x, y);
    }
}
__generic<T : __BuiltinIntegerType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T mul(vector<T, N> x, vector<T, N> y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "mul";
    default:
        return dot(x, y);
    }
}

// vector-matrix
__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, M> mul(vector<T, N> left, matrix<T, N, M> right)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "($1 * $0)";
    case metal: __intrinsic_asm "($1 * $0)";
    case hlsl: __intrinsic_asm "mul";
    case spirv: return spirv_asm {
        OpMatrixTimesVector $$vector<T, M> result $right $left
    };
    case wgsl: __intrinsic_asm "($1 * $0)";
    default:
        vector<T,M> result;
        for( int j = 0; j < M; ++j )
        {
            T sum = T(0);
            for( int i = 0; i < N; ++i )
            {
                sum += left[i] * right[i][j];
            }
            result[j] = sum;
        }
        return result;
    }
}
__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, M> mul(vector<T, N> left, matrix<T, N, M> right)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "($1 * $0)";
    case metal: __intrinsic_asm "($1 * $0)";
    case hlsl: __intrinsic_asm "mul";
    case wgsl: __intrinsic_asm "($1 * $0)";
    default:
        vector<T,M> result;
        for( int j = 0; j < M; ++j )
        {
            T sum = T(0);
            for( int i = 0; i < N; ++i )
            {
                sum += left[i] * right[i][j];
            }
            result[j] = sum;
        }
        return result;
    }
}
__generic<T : __BuiltinLogicalType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, M> mul(vector<T, N> left, matrix<T, N, M> right)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "($1 * $0)";
    case metal: __intrinsic_asm "($1 * $0)";
    case hlsl: __intrinsic_asm "mul";
    case wgsl: __intrinsic_asm "($1 * $0)";
    default:
        vector<T,M> result;
        for( int j = 0; j < M; ++j )
        {
            T sum = T(0);
            for( int i = 0; i < N; ++i )
            {
                sum |= left[i] & right[i][j];
            }
            result[j] = sum;
        }
        return result;
    }
}

// matrix-vector
__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> mul(matrix<T,N,M> left, vector<T,M> right)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "($1 * $0)";
    case metal: __intrinsic_asm "($1 * $0)";
    case hlsl: __intrinsic_asm "mul";
    case spirv: return spirv_asm {
        OpVectorTimesMatrix $$vector<T,N> result $right $left
    };
    case wgsl: __intrinsic_asm "($1 * $0)";
    default:
        vector<T,N> result;
        for( int i = 0; i < N; ++i )
        {
            T sum = T(0);
            for( int j = 0; j < M; ++j )
            {
                sum += left[i][j] * right[j];
            }
            result[i] = sum;
        }
        return result;
    }
}
__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> mul(matrix<T,N,M> left, vector<T,M> right)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "($1 * $0)";
    case metal: __intrinsic_asm "($1 * $0)";
    case hlsl: __intrinsic_asm "mul";
    case wgsl: __intrinsic_asm "($1 * $0)";
    default:
        vector<T,N> result;
        for( int i = 0; i < N; ++i )
        {
            T sum = T(0);
            for( int j = 0; j < M; ++j )
            {
                sum += left[i][j] * right[j];
            }
            result[i] = sum;
        }
        return result;
    }
}
__generic<T : __BuiltinLogicalType, let N : int, let M : int>
[__readNone]
[OverloadRank(-1)]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> mul(matrix<T,N,M> left, vector<T,M> right)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "($1 * $0)";
    case metal: __intrinsic_asm "($1 * $0)";
    case hlsl: __intrinsic_asm "mul";
    case wgsl: __intrinsic_asm "($1 * $0)";
    default:
        vector<T,N> result;
        for( int i = 0; i < N; ++i )
        {
            T sum = T(0);
            for( int j = 0; j < M; ++j )
            {
                sum |= left[i][j] & right[j];
            }
            result[i] = sum;
        }
        return result;
    }
}

// matrix-matrix
__generic<T : __BuiltinFloatingPointType, let R : int, let N : int, let C : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,R,C> mul(matrix<T,R,N> left, matrix<T,N,C> right)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "($1 * $0)";
    case metal: __intrinsic_asm "($1 * $0)";
    case hlsl: __intrinsic_asm "mul";
    case spirv: return spirv_asm {
        OpMatrixTimesMatrix $$matrix<T,R,C> result $right $left
    };
    case wgsl: __intrinsic_asm "($1 * $0)";
    default:
        matrix<T,R,C> result;
        for( int r = 0; r < R; ++r)
        for( int c = 0; c < C; ++c)
        {
            T sum = T(0);
            for( int i = 0; i < N; ++i )
            {
                sum += left[r][i] * right[i][c];
            }
            result[r][c] = sum;
        }
        return result;
    }
}
__generic<T : __BuiltinIntegerType, let R : int, let N : int, let C : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,R,C> mul(matrix<T,R,N> left, matrix<T,N,C> right)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "($1 * $0)";
    case metal: __intrinsic_asm "($1 * $0)";
    case hlsl: __intrinsic_asm "mul";
    case wgsl: __intrinsic_asm "($1 * $0)";
    default:
        matrix<T,R,C> result;
        for( int r = 0; r < R; ++r)
        for( int c = 0; c < C; ++c)
        {
            T sum = T(0);
            for( int i = 0; i < N; ++i )
            {
                sum += left[r][i] * right[i][c];
            }
            result[r][c] = sum;
        }
        return result;
    }
}
__generic<T : __BuiltinLogicalType, let R : int, let N : int, let C : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,R,C> mul(matrix<T,R,N> left, matrix<T,N,C> right)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "($1 * $0)";
    case metal: __intrinsic_asm "($1 * $0)";
    case hlsl: __intrinsic_asm "mul";
    case wgsl: __intrinsic_asm "($1 * $0)";
    default:
        matrix<T,R,C> result;
        for( int r = 0; r < R; ++r)
        for( int c = 0; c < C; ++c)
        {
            T sum = T(0);
            for( int i = 0; i < N; ++i )
            {
                sum |= left[r][i] & right[i][c];
            }
            result[r][c] = sum;
        }
        return result;
    }
}

// next-after: next representable floating-point value
// after x in the direction of y

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, shader5_sm_4_0)]
T nextafter(T x, T y)
{
    __target_switch
    {
    case metal: __intrinsic_asm "nextafter";
    default:
        if (isnan(x)) return x;
        if (isnan(y)) return y;
        if (x == y) return y;
        if (T is half)
        {
            T delta = __realCast<T>(bit_cast<half>(uint16_t(1)));
            return x + ((x < y) ? delta : -delta);
        }
        if (T is float)
        {
            T delta = __realCast<T>(bit_cast<float>(uint32_t(1)));
            return x + ((x < y) ? delta : -delta);
        }
        T delta = __realCast<T>(bit_cast<double>(uint64_t(1)));
        return x + ((x < y) ? delta : -delta);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv, shader5_sm_4_0)]
vector<T,N> nextafter(vector<T,N> x, vector<T,N> y)
{
    __target_switch
    {
    case metal: __intrinsic_asm "nextafter";
    default:
        VECTOR_MAP_BINARY(T, N, nextafter, x, y);
    }
}

/// Indicate that an index may be non-uniform at execution time.
///
/// Shader Model 5.1 and 6.x introduce support for dynamic indexing
/// of arrays of resources, but place the restriction that *by default*
/// the implementation can assume that any value used as an index into
/// such arrays will be dynamically uniform across an entire `Draw` or `Dispatch`
/// (when using instancing, the value must be uniform across all instances;
/// it does not seem that the restriction extends to draws within a multi-draw).
///
/// In order to indicate to the implementation that it cannot make the
/// uniformity assumption, a shader programmer is required to pass the index
/// to the `NonUniformResourceIndex` function before using it as an index.
/// The function superficially acts like an identity function.
///
/// Note: a future version of Slang may take responsibility for inserting calls
/// to this function as necessary in output code, rather than make this
/// the user's responsibility, so that the default behavior of the language
/// is more semantically "correct."
[ForceInline]
[require(spirv)]
T __copyObject<T>(T v)
{
    __target_switch {
    case spirv:
        return spirv_asm {
           result:$$T = OpCopyObject $v;
        };
   }
}

/// `NonUniformResourceIndex` function is used to indicate if the resource index is
/// divergent, and ensure scalarization happens correctly for each divergent lane.
__generic<T:__BuiltinArithmeticType>
__intrinsic_op($(kIROp_NonUniformResourceIndex))
[require(cpp_cuda_glsl_hlsl_spirv, nonuniformqualifier)]
T NonUniformResourceIndex(T index);

/// HLSL allows NonUniformResourceIndex around non int/uint types.
/// It's effect is presumably to ignore it, which the following implementation does.
/// We should also look to add a warning for this scenario.
/// @deprecated
[__unsafeForceInlineEarly]
[deprecated("NonUniformResourceIndex on a type other than uint/int is deprecated and has no effect")]
T NonUniformResourceIndex<T>(T value) { return value; }

/// Normalize a vector.
/// @param x The vector to normalize.
/// @return The normalized vector, `x`/`length(x)`.
/// @category math
__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> normalize(vector<T,N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "normalize";
    case hlsl: __intrinsic_asm "normalize";
    case metal: __intrinsic_asm "normalize";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,N> result glsl450 Normalize $x
    };
    case wgsl: __intrinsic_asm "normalize";
    default:
        return x / length(x);
    }
}

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T normalize(T x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "normalize";
    case hlsl: __intrinsic_asm "normalize";
    case metal: __intrinsic_asm "normalize";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Normalize $x
    };
    case wgsl: __intrinsic_asm "normalize";
    default:
        return x / length(x);
    }
}

/// Raise to a power.
/// @param x The base value.
/// @param y The exponent value.
/// @return The value of `x` raised to the power of `y`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T pow(T x, T y)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_pow($0, $1)";
    case cuda: __intrinsic_asm "$P_pow($0, $1)";
    case glsl: __intrinsic_asm "pow";
    case hlsl: __intrinsic_asm "pow";
    case metal: __intrinsic_asm "pow";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Pow $x $y
    };
    case wgsl: __intrinsic_asm "pow";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> pow(vector<T, N> x, vector<T, N> y)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "pow";
    case hlsl: __intrinsic_asm "pow";
    case metal: __intrinsic_asm "pow";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Pow $x $y
    };
    case wgsl: __intrinsic_asm "pow";
    default:
        VECTOR_MAP_BINARY(T, N, pow, x, y);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> pow(matrix<T,N,M> x, matrix<T,N,M> y)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "pow";
    default:
        MATRIX_MAP_BINARY(T, N, M, pow, x, y);
    }
}

/// Raise positive base value to a power.
/// @param x The base value, must be >= 0.
/// @param y The exponent value.
/// @return The value of `x` raised to the power of `y`.
/// @category math
/// @remarks Return value is undefined for non-positive values of `x`.
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T powr(T x, T y)
{
    __target_switch
    {
    case metal: __intrinsic_asm "powr";
    default:
        return pow(x, y);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> powr(vector<T, N> x, vector<T, N> y)
{
    __target_switch
    {
    case metal: __intrinsic_asm "powr";
    default:
        return pow(x, y);
    }
}

// Output message
// TODO: add check to ensure format is const literal.

/// Print a message to the debug output.
/// @param T The variadic type pack parameter for the arguments to be printed.
/// @param format The format string.
/// @param args (optional) The arguments to be printed.
/// @remarks The function maps to `printf` for HLSL, CPU and CUDA targets, and maps to `OpDebugPrintf` for SPIR-V target,
/// and maps to `debugPrintfEXT` for GLSL target. Depending on the target and execution environment, the function may have
/// no effect.
/// @example
/// ```cpp
/// void test(int x, float y)
/// {
///     printf("hello world!\n");
///     printf(R"(x = "%d", y = "%f")", x, y);
/// }
/// ```
[require(cpp_cuda_glsl_hlsl_spirv, printf)]
[require(slangvm)]
__intrinsic_op($(kIROp_Printf))
void printf<each T>(NativeString format, expand each T args);

// Tessellation factor fixup routines
/// @category tessellation Tessellation functions
[require(hlsl, sm_5_0)]
void Process2DQuadTessFactorsAvg(
    in  float4 RawEdgeFactors,
    in  float2 InsideScale,
    out float4 RoundedEdgeTessFactors,
    out float2 RoundedInsideTessFactors,
    out float2 UnroundedInsideTessFactors);

/// @category tessellation
[require(hlsl, sm_5_0)]
void Process2DQuadTessFactorsMax(
    in  float4 RawEdgeFactors,
    in  float2 InsideScale,
    out float4 RoundedEdgeTessFactors,
    out float2 RoundedInsideTessFactors,
    out float2 UnroundedInsideTessFactors);

/// @category tessellation
[require(hlsl, sm_5_0)]
void Process2DQuadTessFactorsMin(
    in  float4 RawEdgeFactors,
    in  float2 InsideScale,
    out float4 RoundedEdgeTessFactors,
    out float2 RoundedInsideTessFactors,
    out float2 UnroundedInsideTessFactors);

/// @category tessellation
[require(hlsl, sm_5_0)]
void ProcessIsolineTessFactors(
    in  float RawDetailFactor,
    in  float RawDensityFactor,
    out float RoundedDetailFactor,
    out float RoundedDensityFactor);

/// @category tessellation
[require(hlsl, sm_5_0)]
void ProcessQuadTessFactorsAvg(
    in  float4 RawEdgeFactors,
    in  float InsideScale,
    out float4 RoundedEdgeTessFactors,
    out float2 RoundedInsideTessFactors,
    out float2 UnroundedInsideTessFactors);

/// @category tessellation
[require(hlsl, sm_5_0)]
void ProcessQuadTessFactorsMax(
    in  float4 RawEdgeFactors,
    in  float InsideScale,
    out float4 RoundedEdgeTessFactors,
    out float2 RoundedInsideTessFactors,
    out float2 UnroundedInsideTessFactors);

/// @category tessellation
[require(hlsl, sm_5_0)]
void ProcessQuadTessFactorsMin(
    in  float4 RawEdgeFactors,
    in  float InsideScale,
    out float4 RoundedEdgeTessFactors,
    out float2 RoundedInsideTessFactors,
    out float2 UnroundedInsideTessFactors);

/// @category tessellation
[require(hlsl, sm_5_0)]
void ProcessTriTessFactorsAvg(
    in  float3 RawEdgeFactors,
    in  float InsideScale,
    out float3 RoundedEdgeTessFactors,
    out float RoundedInsideTessFactor,
    out float UnroundedInsideTessFactor);

/// @category tessellation
[require(hlsl, sm_5_0)]
void ProcessTriTessFactorsMax(
    in  float3 RawEdgeFactors,
    in  float InsideScale,
    out float3 RoundedEdgeTessFactors,
    out float RoundedInsideTessFactor,
    out float UnroundedInsideTessFactor);

/// @category tessellation
[require(hlsl, sm_5_0)]
void ProcessTriTessFactorsMin(
    in  float3 RawEdgeFactors,
    in  float InsideScale,
    out float3 RoundedEdgeTessFactors,
    out float RoundedInsideTessFactors,
    out float UnroundedInsideTessFactors);

/// Convert degrees to radians.
/// @param x The angle in degrees.
/// @return The angle in radians.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T radians(T x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "radians";
    case hlsl: __intrinsic_asm "radians";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Radians $x
    };
    case wgsl: __intrinsic_asm "radians";
    default:
        return x * (T.getPi() / T(180.0f));
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> radians(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "radians";
    case hlsl: __intrinsic_asm "radians";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Radians $x
    };
    case wgsl: __intrinsic_asm "radians";
    default:
        return x * (T.getPi() / T(180.0f));
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> radians(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "radians";
    default:
        return x * (T.getPi() / T(180.0f));
    }
}

/// Compute approximate reciprocal of `x`.
/// @param x The value to compute the reciprocal of.
/// @return The approximate reciprocal of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T rcp(T x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "rcp";
    default:
        return T(1.0) / x;
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> rcp(vector<T, N> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "rcp";
    case glsl:
    case spirv:
    case wgsl:
        return T(1.0) / x;
    default:
        VECTOR_MAP_UNARY(T, N, rcp, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> rcp(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "rcp";
    default:
        MATRIX_MAP_UNARY(T, N, M, rcp, x);
    }
}

/// Reflect incident vector across plane with given normal.
/// @param i The incident vector.
/// @param n The normal vector.
/// @return The reflected vector.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T reflect(T i, T n)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "reflect";
    case hlsl: __intrinsic_asm "reflect";
    case metal: __intrinsic_asm "reflect";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Reflect $i $n
    };
    case wgsl: __intrinsic_asm "reflect";
    default:
        return i - T(2) * dot(n,i) * n;
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> reflect(vector<T,N> i, vector<T,N> n)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "reflect";
    case hlsl: __intrinsic_asm "reflect";
    case metal: __intrinsic_asm "reflect";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,N> result glsl450 Reflect $i $n
    };
    case wgsl: __intrinsic_asm "reflect";
    default:
        return i - T(2) * dot(n,i) * n;
    }
}

/// Refract incident vector given surface normal and index of refraction.
/// @param i The incident vector.
/// @param n The normal vector.
/// @param eta The relative refractive index.
/// @category math
__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> refract(vector<T,N> i, vector<T,N> n, T eta)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "refract";
    case hlsl: __intrinsic_asm "refract";
    case metal: __intrinsic_asm "refract";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,N> result glsl450 Refract $i $n $eta
    };
    case wgsl: __intrinsic_asm "refract";
    default:
        let dotNI = dot(n,i);
        let k = T(1) - eta*eta*(T(1) - dotNI * dotNI);
        if(k < T(0)) return vector<T,N>(T(0));
        return eta * i - (eta * dotNI + sqrt(k)) * n;
    }
}

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T refract(T i, T n, T eta)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "refract";
    case hlsl: __intrinsic_asm "refract";
    case metal: __intrinsic_asm "refract";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Refract $i $n $eta
    };
    case wgsl: __intrinsic_asm "refract";
    default:
        let dotNI = dot(n,i);
        let k = T(1) - eta*eta*(T(1) - dotNI * dotNI);
        if(k < T(0)) return T(0);
        return eta * i - (eta * dotNI + sqrt(k)) * n;
    }
}

/// Reverse order of bits.
/// @param value The value to reverse bits of.
/// @return The bits of `value`, reversed such that bit n of the result is equal to bit (width - 1 - n) of `value`.
/// @remarks For SPIR-V, this function maps to `OpBitReverse`.
/// @category bitops
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
uint reversebits(uint value)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "reversebits";
    case glsl:
        __intrinsic_asm "bitfieldReverse";
    case cuda:
    case cpp:
        __intrinsic_asm "$P_reversebits($0)";
    case metal:
        __intrinsic_asm "reverse_bits";
    case spirv:
        return spirv_asm {OpBitReverse $$uint result $value};
    case wgsl: __intrinsic_asm "reverseBits";
    }
}

__generic<let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, shader5_sm_5_0)]
vector<uint, N> reversebits(vector<uint, N> value)
{
    __target_switch
    {
    default:
        VECTOR_MAP_UNARY(uint, N, reversebits, value);
    case glsl:
        __intrinsic_asm "bitfieldReverse";
    case metal:
        __intrinsic_asm "reverse_bits";
    case spirv:
        return spirv_asm {OpBitReverse $$vector<uint, N> result $value};
    case wgsl: __intrinsic_asm "reverseBits";
    }
}

/// Round even.
/// @param x The value to round.
/// @return The value rounded to the nearest integer, with ties rounded to the nearest even integer.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
T rint(T x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "roundEven";
    case metal: __intrinsic_asm "rint";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 RoundEven $x
    };
    default:
        T nearest = round(x);

        // Check if the value is exactly halfway between two integers
        if (abs(x - nearest) == T(0.5))
        {
            // If halfway, choose the even number
            if ((nearest / T(2)) * T(2) != nearest)
            {
                // If the nearest number is odd,
                // move to the closest even number
                nearest -= ((x < nearest) ? T(1) : T(-1));
            }
        }
        return nearest;
    }
}

__generic<T : __BuiltinFloatingPointType, let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
vector<T,N> rint(vector<T,N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "roundEven";
    case metal: __intrinsic_asm "rint";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,N> result glsl450 RoundEven $x
    };
    default:
        VECTOR_MAP_UNARY(T, N, rint, x);
    }
}

/// Round-to-nearest.
/// @param x The value to round.
/// @return The value rounded to the nearest integer.
/// @remarks Rounding behavior of .5 is determined by target intrinsic.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T round(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_round($0)";
    case cuda: __intrinsic_asm "$P_round($0)";
    case glsl: __intrinsic_asm "round";
    case hlsl: __intrinsic_asm "round";
    case metal: __intrinsic_asm "round";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Round $x
    };
    case wgsl: __intrinsic_asm "round";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> round(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "round";
    case hlsl: __intrinsic_asm "round";
    case metal: __intrinsic_asm "round";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Round $x
    };
    case wgsl: __intrinsic_asm "round";
    default:
        VECTOR_MAP_UNARY(T, N, round, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> round(matrix<T,N,M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "round";
    default:
        MATRIX_MAP_UNARY(T, N, M, round, x);
    }
}

/// Reciprocal of square root.
/// @param x The value to compute the reciprocal square root of.
/// @return The reciprocal square root of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T rsqrt(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_rsqrt($0)";
    case cuda: __intrinsic_asm "$P_rsqrt($0)";
    case glsl: __intrinsic_asm "inversesqrt($0)";
    case hlsl: __intrinsic_asm "rsqrt";
    case metal: __intrinsic_asm "rsqrt";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 InverseSqrt $x
    };
    default:
        return T(1.0) / sqrt(x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> rsqrt(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "inversesqrt($0)";
    case hlsl: __intrinsic_asm "rsqrt";
    case metal: __intrinsic_asm "rsqrt";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 InverseSqrt $x
    };
    default:
        VECTOR_MAP_UNARY(T, N, rsqrt, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> rsqrt(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "rsqrt";
    default:
        MATRIX_MAP_UNARY(T, N, M, rsqrt, x);
    }
}

/// Clamp value to [0,1] range.
/// @param x The value to clamp.
/// @return The clamped value.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T saturate(T x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "saturate";
    case metal: __intrinsic_asm "saturate";
    case wgsl: __intrinsic_asm "saturate";
    default:
        return clamp<T>(x, T(0), T(1));
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> saturate(vector<T,N> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "saturate";
    case metal: __intrinsic_asm "saturate";
    case wgsl: __intrinsic_asm "saturate";
    default:
        return clamp<T,N>(x,
            vector<T,N>(T(0)),
            vector<T,N>(T(1)));
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> saturate(matrix<T,N,M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "saturate";
    default:
        MATRIX_MAP_UNARY(T, N, M, saturate, x);
    }
}

__generic<T:__BuiltinArithmeticType, U:__BuiltinArithmeticType>
__intrinsic_op($(kIROp_IntCast))
T __int_cast(U val);

__generic<T:__BuiltinArithmeticType, U:__BuiltinArithmeticType>
__intrinsic_op($(kIROp_FloatCast))
T __real_cast(U val);

__generic<T:__BuiltinArithmeticType, U:__BuiltinArithmeticType>
__intrinsic_op($(kIROp_CastIntToFloat))
T __int_to_float_cast(U val);

__generic<T:__BuiltinArithmeticType, U:__BuiltinArithmeticType>
__intrinsic_op($(kIROp_CastFloatToInt))
T __float_to_int_cast(U val);

__generic<T:__BuiltinArithmeticType, U:__BuiltinArithmeticType>
[__unsafeForceInlineEarly]
T __arithmetic_cast(U val)
{
    if (__isFloat<T>() && __isInt<U>())
        return __int_to_float_cast<T>(val);
    else if (__isInt<T>() && __isFloat<U>())
        return __float_to_int_cast<T>(val);
    else if (__isFloat<T>() && __isFloat<U>())
        return __real_cast<T>(val);
    else if (__isInt<T>() && __isInt<U>())
        return __int_cast<T>(val);
    return T(0);
}

__generic<T:__BuiltinArithmeticType, U:__BuiltinArithmeticType, let N : int>
__intrinsic_op($(kIROp_IntCast))
vector<T,N> __int_cast(vector<U,N> val);

__generic<T:__BuiltinArithmeticType, U:__BuiltinArithmeticType, let N : int>
__intrinsic_op($(kIROp_FloatCast))
vector<T,N> __real_cast(vector<U,N> val);

__generic<T:__BuiltinArithmeticType, U:__BuiltinArithmeticType, let N : int>
__intrinsic_op($(kIROp_CastIntToFloat))
vector<T,N> __int_to_float_cast(vector<U,N> val);

__generic<T:__BuiltinArithmeticType, U:__BuiltinArithmeticType, let N : int>
__intrinsic_op($(kIROp_CastFloatToInt))
vector<T,N> __float_to_int_cast(vector<U,N> val);


/// Extract sign of value.
/// @param x The value to extract the sign of.
/// @return -1 if `x` is negative, 0 if `x` is zero, and 1 if `x` is positive.
/// @category math Math functions
__generic<T : __BuiltinSignedArithmeticType>
[__readNone]
int sign(T x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "sign";
    case metal: __intrinsic_asm "int(sign($0))";
    case glsl: __intrinsic_asm "int(sign($0))";
    case cuda:
    case cpp:
        __intrinsic_asm "$P_sign($0)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm
            {
                %fsign:$$T = OpExtInst glsl450 FSign $x;
                result:$$int = OpConvertFToS %fsign
            };
        else
            return __int_cast<int>(spirv_asm {OpExtInst $$T result glsl450 SSign $x});
    case wgsl: __intrinsic_asm "sign";
    }
}

__generic<T : __BuiltinSignedArithmeticType, let N : int>
[__readNone]
vector<int, N> sign(vector<T, N> x)
{
    if(N == 1)
        return vector<int, N>(sign(x[0]));
    __target_switch
    {
    case hlsl: __intrinsic_asm "sign";
    case glsl: __intrinsic_asm "ivec$N0(sign($0))";
    case metal: __intrinsic_asm "vec<int,$N0>(sign($0))";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm
            {
                %fsign:$$vector<T, N> = OpExtInst glsl450 FSign $x;
                result:$$vector<int, N> = OpConvertFToS %fsign
            };
        else
            return __int_cast<int>(spirv_asm {OpExtInst $$vector<T,N> result glsl450 SSign $x});
    case wgsl: __intrinsic_asm "sign";
    default:
        VECTOR_MAP_UNARY(int, N, sign, x);
    }
}

__generic<T : __BuiltinSignedArithmeticType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv_wgsl, sm_4_0_version)]
matrix<int, N, M> sign(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "sign";
    default:
        MATRIX_MAP_UNARY(int, N, M, sign, x);
    }
}

/// Sine.
/// @param x The angle in radians.
/// @return The sine of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T sin(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_sin($0)";
    case cuda: __intrinsic_asm "$P_sin($0)";
    case glsl: __intrinsic_asm "sin";
    case hlsl: __intrinsic_asm "sin";
    case metal: __intrinsic_asm "sin";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Sin $x
    };
    case wgsl: __intrinsic_asm "sin";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> sin(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "sin";
    case hlsl: __intrinsic_asm "sin";
    case metal: __intrinsic_asm "sin";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Sin $x
    };
    case wgsl: __intrinsic_asm "sin";
    default:
        VECTOR_MAP_UNARY(T, N, sin, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> sin(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "sin";
    default:
        MATRIX_MAP_UNARY(T, N, M, sin, x);
    }
}

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(metal)]
T __sincos_metal(T x, out T c)
{
    __target_switch
    {
    case metal: __intrinsic_asm "sincos($0, *$1)";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(metal)]
vector<T,N> __sincos_metal(vector<T,N> x, out vector<T,N> c)
{
    __target_switch
    {
    case metal: __intrinsic_asm "sincos($0, *$1)";
    }
}

/// Sine and cosine.
/// Calculate both the sine and cosine of `x`.
/// @param x The angle in radians.
/// @param[out] s The sine of `x`.
/// @param[out] c The cosine of `x`.
/// @return void
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
void sincos(T x, out T s, out T c)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "$P_sincos($0, $1, $2)";
    case hlsl: __intrinsic_asm "sincos";
    case metal:
        //__intrinsic_asm "*($1) = sincos($0, *($2))";
        s = __sincos_metal(x, c);
        return;
    default:
        s = sin(x);
        c = cos(x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
void sincos(vector<T,N> x, out vector<T,N> s, out vector<T,N> c)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "sincos";
    case metal:
        //__intrinsic_asm "*($1) = sincos($0, *($2))";
        s = __sincos_metal(x, c);
        return;
    default:
        s = sin(x);
        c = cos(x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int, let L1: int, let L2 : int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
void sincos(matrix<T,N,M> x, out matrix<T,N,M,L1> s, out matrix<T,N,M,L2> c)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "sincos";
    default:
        s = sin(x);
        c = cos(x);
    }
}

/// Hyperbolic sine.
/// @param x The value to compute the hyperbolic sine of.
/// @return The hyperbolic sine of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T sinh(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_sinh($0)";
    case cuda: __intrinsic_asm "$P_sinh($0)";
    case glsl: __intrinsic_asm "sinh";
    case hlsl: __intrinsic_asm "sinh";
    case metal: __intrinsic_asm "sinh";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Sinh $x
    };
    case wgsl: __intrinsic_asm "sinh";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> sinh(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "sinh";
    case hlsl: __intrinsic_asm "sinh";
    case metal: __intrinsic_asm "sinh";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Sinh $x
    };
    case wgsl: __intrinsic_asm "sinh";
    default:
        VECTOR_MAP_UNARY(T, N, sinh, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> sinh(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "sinh";
    default:
        MATRIX_MAP_UNARY(T, N, M, sinh, x);
    }
}

/// Compute the sine of `x * pi`.
/// @param x The value to compute the sine of.
/// @return The sine of `x * pi`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T sinpi(T x)
{
    __target_switch
    {
    case metal: __intrinsic_asm "sinpi";
    default:
        return sin(T.getPi() * x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> sinpi(vector<T,N> x)
{
    __target_switch
    {
    case metal: __intrinsic_asm "sinpi";
    default:
        return sin(T.getPi() * x);
    }
}


/// Smooth step (Hermite interpolation).
/// @param min The lower edge of the interpolation range.
/// @param max The upper edge of the interpolation range.
/// @param x The value to interpolate.
/// @return 0 if `x` is less than `min`, 1 if `x` is greater than `max`, and a smooth interpolation between 0 and 1 otherwise.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T smoothstep(T min, T max, T x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "smoothstep";
    case hlsl: __intrinsic_asm "smoothstep";
    case metal: __intrinsic_asm "smoothstep";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 SmoothStep $min $max $x
    };
    case wgsl: __intrinsic_asm "smoothstep";
    default:
        let t = saturate((x - min) / (max - min));
        return t * t * (T(3.0f) - (t + t));
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> smoothstep(vector<T, N> min, vector<T, N> max, vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "smoothstep";
    case hlsl: __intrinsic_asm "smoothstep";
    case metal: __intrinsic_asm "smoothstep";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 SmoothStep $min $max $x
    };
    case wgsl: __intrinsic_asm "smoothstep";
    default:
        VECTOR_MAP_TRINARY(T, N, smoothstep, min, max, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> smoothstep(matrix<T, N, M> min, matrix<T, N, M> max, matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "smoothstep";
    default:
        MATRIX_MAP_TRINARY(T, N, M, smoothstep, min, max, x);
    }
}

/// Compute the square root of `x`.
/// @param x The value to compute the square root of.
/// @return The square root of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T sqrt(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_sqrt($0)";
    case cuda: __intrinsic_asm "$P_sqrt($0)";
    case glsl: __intrinsic_asm "sqrt";
    case hlsl: __intrinsic_asm "sqrt";
    case metal: __intrinsic_asm "sqrt";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Sqrt $x
    };
    case wgsl: __intrinsic_asm "sqrt";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> sqrt(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "sqrt";
    case hlsl: __intrinsic_asm "sqrt";
    case metal: __intrinsic_asm "sqrt";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Sqrt $x
    };
    case wgsl: __intrinsic_asm "sqrt";
    default:
        VECTOR_MAP_UNARY(T, N, sqrt, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> sqrt(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "sqrt";
    default:
        MATRIX_MAP_UNARY(T, N, M, sqrt, x);
    }
}

/// Step function.
/// @param y The threshold value.
/// @param x The value to compare against the threshold.
/// @return 0 if `x` is less than `y`, and 1 otherwise.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T step(T y, T x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "step";
    case hlsl: __intrinsic_asm "step";
    case metal: __intrinsic_asm "step";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Step $y $x
    };
    case wgsl: __intrinsic_asm "step";
    default:
        return x < y ? T(0.0f) : T(1.0f);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> step(vector<T,N> y, vector<T,N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "step";
    case hlsl: __intrinsic_asm "step";
    case metal: __intrinsic_asm "step";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,N> result glsl450 Step $y $x
    };
    case wgsl: __intrinsic_asm "step";
    default:
        VECTOR_MAP_BINARY(T, N, step, y, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> step(matrix<T, N, M> y, matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "step";
    default:
        MATRIX_MAP_BINARY(T, N, M, step, y, x);
    }
}

/// Compute the tangent of `x`.
/// @param x The angle in radians.
/// @return The tangent of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T tan(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_tan($0)";
    case cuda: __intrinsic_asm "$P_tan($0)";
    case glsl: __intrinsic_asm "tan";
    case hlsl: __intrinsic_asm "tan";
    case metal: __intrinsic_asm "tan";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Tan $x
    };
    case wgsl: __intrinsic_asm "tan";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> tan(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "tan";
    case hlsl: __intrinsic_asm "tan";
    case metal: __intrinsic_asm "tan";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Tan $x
    };
    case wgsl: __intrinsic_asm "tan";
    default:
        VECTOR_MAP_UNARY(T, N, tan, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> tan(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "tan";
    default:
        MATRIX_MAP_UNARY(T, N, M, tan, x);
    }
}

/// Compute the hyperbolic tangent of `x`.
/// @param x The value to compute the hyperbolic tangent of, in radians.
/// @return The hyperbolic tangent of `x`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T tanh(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_tanh($0)";
    case cuda: __intrinsic_asm "$P_tanh($0)";
    case glsl: __intrinsic_asm "tanh";
    case hlsl: __intrinsic_asm "tanh";
    case metal: __intrinsic_asm "tanh";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Tanh $x
    };
    case wgsl: __intrinsic_asm "tanh";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> tanh(vector<T,N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "tanh";
    case hlsl: __intrinsic_asm "tanh";
    case metal: __intrinsic_asm "tanh";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T,N> result glsl450 Tanh $x
    };
    case wgsl: __intrinsic_asm "tanh";
    default:
        VECTOR_MAP_UNARY(T, N, tanh, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T,N,M> tanh(matrix<T,N,M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "tanh";
    default:
        MATRIX_MAP_UNARY(T, N, M, tanh, x);
    }
}

/// Compute the tangent of `x * pi`.
/// @param x The value to compute the tangent of.
/// @return The tangent of `x * pi`.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T tanpi(T x)
{
    __target_switch
    {
    case metal: __intrinsic_asm "tanpi";
    default:
        return tan(T.getPi() * x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T,N> tanpi(vector<T,N> x)
{
    __target_switch
    {
    case metal: __intrinsic_asm "tanpi";
    default:
        return tan(T.getPi() * x);
    }
}


/// Matrix transpose.
/// @param x The matrix to transpose.
/// @return The transposed matrix.
/// @category math
__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv_wgsl, sm_4_0_version)]
[PreferRecompute]
matrix<T, M, N> transpose(matrix<T, N, M> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "transpose";
    case hlsl: __intrinsic_asm "transpose";
    case spirv: return spirv_asm {
        OpTranspose $$matrix<T, M, N> result $x
    };
    case wgsl: __intrinsic_asm "transpose";
    default:
        matrix<T,M,N> result;
        for(int r = 0; r < M; ++r)
            for(int c = 0; c < N; ++c)
                result[r][c] = x[c][r];
        return result;
    }
}
__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv_wgsl, sm_4_0_version)]
[PreferRecompute]
matrix<T, M, N> transpose(matrix<T, N, M> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "transpose";
    case hlsl: __intrinsic_asm "transpose";
    case spirv: return spirv_asm {
        OpTranspose $$matrix<T, M, N> result $x
    };
    case wgsl: __intrinsic_asm "transpose";
    default:
        matrix<T, M, N> result;
        for (int r = 0; r < M; ++r)
            for (int c = 0; c < N; ++c)
                result[r][c] = x[c][r];
        return result;
    }
}
__generic<T : __BuiltinLogicalType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv_wgsl, sm_4_0_version)]
[PreferRecompute]
[OverloadRank(-1)]
matrix<T, M, N> transpose(matrix<T, N, M> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "transpose";
    case hlsl: __intrinsic_asm "transpose";
    case spirv: return spirv_asm {
        OpTranspose $$matrix<T, M, N> result $x
    };
    case wgsl: __intrinsic_asm "transpose";
    default:
        matrix<T, M, N> result;
        for (int r = 0; r < M; ++r)
            for (int c = 0; c < N; ++c)
                result[r][c] = x[c][r];
        return result;
    }
}

/// Truncate to integer.
/// @category math
__generic<T : __BuiltinFloatingPointType>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
T trunc(T x)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "$P_trunc($0)";
    case cuda: __intrinsic_asm "$P_trunc($0)";
    case glsl: __intrinsic_asm "trunc";
    case hlsl: __intrinsic_asm "trunc";
    case metal: __intrinsic_asm "trunc";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 Trunc $x
    };
    case wgsl: __intrinsic_asm "trunc";
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
vector<T, N> trunc(vector<T, N> x)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "trunc";
    case hlsl: __intrinsic_asm "trunc";
    case metal: __intrinsic_asm "trunc";
    case spirv: return spirv_asm {
        OpExtInst $$vector<T, N> result glsl450 Trunc $x
    };
    case wgsl: __intrinsic_asm "trunc";
    default:
        VECTOR_MAP_UNARY(T, N, trunc, x);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_4_0_version)]
matrix<T, N, M> trunc(matrix<T, N, M> x)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "trunc";
    default:
        MATRIX_MAP_UNARY(T, N, M, trunc, x);
    }
}

// Slang Specific 'Mask' Wave Intrinsics

//@hidden:
typedef uint WaveMask;

__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_ballot)]
WaveMask WaveGetConvergedMask()
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupBallot(true).x";
    case hlsl:
        __intrinsic_asm "WaveActiveBallot(true).x";
    case cuda:
        __intrinsic_asm "__activemask()";
    case spirv:
        let _true = true;
        return (spirv_asm
        {
            OpCapability GroupNonUniformBallot;
            OpGroupNonUniformBallot $$uint4 result Subgroup $_true
        }).x;
    }
}

__intrinsic_op($(kIROp_WaveGetActiveMask))
WaveMask __WaveGetActiveMask();

__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_ballot_activemask)]
WaveMask WaveGetActiveMask()
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupBallot(true).x";
    case hlsl:
        __intrinsic_asm "WaveActiveBallot(true).x";
    case spirv:
        let _true = true;
        return (spirv_asm
        {
            OpCapability GroupNonUniformBallot;
            OpGroupNonUniformBallot $$uint4 result Subgroup $_true
        }).x;
    case cuda:
        return __WaveGetActiveMask();
    }
}

__glsl_extension(GL_KHR_shader_subgroup_basic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_basic)]
bool WaveMaskIsFirstLane(WaveMask mask)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupElect()";
    case cuda:
        __intrinsic_asm "(($0 & -$0) == (WarpMask(1) << _getLaneId()))";
    case hlsl:
        __intrinsic_asm "WaveIsFirstLane()";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformBallot;
            OpGroupNonUniformElect $$bool result Subgroup
        };
    }
}

__glsl_extension(GL_KHR_shader_subgroup_vote)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_vote)]
bool WaveMaskAllTrue(WaveMask mask, bool condition)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupAll($1)";
    case cuda:
        __intrinsic_asm "(__all_sync($0, $1) != 0)";
    case hlsl:
        __intrinsic_asm "WaveActiveAllTrue($1)";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformBallot;
            OpGroupNonUniformAll $$bool result Subgroup $condition
        };
    }
}

__glsl_extension(GL_KHR_shader_subgroup_vote)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_vote)]
bool WaveMaskAnyTrue(WaveMask mask, bool condition)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupAny($1)";
    case cuda:
        __intrinsic_asm "(__any_sync($0, $1) != 0)";
    case hlsl:
        __intrinsic_asm "WaveActiveAnyTrue($1)";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformBallot;
            OpGroupNonUniformAny $$bool result Subgroup $condition
        };
    }
}

__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_ballot)]
WaveMask WaveMaskBallot(WaveMask mask, bool condition)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupBallot($1).x";
    case cuda:
        __intrinsic_asm "__ballot_sync($0, $1)";
    case hlsl:
        __intrinsic_asm "WaveActiveBallot($1)";
    case spirv:
        return (spirv_asm
        {
            OpCapability GroupNonUniformBallot;
            OpGroupNonUniformBallot $$uint4 result Subgroup $condition
        }).x;
    }
}

[require(cuda_glsl_hlsl_spirv, subgroup_basic_ballot)]
uint WaveMaskCountBits(WaveMask mask, bool value)
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm  "__popc(__ballot_sync($0, $1))";
    case hlsl:
        __intrinsic_asm  "WaveActiveCountBits($1)";
    default:
        return _WaveCountBits(WaveActiveBallot(value));
    }
}

// Waits until all warp lanes named in mask have executed a WaveMaskSharedSync (with the same mask)
// before resuming execution. Guarantees memory ordering in shared memory among threads participating
// in the barrier.
//
// The CUDA intrinsic says it orders *all* memory accesses, which appears to match most closely subgroupBarrier.
//
// TODO(JS):
// For HLSL it's not clear what to do. There is no explicit mechanism to 'reconverge' threads. In the docs it describes
// behavior as
// "These intrinsics are dependent on active lanes and therefore flow control. In the model of this document, implementations
// must enforce that the number of active lanes exactly corresponds to the programmer’s view of flow control."
//
// It seems this can only mean the active threads are the "threads the program flow would lead to". This implies a lockstep
// "straight SIMD" style interpretation. That being the case this op on HLSL is just a memory barrier without any Sync.

[require(cuda_glsl_hlsl_spirv, memorybarrier)]
void AllMemoryBarrierWithWaveMaskSync(WaveMask mask)
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__syncwarp($0)";
    case hlsl:
        __intrinsic_asm "AllMemoryBarrier()";
    case glsl:
    case spirv:
        __subgroupBarrier();
        return;
    }
}

// On GLSL, it appears we can't use subgroupMemoryBarrierShared, because it only implies a memory ordering, it does not
// imply convergence. For subgroupBarrier we have from the docs..
// "The function subgroupBarrier() enforces that all active invocations within a subgroup must execute this function before any
// are allowed to continue their execution"
// TODO(JS):
// It's not entirely clear what to do here on HLSL.
// Reading the dxc wiki (https://github.com/Microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics), we have statements like:
//    ... these intrinsics enable the elimination of barrier constructs when the scope of synchronization is within the width of the SIMD processor.
//    Wave: A set of lanes executed simultaneously in the processor. No explicit barriers are required to guarantee that they execute in parallel.
// Which seems to imply at least some memory barriers like Shared might not be needed.
//
// The barrier is left here though, because not only is the barrier make writes before the barrier across the wave appear to others afterwards, it's
// also there to inform the compiler on what order reads and writes can take place. This might seem to be silly because of the 'Active' lanes
// aspect of HLSL seems to make everything in lock step - but that's not quite so, it only has to apparently be that way as far as the programmers
// model appears - divergence could perhaps potentially still happen.

[require(cuda_glsl_hlsl_spirv, memorybarrier)]
void GroupMemoryBarrierWithWaveMaskSync(WaveMask mask)
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__syncwarp($0)";
    case hlsl:
        __intrinsic_asm "GroupMemoryBarrier()";
    case glsl:
    case spirv:
        __subgroupBarrier();
        return;
    }
}

[require(cuda_glsl_hlsl_spirv, memorybarrier)]
void AllMemoryBarrierWithWaveSync()
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__syncwarp()";
    case hlsl:
        __intrinsic_asm "AllMemoryBarrier()";
    case glsl:
    case spirv:
        __subgroupBarrier();
        return;
    }
}

[require(cuda_glsl_hlsl_spirv, memorybarrier)]
void GroupMemoryBarrierWithWaveSync()
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__syncwarp()";
    case hlsl:
        __intrinsic_asm "GroupMemoryBarrier()";
    case glsl:
    case spirv:
        __subgroupBarrier();
        return;
    }
}

// NOTE! WaveMaskBroadcastLaneAt is *NOT* standard HLSL
// It is provided as access to subgroupBroadcast which can only take a
// constexpr laneId.
// https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_shader_subgroup.txt
// Versions SPIR-V greater than 1.4 loosen this restriction, and allow 'dynamic uniform' index
// If that's the behavior required then client code should use WaveReadLaneAt which works this way.

__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_ballot)]
T WaveMaskBroadcastLaneAt(WaveMask mask, T value, constexpr int lane)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupBroadcast($1, $2)";
    case cuda: __intrinsic_asm "__shfl_sync($0, $1, $2)";
    case hlsl: __intrinsic_asm "WaveReadLaneAt($1, $2)";
    case spirv:
        let ulane = uint(lane);
        return spirv_asm {
            OpCapability GroupNonUniformBallot;
            OpGroupNonUniformBroadcast $$T result Subgroup $value $ulane;
        };
    }
}

__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_ballot)]
vector<T,N> WaveMaskBroadcastLaneAt(WaveMask mask, vector<T,N> value, constexpr int lane)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupBroadcast($1, $2)";
    case cuda: __intrinsic_asm "_waveShuffleMultiple($0, $1, $2)";
    case hlsl: __intrinsic_asm "WaveReadLaneAt($1, $2)";
    case spirv:
        let ulane = uint(lane);
        return spirv_asm {
            OpCapability GroupNonUniformBallot;
            OpGroupNonUniformBroadcast $$vector<T,N> result Subgroup $value $ulane;
        };
    }
}
__generic<T : __BuiltinType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_ballot)]
matrix<T,N,M> WaveMaskBroadcastLaneAt(WaveMask mask, matrix<T,N,M> value, constexpr int lane)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_waveShuffleMultiple($0, $1, $2)";
    case hlsl: __intrinsic_asm "WaveReadLaneAt($1, $2)";
    }
}

// TODO(JS): If it can be determines that the `laneId` is constExpr, then subgroupBroadcast
// could be used on GLSL. For now we just use subgroupShuffle
__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_shuffle)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_shuffle)]
T WaveMaskReadLaneAt(WaveMask mask, T value, int lane)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupShuffle($1, $2)";
    case cuda: __intrinsic_asm "__shfl_sync($0, $1, $2)";
    case hlsl: __intrinsic_asm "WaveReadLaneAt($1, $2)";
    case spirv:
        let ulane = uint(lane);
        return spirv_asm {
            OpCapability GroupNonUniformShuffle;
            OpGroupNonUniformShuffle $$T result Subgroup $value $ulane;
        };
    }
}
__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_shuffle)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_shuffle)]
vector<T,N> WaveMaskReadLaneAt(WaveMask mask, vector<T,N> value, int lane)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupShuffle($1, $2)";
    case cuda: __intrinsic_asm "_waveShuffleMultiple($0, $1, $2)";
    case hlsl: __intrinsic_asm "WaveReadLaneAt($1, $2)";
    case spirv:
        let ulane = uint(lane);
        return spirv_asm {
            OpCapability GroupNonUniformShuffle;
            OpGroupNonUniformShuffle $$vector<T,N> result Subgroup $value $ulane;
        };
    }
}
__generic<T : __BuiltinType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_shuffle)]
matrix<T,N,M> WaveMaskReadLaneAt(WaveMask mask, matrix<T,N,M> value, int lane)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_waveShuffleMultiple($0, $1, $2)";
    case hlsl: __intrinsic_asm "WaveReadLaneAt($1, $2)";
    }
}

// NOTE! WaveMaskShuffle is a NON STANDARD HLSL intrinsic! It will map to WaveReadLaneAt on HLSL
// which means it will only work on hardware which allows arbitrary laneIds which is not true
// in general because it breaks the HLSL standard, which requires it's 'dynamically uniform' across the Wave.
__generic<T : __BuiltinType>
[__unsafeForceInlineEarly]
T WaveMaskShuffle(WaveMask mask, T value, int lane)
{
    return WaveMaskReadLaneAt(mask, value, lane);
}
__generic<T : __BuiltinType, let N : int>
[__unsafeForceInlineEarly]
vector<T,N> WaveMaskShuffle(WaveMask mask, vector<T,N> value, int lane)
{
    return WaveMaskReadLaneAt(mask, value, lane);
}
__generic<T : __BuiltinType, let N : int, let M : int>
[__unsafeForceInlineEarly]
matrix<T,N,M> WaveMaskShuffle(WaveMask mask, matrix<T,N,M> value, int lane)
{
    return WaveMaskReadLaneAt(mask, value, lane);
}

__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_ballot)]
uint WaveMaskPrefixCountBits(WaveMask mask, bool value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupBallotExclusiveBitCount(subgroupBallot($1))";
    case cuda: __intrinsic_asm "__popc(__ballot_sync($0, $1)  & _getLaneLtMask())";
    case hlsl: __intrinsic_asm "WavePrefixCountBits($1)";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformBallot;
            %mask:$$uint4 = OpGroupNonUniformBallot Subgroup $value;
            OpGroupNonUniformBallotBitCount $$uint result Subgroup 2 %mask
        };
    }
}

// Across lane ops

__generic<T : __BuiltinIntegerType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
T WaveMaskBitAnd(WaveMask mask, T expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupAnd($1)";
    case cuda: __intrinsic_asm "_waveAnd($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveBitAnd($1)";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformArithmetic;
            OpGroupNonUniformBitwiseAnd $$T result Subgroup 0 $expr
        };
    }
}

__generic<T : __BuiltinIntegerType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
vector<T,N> WaveMaskBitAnd(WaveMask mask, vector<T,N> expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupAnd($1)";
    case cuda: __intrinsic_asm "_waveAndMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveBitAnd($1)";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformArithmetic;
            OpGroupNonUniformBitwiseAnd $$vector<T,N> result Subgroup 0 $expr
        };
    }
}
__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_arithmetic)]
matrix<T,N,M> WaveMaskBitAnd(WaveMask mask, matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_waveAndMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveBitAnd($1)";
    }
}

__generic<T : __BuiltinIntegerType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
T WaveMaskBitOr(WaveMask mask, T expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupOr($1)";
    case cuda: __intrinsic_asm "_waveOr($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveBitOr($1)";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformArithmetic;
            OpGroupNonUniformBitwiseOr $$T result Subgroup 0 $expr
        };
    }
}
__generic<T : __BuiltinIntegerType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
vector<T,N> WaveMaskBitOr(WaveMask mask, vector<T,N> expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupOr($1)";
    case cuda: __intrinsic_asm "_waveOrMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveBitOr($1)";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformArithmetic;
            OpGroupNonUniformBitwiseOr $$vector<T,N> result Subgroup 0 $expr
        };
    }
}
__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_arithmetic)]
matrix<T,N,M> WaveMaskBitOr(WaveMask mask, matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_waveOrMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveBitOr($1)";
    }
}

__generic<T : __BuiltinIntegerType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
T WaveMaskBitXor(WaveMask mask, T expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupXor($1)";
    case cuda: __intrinsic_asm "_waveXor($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveBitXor($1)";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformArithmetic;
            OpGroupNonUniformBitwiseXor $$T result Subgroup 0 $expr
        };
    }
}
__generic<T : __BuiltinIntegerType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
vector<T,N> WaveMaskBitXor(WaveMask mask, vector<T,N> expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupXor($1)";
    case cuda: __intrinsic_asm "_waveXorMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveBitXor($1)";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformArithmetic;
            OpGroupNonUniformBitwiseXor $$vector<T,N> result Subgroup 0 $expr
        };
    }
}
__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_arithmetic)]
matrix<T,N,M> WaveMaskBitXor(WaveMask mask, matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_waveXorMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveBitXor($1)";
    }
}

__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
T WaveMaskMax(WaveMask mask, T expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupMax($1)";
    case cuda: __intrinsic_asm "_waveMax($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveMax($1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMax $$T result Subgroup 0 $expr};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformSMax $$T result Subgroup 0 $expr};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformUMax $$T result Subgroup 0 $expr};
        else return expr;
    }
}
__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
vector<T,N> WaveMaskMax(WaveMask mask, vector<T,N> expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupMax($1)";
    case cuda: __intrinsic_asm "_waveMaxMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveMax($1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMax $$vector<T,N> result Subgroup 0 $expr};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformSMax $$vector<T,N> result Subgroup 0 $expr};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformUMax $$vector<T,N> result Subgroup 0 $expr};
        else return expr;
    }
}

__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_arithmetic)]
matrix<T,N,M> WaveMaskMax(WaveMask mask, matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_waveMaxMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveMax($1)";
    }
}

__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
T WaveMaskMin(WaveMask mask, T expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupMin($1)";
    case cuda: __intrinsic_asm "_waveMin($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveMin($1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMin $$T result Subgroup 0 $expr};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformSMin $$T result Subgroup 0 $expr};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformUMin $$T result Subgroup 0 $expr};
        else return expr;
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
vector<T,N> WaveMaskMin(WaveMask mask, vector<T,N> expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupMin($1)";
    case cuda: __intrinsic_asm "_waveMinMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveMin($1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMin $$vector<T,N>  result Subgroup 0 $expr};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformSMin $$vector<T,N>  result Subgroup 0 $expr};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformUMin $$vector<T,N>  result Subgroup 0 $expr};
        else return expr;
    }
}

__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_arithmetic)]
matrix<T,N,M> WaveMaskMin(WaveMask mask, matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_waveMinMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveMin($1)";
    }
}

__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
T WaveMaskProduct(WaveMask mask, T expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupMul($1)";
    case cuda: __intrinsic_asm "_waveProduct($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveProduct($1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMul $$T result Subgroup 0 $expr};
        else if (__isInt<T>())
        {
            return spirv_asm
            {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformIMul $$T result Subgroup 0 $expr;
            };
        }
        else return expr;
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
vector<T,N> WaveMaskProduct(WaveMask mask, vector<T,N> expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupMul($1)";
    case cuda: __intrinsic_asm "_waveProductMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveProduct($1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMul $$vector<T,N> result Subgroup 0 $expr};
        else if (__isInt<T>())
        {
            return spirv_asm
            {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformIMul $$vector<T,N> result Subgroup 0 $expr;
            };
        }
        else return expr;
    }
}

__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_arithmetic)]
matrix<T,N,M> WaveMaskProduct(WaveMask mask, matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_waveProductMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveProduct($1)";
    }
}

__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
T WaveMaskSum(WaveMask mask, T expr)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupAdd($1)";
    case cuda: __intrinsic_asm "_waveSum($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveSum($1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFAdd $$T result Subgroup 0 $expr};
        else if (__isInt<T>())
        {
            return spirv_asm
            {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformIAdd $$T result Subgroup 0 $expr;
            };
        }
        else return expr;
    }
}
__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
vector<T,N> WaveMaskSum(WaveMask mask, vector<T,N> expr)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupAdd($1)";
    case cuda: __intrinsic_asm "_waveSumMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveSum($1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFAdd $$vector<T,N> result Subgroup 0 $expr};
        else if (__isInt<T>())
        {
            return spirv_asm
            {
                OpCapability GroupNonUniformArithmetic; 
                OpGroupNonUniformIAdd $$vector<T,N> result Subgroup 0 $expr;
            };
        }
        else return expr;
    }
}
__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_arithmetic)]
matrix<T,N,M> WaveMaskSum(WaveMask mask, matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_waveSumMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveSum($1)";
    }
}

__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_vote)
__spirv_version(1.3)
__cuda_sm_version(7.0)
[require(cuda_glsl_hlsl_spirv, subgroup_vote)]
bool WaveMaskAllEqual(WaveMask mask, T value)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupAllEqual($1)";
    case hlsl:
        __intrinsic_asm "WaveActiveAllEqual($1)";
    case cuda:
        __intrinsic_asm "_waveAllEqual($0, $1)";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformVote;
            OpGroupNonUniformAllEqual $$bool result Subgroup $value
        };
    }
}
__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_vote)
__spirv_version(1.3)
__cuda_sm_version(7.0)
[require(cuda_glsl_hlsl_spirv, subgroup_vote)]
bool WaveMaskAllEqual(WaveMask mask, vector<T,N> value)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupAllEqual($1)";
    case hlsl:
        __intrinsic_asm "WaveActiveAllEqual($1)";
    case cuda:
        __intrinsic_asm "_waveAllEqualMultiple($0, $1)";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformVote;
            OpGroupNonUniformAllEqual $$bool result Subgroup $value
        };
    }
}
__generic<T : __BuiltinType, let N : int, let M : int>
__cuda_sm_version(7.0)
[require(cuda_hlsl, subgroup_vote)]
bool WaveMaskAllEqual(WaveMask mask, matrix<T,N,M> value)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_waveAllEqualMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveActiveAllEqual($1)";
    }
}

// Prefix

__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
T WaveMaskPrefixProduct(WaveMask mask, T expr)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupExclusiveMul($1)";
    case cuda: __intrinsic_asm "_wavePrefixProduct($0, $1)";
    case hlsl: __intrinsic_asm "WavePrefixProduct($1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMul $$T result Subgroup ExclusiveScan $expr};
        else if (__isInt<T>())
        {
            return spirv_asm
            {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformIMul $$T result Subgroup ExclusiveScan $expr;
            };
        }
        else return expr;
    }
}
__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
vector<T,N> WaveMaskPrefixProduct(WaveMask mask, vector<T,N> expr)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupExclusiveMul($1)";
    case cuda: __intrinsic_asm "_wavePrefixProductMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WavePrefixProduct($1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMul $$vector<T,N> result Subgroup ExclusiveScan $expr};
        else if (__isInt<T>())
        {
            return spirv_asm
            {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformIMul $$vector<T,N> result Subgroup ExclusiveScan $expr;
            };
        }
        else return expr;
    }
}
__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_arithmetic)]
matrix<T,N,M> WaveMaskPrefixProduct(WaveMask mask, matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixProductMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WavePrefixProduct($1)";
    }
}

__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
T WaveMaskPrefixSum(WaveMask mask, T expr)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupExclusiveAdd($1)";
    case cuda: __intrinsic_asm "_wavePrefixSum($0, $1)";
    case hlsl: __intrinsic_asm "WavePrefixSum($1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFAdd $$T result Subgroup ExclusiveScan $expr};
        else if (__isInt<T>())
        {
            return spirv_asm
            {
                OpCapability GroupNonUniformArithmetic;
                result:$$T = OpGroupNonUniformIAdd Subgroup ExclusiveScan $expr;
            };
        }
        else return expr;
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
vector<T,N> WaveMaskPrefixSum(WaveMask mask, vector<T,N> expr)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupExclusiveAdd($1)";
    case cuda: __intrinsic_asm "_wavePrefixSumMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WavePrefixSum($1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFAdd $$vector<T,N> result Subgroup ExclusiveScan $expr};
        else if (__isInt<T>())
        {
            return spirv_asm
            {
                OpCapability GroupNonUniformArithmetic;
                result:$$vector<T,N> = OpGroupNonUniformIAdd Subgroup ExclusiveScan $expr;
            };
        }
        else return expr;
    }
}
__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_arithmetic)]
matrix<T,N,M> WaveMaskPrefixSum(WaveMask mask, matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixSumMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WavePrefixSum($1)";
    }
}

__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_ballot)]
T WaveMaskReadLaneFirst(WaveMask mask, T expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupBroadcastFirst($1)";
    case cuda: __intrinsic_asm "_waveReadFirst($0, $1)";
    case hlsl: __intrinsic_asm "WaveReadLaneFirst($1)";
    case spirv:
        return spirv_asm {OpCapability GroupNonUniformBallot; OpGroupNonUniformBroadcastFirst $$T result Subgroup $expr};
    }
}
__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_ballot)]
vector<T,N> WaveMaskReadLaneFirst(WaveMask mask, vector<T,N> expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupBroadcastFirst($1)";
    case cuda: __intrinsic_asm "_waveReadFirstMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveReadLaneFirst($1)";
    case spirv:
        return spirv_asm {OpCapability GroupNonUniformBallot; OpGroupNonUniformBroadcastFirst $$vector<T,N> result Subgroup $expr};
    }
}

__generic<T : __BuiltinType, let N : int, let M : int>
[require(cuda, subgroup_ballot)]
matrix<T,N,M> WaveMaskReadLaneFirst(WaveMask mask, matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_waveReadFirstMultiple($0, $1)";
    }
}

// WaveMask SM6.5 like intrinsics

// TODO(JS): On HLSL it only works for 32 bits or less

__generic<T : __BuiltinType>
__glsl_extension(GL_NV_shader_subgroup_partitioned)
__spirv_version(1.1)
__cuda_sm_version(7.0)
[require(cuda_glsl_hlsl_spirv, subgroup_partitioned)]
WaveMask WaveMaskMatch(WaveMask mask, T value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupPartitionNV($1).x";
    case cuda: __intrinsic_asm "_waveMatchScalar($0, $1).x";
    case hlsl: __intrinsic_asm "WaveMatch($1).x";
    case spirv:
        return (spirv_asm
        {
            OpCapability GroupNonUniformPartitionedNV;
            OpExtension "SPV_NV_shader_subgroup_partitioned";
            OpGroupNonUniformPartitionNV $$uint4 result $value
        }).x;
    }
}
__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_NV_shader_subgroup_partitioned)
__spirv_version(1.1)
__cuda_sm_version(7.0)
[require(cuda_glsl_hlsl_spirv, subgroup_partitioned)]
WaveMask WaveMaskMatch(WaveMask mask, vector<T,N> value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupPartitionNV($1).x";
    case cuda: __intrinsic_asm "_waveMatchMultiple($0, $1).x";
    case hlsl: __intrinsic_asm "WaveMatch($1).x";
    case spirv:
        return (spirv_asm
        {
            OpCapability GroupNonUniformPartitionedNV;
            OpExtension "SPV_NV_shader_subgroup_partitioned";
            OpGroupNonUniformPartitionNV $$uint4 result $value
        }).x;
    }
}

__generic<T : __BuiltinType, let N : int, let M : int>
__glsl_extension(GL_NV_shader_subgroup_partitioned)
__spirv_version(1.3)
__cuda_sm_version(7.0)
[require(cuda_glsl_hlsl, subgroup_partitioned)]
WaveMask WaveMaskMatch(WaveMask mask, matrix<T,N,M> value)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_waveMatchMultiple($0, $1)";
    case glsl: __intrinsic_asm "subgroupPartitionNV($1).x";
    case hlsl: __intrinsic_asm "WaveMatch($1).x";
    }
}

__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
T WaveMaskPrefixBitAnd(WaveMask mask, T expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupExclusiveAnd($1)";
    case cuda: __intrinsic_asm "_wavePrefixAnd($0, $1)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitAnd($1, uint4($0, 0, 0, 0))";
    case spirv:
        return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseAnd $$T result Subgroup ExclusiveScan $expr};
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
vector<T,N> WaveMaskPrefixBitAnd(WaveMask mask, vector<T,N> expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupExclusiveAnd($1)";
    case cuda: __intrinsic_asm "_wavePrefixAndMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitAnd($1, uint4($0, 0, 0, 0))";
    case spirv:
        return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseAnd $$vector<T,N> result Subgroup ExclusiveScan $expr};
    }
}

__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_arithmetic)]
matrix<T,N,M> WaveMaskPrefixBitAnd(WaveMask mask, matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixAndMultiple(_getMultiPrefixMask($0, $1)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitAnd($1, uint4($0, 0, 0, 0))";
    }
}

__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
T WaveMaskPrefixBitOr(WaveMask mask, T expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupExclusiveOr($1)";
    case cuda: __intrinsic_asm "_wavePrefixOr($0, $1)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitOr($1, uint4($0, 0, 0, 0))";
    case spirv:
        return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseAnd $$T result Subgroup ExclusiveScan $expr};
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
vector<T,N> WaveMaskPrefixBitOr(WaveMask mask, vector<T,N> expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupExclusiveOr($1)";
    case cuda: __intrinsic_asm "_wavePrefixOrMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitOr($1, uint4($0, 0, 0, 0))";
    case spirv:
        return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseOr $$vector<T,N> result Subgroup ExclusiveScan $expr};
    }
}

__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_arithmetic)]
matrix<T,N,M> WaveMaskPrefixBitOr(WaveMask mask, matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixOrMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitOr($1, uint4($0, 0, 0, 0))";
    }
}

__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
T WaveMaskPrefixBitXor(WaveMask mask, T expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupExclusiveXor($1)";
    case cuda: __intrinsic_asm "_wavePrefixXor($0, $1)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitXor($1, uint4($0, 0, 0, 0))";
    case spirv:
        return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseXor $$T result Subgroup ExclusiveScan $expr};
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_arithmetic)]
vector<T,N> WaveMaskPrefixBitXor(WaveMask mask, vector<T,N> expr)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupExclusiveXor($1)";
    case cuda: __intrinsic_asm "_wavePrefixXorMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitXor($1, uint4($0, 0, 0, 0))";
    case spirv:
        return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseXor $$vector<T,N> result Subgroup ExclusiveScan $expr};
    }
}

__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_arithmetic)]
matrix<T,N,M> WaveMaskPrefixBitXor(WaveMask mask, matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixXorMultiple($0, $1)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitXor($1, uint4($0, 0, 0, 0))";
    }
}

//@public:

// Shader model 6.0 stuff

// Information for GLSL wave/subgroup support
// https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_shader_subgroup.txt

/// @category wave
__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_quad)
__spirv_version(1.3)
[require(glsl_hlsl_metal_spirv, subgroup_quad)]
T QuadReadLaneAt(T sourceValue, uint quadLaneID)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "QuadReadLaneAt";
    case glsl:
        __intrinsic_asm "subgroupQuadBroadcast";
    case metal:
        // TODO: Need to add intrinsics to access Metal and WGSL's broadcast variant where lane is const for all threads.
        __intrinsic_asm "quad_shuffle($0, ushort($1))";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformQuad;
            result:$$T = OpGroupNonUniformQuadBroadcast Subgroup $sourceValue $quadLaneID;
        };
    }
}
__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_quad)
__spirv_version(1.3)
[require(glsl_hlsl_metal_spirv, subgroup_quad)]
vector<T,N> QuadReadLaneAt(vector<T,N> sourceValue, uint quadLaneID)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "QuadReadLaneAt";
    case glsl:
        __intrinsic_asm "subgroupQuadBroadcast";
    case metal:
        __intrinsic_asm "quad_shuffle($0, ushort($1))";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformQuad;
            result:$$vector<T,N> = OpGroupNonUniformQuadBroadcast Subgroup $sourceValue $quadLaneID;
        };
    }
}
__generic<T : __BuiltinType, let N : int, let M : int> matrix<T,N,M> QuadReadLaneAt(matrix<T,N,M> sourceValue, uint quadLaneID);

/// @category wave
__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_quad)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_hlsl_metal_spirv_wgsl, subgroup_quad)]
T QuadReadAcrossX(T localValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "QuadReadAcrossX";
    case glsl: __intrinsic_asm "subgroupQuadSwapHorizontal($0)";
    case metal:
        return QuadReadLaneAt(localValue, __builtinQuadLaneIndex ^ 1U);
    case spirv:
        uint direction = 0u;
        return spirv_asm 
        {
            OpCapability GroupNonUniformQuad;
            result:$$T = OpGroupNonUniformQuadSwap Subgroup $localValue $direction;
        };
    case wgsl: __intrinsic_asm "quadSwapX";
    }
}

__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_quad)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_hlsl_metal_spirv_wgsl, subgroup_quad)]
vector<T,N> QuadReadAcrossX(vector<T,N> localValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "QuadReadAcrossX";
    case glsl: __intrinsic_asm "subgroupQuadSwapHorizontal($0)";
    case metal:
        return QuadReadLaneAt(localValue, __builtinQuadLaneIndex ^ 1U);
    case spirv:
        uint direction = 0u;
        return spirv_asm 
        {
            OpCapability GroupNonUniformQuad;
            result:$$vector<T,N> = OpGroupNonUniformQuadSwap Subgroup $localValue $direction;
        };
    case wgsl: __intrinsic_asm "quadSwapX";
    }
}
__generic<T : __BuiltinType, let N : int, let M : int> matrix<T,N,M> QuadReadAcrossX(matrix<T,N,M> localValue);

/// @category wave
__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_quad)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_hlsl_metal_spirv_wgsl, subgroup_quad)]
T QuadReadAcrossY(T localValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "QuadReadAcrossY";
    case glsl: __intrinsic_asm "subgroupQuadSwapVertical($0)";
    case metal:
        return QuadReadLaneAt(localValue, __builtinQuadLaneIndex ^ 2U);
    case spirv:
        uint direction = 1u;
        return spirv_asm 
        {
            OpCapability GroupNonUniformQuad;
            result:$$T = OpGroupNonUniformQuadSwap Subgroup $localValue $direction;
        };
    case wgsl: __intrinsic_asm "quadSwapY";
    }
}
__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_quad)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_hlsl_metal_spirv_wgsl, subgroup_quad)]
vector<T,N> QuadReadAcrossY(vector<T,N> localValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "QuadReadAcrossY";
    case glsl: __intrinsic_asm "subgroupQuadSwapVertical($0)";
    case metal:
        return QuadReadLaneAt(localValue, __builtinQuadLaneIndex ^ 2U);
    case spirv:
        uint direction = 1u;
        return spirv_asm 
        {
            OpCapability GroupNonUniformQuad;
            result:$$vector<T,N> = OpGroupNonUniformQuadSwap Subgroup $localValue $direction;
        };
    case wgsl: __intrinsic_asm "quadSwapY";
    }
}
__generic<T : __BuiltinType, let N : int, let M : int> matrix<T,N,M> QuadReadAcrossY(matrix<T,N,M> localValue);

/// @category wave
__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_quad)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_hlsl_metal_spirv_wgsl, subgroup_quad)]
T QuadReadAcrossDiagonal(T localValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "QuadReadAcrossDiagonal";
    case glsl: __intrinsic_asm "subgroupQuadSwapDiagonal($0)";
    case metal:
        return QuadReadLaneAt(localValue, __builtinQuadLaneIndex ^ 3U);
    case spirv:
        uint direction = 2u;
        return spirv_asm 
        {
            OpCapability GroupNonUniformQuad;
            result:$$T = OpGroupNonUniformQuadSwap Subgroup $localValue $direction;
        };
    case wgsl: __intrinsic_asm "quadSwapDiagonal";
    }
}
__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_quad)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_hlsl_metal_spirv_wgsl, subgroup_quad)]
vector<T,N> QuadReadAcrossDiagonal(vector<T,N> localValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "QuadReadAcrossDiagonal";
    case glsl: __intrinsic_asm "subgroupQuadSwapDiagonal($0)";
    case metal:
        return QuadReadLaneAt(localValue, __builtinQuadLaneIndex ^ 3U);
    case spirv:
        uint direction = 2u;
        return spirv_asm 
        {
            OpCapability GroupNonUniformQuad;
            result:$$vector<T,N> = OpGroupNonUniformQuadSwap Subgroup $localValue $direction;
        };
    case wgsl: __intrinsic_asm "quadSwapDiagonal";
    }
}
__generic<T : __BuiltinType, let N : int, let M : int> matrix<T,N,M> QuadReadAcrossDiagonal(matrix<T,N,M> localValue);

// WaveActiveBitAnd, WaveActiveBitOr, WaveActiveBitXor
${{{{
struct WaveActiveBitOpEntry { const char* hlslName; const char* glslName; const char* spirvName; const char* metalName; };
const WaveActiveBitOpEntry kWaveActiveBitOpEntries[] = {{"BitAnd", "And", "BitwiseAnd", "and"}, {"BitOr", "Or", "BitwiseOr", "or"}, {"BitXor", "Xor", "BitwiseXor", "xor"}};
for (auto opName : kWaveActiveBitOpEntries) {
}}}}
/// @category wave Wave and quad functions
__generic<T : __BuiltinIntegerType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
T WaveActive$(opName.hlslName)(T expr)
{
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroup$(opName.glslName)";
    case hlsl:
        __intrinsic_asm "WaveActive$(opName.hlslName)";
    case metal:
        __intrinsic_asm "simd_$(opName.metalName)";
    case spirv:
        return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniform$(opName.spirvName) $$T result Subgroup Reduce $expr};
    case cuda:
        return WaveMask$(opName.hlslName)(WaveGetActiveMask(), expr);
    }
}

__generic<T : __BuiltinIntegerType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
vector<T, N> WaveActive$(opName.hlslName)(vector<T, N> expr)
{
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroup$(opName.glslName)";
    case hlsl:
        __intrinsic_asm "WaveActive$(opName.hlslName)";
    case metal:
        __intrinsic_asm "simd_$(opName.metalName)";
    case spirv:
        return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniform$(opName.spirvName) $$vector<T, N> result Subgroup Reduce $expr};
    case cuda:
        return WaveMask$(opName.hlslName)(WaveGetActiveMask(), expr);
    }
}

__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
matrix<T, N, M> WaveActive$(opName.hlslName)(matrix<T, N, M> expr)
{
    __target_switch
    {
    case cuda:
        return WaveMask$(opName.hlslName)(WaveGetActiveMask(), expr);
    case hlsl:
        __intrinsic_asm "WaveActive$(opName.hlslName)";
    default:
        matrix<T,N,M> result;
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
            result[i] = WaveActive$(opName.hlslName)(expr[i]);
        return result;
    }
}
${{{{
} // WaveActiveBitAnd, WaveActiveBitOr, WaveActiveBitXor
}}}}

// WaveActiveMin/Max
${{{{
struct WaveActiveMinMaxEntry { const char* name; const char* metalName; };
const WaveActiveMinMaxEntry kWaveActiveMinMaxNames[] = {{"Min", "min"}, {"Max", "max"}};
for (const auto opName : kWaveActiveMinMaxNames) {
}}}}
/// @category wave
__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
T WaveActive$(opName.name)(T expr)
{
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroup$(opName.name)";
    case hlsl:
        __intrinsic_asm "WaveActive$(opName.name)";
    case metal:
        __intrinsic_asm "simd_$(opName.metalName)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformF$(opName.name) $$T result Subgroup Reduce $expr};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformU$(opName.name) $$T result Subgroup Reduce $expr};
        else
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformS$(opName.name) $$T result Subgroup Reduce $expr};
    case cuda:
        return WaveMask$(opName.name)(WaveGetActiveMask(), expr);
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
vector<T, N> WaveActive$(opName.name)(vector<T, N> expr)
{
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroup$(opName.name)";
    case hlsl:
        __intrinsic_asm "WaveActive$(opName.name)";
    case metal:
        __intrinsic_asm "simd_$(opName.metalName)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformF$(opName.name) $$vector<T, N> result Subgroup Reduce $expr};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformU$(opName.name) $$vector<T, N> result Subgroup Reduce $expr};
        else
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformS$(opName.name) $$vector<T, N> result Subgroup Reduce $expr};
    case cuda:
        return WaveMask$(opName.name)(WaveGetActiveMask(), expr);
    }
}

__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
matrix<T, N, M> WaveActive$(opName.name)(matrix<T, N, M> expr)
{
    __target_switch
    {
    case cuda:
        return WaveMask$(opName.name)(WaveGetActiveMask(), expr);
    case hlsl:
        __intrinsic_asm "WaveActive$(opName.name)";
    default:
        matrix<T, N, M> result;
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
            result[i] = WaveActive$(opName.name)(expr[i]);
        return result;
    }
}

${{{{
} // WaveActiveMinMax.
}}}}

// WaveActiveProduct/Sum
${{{{
struct WaveActiveProductSumEntry { const char* hlslName; const char* glslName; const char* metalName; };
const WaveActiveProductSumEntry kWaveActivProductSumNames[] = {{"Product", "Mul", "product"}, {"Sum", "Add", "sum"}};
for (auto opName : kWaveActivProductSumNames) {
}}}}
/// @category wave
__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
T WaveActive$(opName.hlslName)(T expr)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroup$(opName.glslName)($0)";
    case hlsl: __intrinsic_asm "WaveActive$(opName.hlslName)";
    case metal: __intrinsic_asm "simd_$(opName.metalName)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformF$(opName.glslName) $$T result Subgroup 0 $expr
            };
        else if (__isInt<T>())
        {
            return spirv_asm
            {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformI$(opName.glslName) $$T result Subgroup 0 $expr;
            };
        }
        else return expr;
    case wgsl: __intrinsic_asm "subgroup$(opName.glslName)";
    case cuda:
        return WaveMask$(opName.hlslName)(WaveGetActiveMask(), expr);
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
vector<T,N> WaveActive$(opName.hlslName)(vector<T,N> expr)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroup$(opName.glslName)($0)";
    case hlsl: __intrinsic_asm "WaveActive$(opName.hlslName)";
    case metal: __intrinsic_asm "simd_$(opName.metalName)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformF$(opName.glslName) $$vector<T,N> result Subgroup 0 $expr
            };
        else if (__isInt<T>())
        {
            return spirv_asm
            {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformI$(opName.glslName) $$vector<T,N> result Subgroup 0 $expr;
            };
        }
        else return expr;
    case wgsl: __intrinsic_asm "subgroup$(opName.glslName)";
    case cuda:
        return WaveMask$(opName.hlslName)(WaveGetActiveMask(), expr);
    }
}

__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
matrix<T, N, M> WaveActive$(opName.hlslName)(matrix<T, N, M> expr)
{
    __target_switch
    {
    case cuda:
        return WaveMask$(opName.hlslName)(WaveGetActiveMask(), expr);
    case hlsl:
        __intrinsic_asm "WaveActive$(opName.hlslName)";
    default:
        matrix<T, N, M> result;
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
            result[i] = WaveActive$(opName.hlslName)(expr[i]);
        return result;
    }
}
${{{{
} // WaveActiveProduct/WaveActiveProductSum.
}}}}

/// @category wave
__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_vote)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_vote)]
bool WaveActiveAllEqual(T value)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupAllEqual($0)";
    case hlsl:
        __intrinsic_asm "WaveActiveAllEqual";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformVote;
            OpGroupNonUniformAllEqual $$bool result Subgroup $value
        };
    default:
        return WaveMaskAllEqual(WaveGetActiveMask(), value);
    }
}

__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_vote)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_vote)]
bool WaveActiveAllEqual(vector<T,N> value)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupAllEqual($0)";
    case hlsl:
        __intrinsic_asm "WaveActiveAllEqual";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformVote;
            OpGroupNonUniformAllEqual $$bool result Subgroup $value
        };
    default:
        return WaveMaskAllEqual(WaveGetActiveMask(), value);
    }
}

__generic<T : __BuiltinType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_vote)]
bool WaveActiveAllEqual(matrix<T, N, M> value)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "WaveActiveAllEqual";
    default:
        return WaveMaskAllEqual(WaveGetActiveMask(), value);
    }
}

/// @category wave
__glsl_extension(GL_KHR_shader_subgroup_vote)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_vote)]
bool WaveActiveAllTrue(bool condition)
{
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupAll";
    case hlsl:
        __intrinsic_asm "WaveActiveAllTrue($0)";
    case metal:
        __intrinsic_asm "simd_all";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformVote;
            OpGroupNonUniformAll $$bool result Subgroup $condition
        };
    case cuda:
        return WaveMaskAllTrue(WaveGetActiveMask(), condition);
    }
}

/// @category wave
__glsl_extension(GL_KHR_shader_subgroup_vote)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_vote)]
bool WaveActiveAnyTrue(bool condition)
{
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupAny";
    case hlsl:
        __intrinsic_asm "WaveActiveAnyTrue($0)";
    case metal:
        __intrinsic_asm "simd_any";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformVote;
            OpGroupNonUniformAny $$bool result Subgroup $condition
        };
    default:
        return WaveMaskAnyTrue(WaveGetActiveMask(), condition);
    }
}


//@hidden:
[ForceInline]
uint64_t __metal_simd_ballot(bool expr)
{
    __intrinsic_asm "uint64_t(simd_ballot($0))";
}

[ForceInline]
uint4 __metal_simd_vote_mask_to_uint4(uint64_t mask)
{
    return uint4(uint(mask & 0xFFFFFFFF), uint(mask >> 32), 0, 0);
}

//@public:

/// @category wave
__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[NonUniformReturn]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_ballot)]
uint4 WaveActiveBallot(bool condition)
{
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupBallot";
    case hlsl:
        __intrinsic_asm "WaveActiveBallot";
    case metal: return __metal_simd_vote_mask_to_uint4(__metal_simd_ballot(condition));
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformBallot;
            OpGroupNonUniformBallot $$uint4 result Subgroup $condition
        };
    default:
        return WaveMaskBallot(WaveGetActiveMask(), condition);
    }
}

/// @category wave
[require(cuda_glsl_hlsl_spirv, subgroup_basic_ballot)]
uint WaveActiveCountBits(bool value)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "WaveActiveCountBits";
    case glsl:
    case spirv:
        return _WaveCountBits(WaveActiveBallot(value));
    default:
        return WaveMaskCountBits(WaveGetActiveMask(), value);
    }
}

/// @category wave
__glsl_extension(GL_KHR_shader_subgroup_basic)
__spirv_version(1.3)
[NonUniformReturn]
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_basic)]
uint WaveGetLaneCount()
{
    __target_switch
    {
    case glsl: __intrinsic_asm  "(gl_SubgroupSize)";
    case cuda: __intrinsic_asm  "(warpSize)";
    case hlsl: __intrinsic_asm  "WaveGetLaneCount()";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniform;
            result:$$uint = OpLoad builtin(SubgroupSize:uint)
        };
    case metal:
        return __builtinWaveLaneCount;
    case wgsl:
        __requireTargetExtension("subgroups");
        return __builtinWaveLaneCount;
    }
}

/// @category wave
__glsl_extension(GL_KHR_shader_subgroup_basic)
__spirv_version(1.3)
[NonUniformReturn]
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_basic)]
uint WaveGetLaneIndex()
{
    __target_switch
    {
    case glsl: __intrinsic_asm  "(gl_SubgroupInvocationID)";
    case cuda: __intrinsic_asm  "_getLaneId()";
    case hlsl: __intrinsic_asm  "WaveGetLaneIndex()";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniform;
            result:$$uint = OpLoad builtin(SubgroupLocalInvocationId:uint)
        };
    case metal:
        return __builtinWaveLaneIndex;
    case wgsl:
        __requireTargetExtension("subgroups");
        return __builtinWaveLaneIndex;
    }
}

/// @category wave
__glsl_extension(GL_KHR_shader_subgroup_basic)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[NonUniformReturn]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_basic)]
bool WaveIsFirstLane()
{
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupElect()";
    case hlsl: __intrinsic_asm "WaveIsFirstLane()";
    case metal: __intrinsic_asm "simd_is_first";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformBallot;
            OpGroupNonUniformElect $$bool result Subgroup
        };
    case wgsl: __intrinsic_asm "subgroupElect";
    case cuda:
        return WaveMaskIsFirstLane(WaveGetActiveMask());
    }
}

// It's useful to have a wave uint4 version of countbits, because some wave functions return uint4.
// This implementation tries to limit the amount of work required by the actual lane count.
/// @category wave
__spirv_version(1.3)
[require(cpp_cuda_glsl_hlsl_spirv, subgroup_basic_ballot)]
uint _WaveCountBits(uint4 value)
{
    __target_switch
    {
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformBallot;
            OpGroupNonUniformBallotBitCount $$uint result Subgroup Reduce $value
        };
    default:
        // Assume since WaveGetLaneCount should be known at compile time, the branches will hopefully boil away
        const uint waveLaneCount = WaveGetLaneCount();
        switch ((waveLaneCount - 1) / 32)
        {
            default:
            case 0: return countbits(value.x);
            case 1: return countbits(value.x) + countbits(value.y);
            case 2: return countbits(value.x) + countbits(value.y) + countbits(value.z);
            case 3: return countbits(value.x) + countbits(value.y) + countbits(value.z) + countbits(value.w);
        }
    }
}

// Prefix
/// @category wave
__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
T WavePrefixProduct(T expr)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupExclusiveMul($0)";
    case hlsl: __intrinsic_asm "WavePrefixProduct";
    case metal: __intrinsic_asm "simd_prefix_exclusive_product";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformFMul $$T result Subgroup ExclusiveScan $expr
            };
        else if (__isInt<T>())
        {
            return spirv_asm
            {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformIMul $$T result Subgroup ExclusiveScan $expr;
            };
        }
        else return expr;
    case wgsl: __intrinsic_asm "subgroupExclusiveMul";
    case cuda:
        return WaveMaskPrefixProduct(WaveGetActiveMask(), expr);
    }
}

/// @category wave
__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
vector<T,N> WavePrefixProduct(vector<T,N> expr)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupExclusiveMul($0)";
    case hlsl: __intrinsic_asm "WavePrefixProduct";
    case metal: __intrinsic_asm "simd_prefix_exclusive_product";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMul $$vector<T,N> result Subgroup ExclusiveScan $expr};
        else if (__isInt<T>())
        {
            return spirv_asm
            {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformIMul $$vector<T,N> result Subgroup ExclusiveScan $expr;
            };
        }
        else return expr;
    case wgsl: __intrinsic_asm "subgroupExclusiveMul";
    case cuda:
        return WaveMaskPrefixProduct(WaveGetActiveMask(), expr);
    }
}
/// @category wave
__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_glsl_hlsl_spirv_wgsl, subgroup_arithmetic)]
matrix<T, N, M> WavePrefixProduct(matrix<T, N, M> expr)
{
    __target_switch
    {
    case cuda:
        return WaveMaskPrefixProduct(WaveGetActiveMask(), expr);
    case hlsl:
        __intrinsic_asm "WavePrefixProduct";
    default:
        matrix<T, N, M> result;
        for (int i = 0; i < N; ++i)
            result[i] = WavePrefixProduct(expr[i]);
        return result;
    }
}

/// @category wave
__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
T WavePrefixSum(T expr)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupExclusiveAdd($0)";
    case hlsl: __intrinsic_asm "WavePrefixSum";
    case metal: __intrinsic_asm "simd_prefix_exclusive_sum";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFAdd $$T result Subgroup ExclusiveScan $expr};
        else if (__isInt<T>())
        {
            return spirv_asm
            {
                OpCapability GroupNonUniformArithmetic;
                result:$$T = OpGroupNonUniformIAdd Subgroup ExclusiveScan $expr;
            };
        }
        else return expr;
    case wgsl: __intrinsic_asm "subgroupExclusiveAdd";
    case cuda:
        return WaveMaskPrefixSum(WaveGetActiveMask(), expr);
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
vector<T,N> WavePrefixSum(vector<T,N> expr)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupExclusiveAdd($0)";
    case hlsl: __intrinsic_asm "WavePrefixSum";
    case metal: __intrinsic_asm "simd_prefix_exclusive_sum";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFAdd $$vector<T,N> result Subgroup ExclusiveScan $expr};
        else if (__isInt<T>())
        {
            return spirv_asm
            {
                OpCapability GroupNonUniformArithmetic;
                result:$$vector<T,N> = OpGroupNonUniformIAdd Subgroup ExclusiveScan $expr;
            };
        }
        else return expr;
    case wgsl: __intrinsic_asm "subgroupExclusiveAdd";
    case cuda:
        return WaveMaskPrefixSum(WaveGetActiveMask(), expr);
    }
}

__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
matrix<T,N,M> WavePrefixSum(matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda:
        return WaveMaskPrefixSum(WaveGetActiveMask(), expr);
    case hlsl:
        __intrinsic_asm "WavePrefixSum";
    default:
        matrix<T, N, M> result;
        for (int i = 0; i < N; ++i)
            result[i] = WavePrefixSum(expr[i]);
        return result;
    }
}

/// @category wave
__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_ballot)]
T WaveReadLaneFirst(T expr)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupBroadcastFirst($0)";
    case hlsl: __intrinsic_asm "WaveReadLaneFirst";
    case metal: __intrinsic_asm "simd_broadcast_first";
    case spirv:
        return spirv_asm {OpCapability GroupNonUniformBallot; OpGroupNonUniformBroadcastFirst $$T result Subgroup $expr};
    case wgsl: __intrinsic_asm "subgroupBroadcastFirst";
    case cuda:
        return WaveMaskReadLaneFirst(WaveGetActiveMask(), expr);
    }
}

__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_ballot)]
vector<T,N> WaveReadLaneFirst(vector<T,N> expr)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupBroadcastFirst($0)";
    case hlsl: __intrinsic_asm "WaveReadLaneFirst";
    case metal: __intrinsic_asm "simd_broadcast_first";
    case spirv:
        return spirv_asm {OpCapability GroupNonUniformBallot; OpGroupNonUniformBroadcastFirst $$vector<T,N> result Subgroup $expr};
    case wgsl: __intrinsic_asm "subgroupBroadcastFirst";
    case cuda:
        return WaveMaskReadLaneFirst(WaveGetActiveMask(), expr);
    }
}

__generic<T : __BuiltinType, let N : int, let M : int>
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_ballot)]
matrix<T,N,M> WaveReadLaneFirst(matrix<T,N,M> expr)
{
    __target_switch
    {
    case cuda:
        return WaveMaskReadLaneFirst(WaveGetActiveMask(), expr);
    case hlsl: __intrinsic_asm "WaveReadLaneFirst";
    default:
        matrix<T, N, M> result;
        for (int i = 0; i < N; ++i)
            result[i] = WaveReadLaneFirst(expr[i]);
        return result;
    }
}

// NOTE! WaveBroadcastLaneAt is *NOT* standard HLSL
// It is provided as access to subgroupBroadcast which can only take a
// constexpr laneId.
// https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_shader_subgroup.txt
// Versions SPIR-V greater than 1.4 loosen this restriction, and allow 'dynamic uniform' index
// If that's the behavior required then client code should use WaveReadLaneAt which works this way.
/// @category wave
__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_ballot)]
T WaveBroadcastLaneAt(T value, constexpr int lane)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupBroadcast($0, $1)";
    case hlsl: __intrinsic_asm "WaveReadLaneAt";
    case metal: __intrinsic_asm "simd_broadcast($0, ushort($1))";
    case spirv:
        let ulane = uint(lane);
        return spirv_asm {OpCapability GroupNonUniformBallot; OpGroupNonUniformBroadcast $$T result Subgroup $value $ulane};
    case wgsl: __intrinsic_asm "subgroupBroadcast";
    case cuda:
        return WaveMaskBroadcastLaneAt(WaveGetActiveMask(), value, lane);
    }
}

/// @category wave
__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_ballot)]
vector<T,N> WaveBroadcastLaneAt(vector<T,N> value, constexpr int lane)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupBroadcast($0, $1)";
    case hlsl: __intrinsic_asm "WaveReadLaneAt";
    case metal: __intrinsic_asm "simd_broadcast($0, ushort($1))";
    case spirv:
        let ulane = uint(lane);
        return spirv_asm {OpCapability GroupNonUniformBallot; OpGroupNonUniformBroadcast $$vector<T,N> result Subgroup $value $ulane};
    case wgsl: __intrinsic_asm "subgroupBroadcast";
    case cuda:
        return WaveMaskBroadcastLaneAt(WaveGetActiveMask(), value, lane);
    }
}

__generic<T : __BuiltinType, let N : int, let M : int>
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_ballot)]
matrix<T, N, M> WaveBroadcastLaneAt(matrix<T, N, M> value, constexpr int lane)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_waveShuffleMultiple(_getActiveMask(), $0, $1)";
    case hlsl: __intrinsic_asm "WaveReadLaneAt";
    default:
        matrix<T, N, M> result;
        for (int i = 0; i < N; ++i)
            result[i] = WaveBroadcastLaneAt(value[i], lane);
        return result;
    }
}

// TODO(JS): If it can be determines that the `laneId` is constExpr, then subgroupBroadcast
// could be used on GLSL. For now we just use subgroupShuffle
/// @category wave
__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_shuffle)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_shuffle)]
T WaveReadLaneAt(T value, int lane)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupShuffle($0, $1)";
    case hlsl: __intrinsic_asm "WaveReadLaneAt";
    case metal: __intrinsic_asm "simd_shuffle($0, ushort($1))";
    case spirv:
        let ulane = uint(lane);
        return spirv_asm {OpCapability GroupNonUniformShuffle; OpGroupNonUniformShuffle $$T result Subgroup $value $ulane};
    case wgsl: __intrinsic_asm "subgroupShuffle";
    case cuda:
        return WaveMaskReadLaneAt(WaveGetActiveMask(), value, lane);
    }
}

__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_shuffle)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_shuffle)]
vector<T,N> WaveReadLaneAt(vector<T,N> value, int lane)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupShuffle($0, $1)";
    case hlsl: __intrinsic_asm "WaveReadLaneAt";
    case metal: __intrinsic_asm "simd_shuffle($0, ushort($1))";
    case spirv:
        let ulane = uint(lane);
        return spirv_asm {OpCapability GroupNonUniformShuffle; OpGroupNonUniformShuffle $$vector<T,N> result Subgroup $value $ulane};
    case wgsl: __intrinsic_asm "subgroupShuffle";
    case cuda:
        return WaveMaskReadLaneAt(WaveGetActiveMask(), value, lane);
    }
}

__generic<T : __BuiltinType, let N : int, let M : int>
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_shuffle)]
matrix<T, N, M> WaveReadLaneAt(matrix<T, N, M> value, int lane)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_waveShuffleMultiple(_getActiveMask(), $0, $1)";
    case hlsl: __intrinsic_asm "WaveReadLaneAt";
    default:
        matrix<T,N,M> result;
        for (int i = 0; i < N; ++i)
            result[i] = WaveReadLaneAt(value[i], lane);
        return result;
    }
}

// NOTE! WaveShuffle is a NON STANDARD HLSL intrinsic! It will map to WaveReadLaneAt on HLSL
// which means it will only work on hardware which allows arbitrary laneIds which is not true
// in general because it breaks the HLSL standard, which requires it's 'dynamically uniform' across the Wave.
/// @category wave
__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_shuffle)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_shuffle)]
T WaveShuffle(T value, int lane)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupShuffle($0, $1)";
    case hlsl: __intrinsic_asm "WaveReadLaneAt";
    case metal: __intrinsic_asm "simd_shuffle($0, ushort($1))";
    case spirv:
        let ulane = uint(lane);
        return spirv_asm {OpCapability GroupNonUniformShuffle; OpGroupNonUniformShuffle $$T result Subgroup $value $ulane};
    case wgsl: __intrinsic_asm "subgroupShuffle";
    case cuda:
        return WaveMaskShuffle(WaveGetActiveMask(), value, lane);
    }
}

/// @category wave
__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_shuffle)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_shuffle)]
vector<T,N> WaveShuffle(vector<T,N> value, int lane)
{
    __target_switch
    {
    case glsl:
        if (__isHalf<T>()) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
        __intrinsic_asm "subgroupShuffle($0, $1)";
    case hlsl: __intrinsic_asm "WaveReadLaneAt";
    case metal: __intrinsic_asm "simd_shuffle($0, ushort($1))";
    case spirv:
        let ulane = uint(lane);
        return spirv_asm {OpCapability GroupNonUniformShuffle; OpGroupNonUniformShuffle $$vector<T,N> result Subgroup $value $ulane};
    case wgsl: __intrinsic_asm "subgroupShuffle";
    case cuda:
        return WaveMaskShuffle(WaveGetActiveMask(), value, lane);
    }
}

__generic<T : __BuiltinType, let N : int, let M : int>
[require(cuda_hlsl, subgroup_shuffle)]
matrix<T, N, M> WaveShuffle(matrix<T, N, M> value, int lane)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "WaveReadLaneAt";
    default:
        return WaveMaskShuffle(WaveGetActiveMask(), value, lane);
    }
}

/// @category wave
__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, subgroup_ballot)]
uint WavePrefixCountBits(bool value)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupBallotExclusiveBitCount(subgroupBallot($0))";
    case hlsl: __intrinsic_asm "WavePrefixCountBits($0)";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformBallot;
            %mask:$$uint4 = OpGroupNonUniformBallot Subgroup $value;
            OpGroupNonUniformBallotBitCount $$uint result Subgroup 2 %mask
        };
    default:
        return WaveMaskPrefixCountBits(WaveGetActiveMask(), value);
    }
}

/// @category wave
__glsl_extension(GL_KHR_shader_subgroup_ballot)
__spirv_version(1.3)
__wgsl_extension(subgroups)
[require(cuda_glsl_hlsl_spirv_wgsl, subgroup_ballot)]
uint4 WaveGetConvergedMulti()
{
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupBallot(true)";
    case hlsl: __intrinsic_asm "WaveActiveBallot(true)";
    case cuda: __intrinsic_asm "make_uint4(__activemask(), 0, 0, 0)";
    case spirv:
        let _true = true;
        return spirv_asm
        {
            OpCapability GroupNonUniformBallot;
            OpGroupNonUniformBallot $$uint4 result Subgroup $_true
        };
    }
}

/// @category wave
[ForceInline]
uint4 WaveGetActiveMulti()
{
    return WaveGetConvergedMulti();
}

// Shader model 6.5 stuff
// https://github.com/microsoft/DirectX-Specs/blob/master/d3d/HLSL_ShaderModel6_5.md

/// @category wave
__generic<T : __BuiltinType>
[require(cuda_glsl_hlsl_spirv, subgroup_partitioned)]
uint4 WaveMatch(T value)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "WaveMatch";
    case glsl: __intrinsic_asm "subgroupPartitionNV($0)";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformPartitionedNV;
            OpExtension "SPV_NV_shader_subgroup_partitioned";
            OpGroupNonUniformPartitionNV $$uint4 result $value
        };
    default:
        return WaveMaskMatch(WaveGetActiveMask(), value);
    }
}

__generic<T : __BuiltinType, let N : int>
[require(cuda_glsl_hlsl_spirv, subgroup_partitioned)]
uint4 WaveMatch(vector<T,N> value)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "WaveMatch";
    case glsl: __intrinsic_asm "subgroupPartitionNV($0)";
    case spirv:
        return spirv_asm
        {
            OpCapability GroupNonUniformPartitionedNV;
            OpExtension "SPV_NV_shader_subgroup_partitioned";
            OpGroupNonUniformPartitionNV $$uint4 result $value
        };
    default:
        return WaveMaskMatch(WaveGetActiveMask(), value);
    }
}

__generic<T : __BuiltinType, let N : int, let M : int>
[require(cuda_glsl_hlsl_spirv, subgroup_partitioned)]
uint4 WaveMatch(matrix<T,N,M> value)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "WaveMatch";
    case glsl:
    case cuda:
    case spirv:
        uint4 result = uint4(0xFFFFFFFF);
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            result &= WaveMatch(value[i]);
        return result;
    default:
        return WaveMaskMatch(WaveGetActiveMask(), value);
    }
}

/// @category wave
[require(cuda_hlsl, wave_multi_prefix)]
uint WaveMultiPrefixCountBits(bool value, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_popc(__ballot_sync(($1).x, $0) & _getLaneLtMask())";
    case hlsl: __intrinsic_asm "WaveMultiPrefixCountBits";
    }
}

/// @category wave
__generic<T : __BuiltinIntegerType>
__glsl_extension(GL_NV_shader_subgroup_partitioned)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
T WaveMultiPrefixBitAnd(T expr, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixAnd(_getMultiPrefixMask(($1).x), $0)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitAnd";
    case glsl: __intrinsic_asm "subgroupPartitionedExclusiveAndNV";
    case spirv:
        return spirv_asm
        {
            OpExtension "SPV_NV_shader_subgroup_partitioned";
            OpCapability GroupNonUniformPartitionedNV;
            result:$$T = OpGroupNonUniformBitwiseAnd Subgroup PartitionedExclusiveScanNV $expr $mask
        };
    }
}

__generic<T : __BuiltinIntegerType, let N : int>
__glsl_extension(GL_NV_shader_subgroup_partitioned)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
vector<T,N> WaveMultiPrefixBitAnd(vector<T,N> expr, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixAndMultiple(_getMultiPrefixMask(($1).x), $0)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitAnd";
    case glsl: __intrinsic_asm "subgroupPartitionedExclusiveAndNV";
    case spirv:
        return spirv_asm
        {
            OpExtension "SPV_NV_shader_subgroup_partitioned";
            OpCapability GroupNonUniformPartitionedNV;
            result:$$vector<T,N> = OpGroupNonUniformBitwiseAnd Subgroup PartitionedExclusiveScanNV $expr $mask
        };
    }
}

__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
matrix<T,N,M> WaveMultiPrefixBitAnd(matrix<T,N,M> expr, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixAndMultiple(_getMultiPrefixMask(($1).x), $0)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitAnd";
    case glsl:
    case spirv:
        matrix<T, N, M> result;
        for (int i = 0; i < N; ++i)
            result[i] = WaveMultiPrefixBitAnd(expr[i], mask);
        return result;
    }
}

/// @category wave
__generic<T : __BuiltinIntegerType>
__glsl_extension(GL_NV_shader_subgroup_partitioned)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
T WaveMultiPrefixBitOr(T expr, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixOr(, _getMultiPrefixMask(($1).x), $0)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitOr";
    case glsl: __intrinsic_asm "subgroupPartitionedExclusiveOrNV";
    case spirv:
        return spirv_asm
        {
            OpExtension "SPV_NV_shader_subgroup_partitioned";
            OpCapability GroupNonUniformPartitionedNV;
            result:$$T = OpGroupNonUniformBitwiseOr Subgroup PartitionedExclusiveScanNV $expr $mask
        };
    }
}

__generic<T : __BuiltinIntegerType, let N : int>
__glsl_extension(GL_NV_shader_subgroup_partitioned)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
vector<T,N> WaveMultiPrefixBitOr(vector<T,N> expr, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixOrMultiple(_getMultiPrefixMask(($1).x), $0)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitOr";
    case glsl: __intrinsic_asm "subgroupPartitionedExclusiveOrNV";
    case spirv:
        return spirv_asm
        {
            OpExtension "SPV_NV_shader_subgroup_partitioned";
            OpCapability GroupNonUniformPartitionedNV;
            result:$$vector<T,N> = OpGroupNonUniformBitwiseOr Subgroup PartitionedExclusiveScanNV $expr $mask
        };
    }
}

__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
matrix<T,N,M> WaveMultiPrefixBitOr(matrix<T,N,M> expr, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixOrMultiple(_getMultiPrefixMask(($1).x), $0)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitOr";
    case glsl:
    case spirv:
        matrix<T, N, M> result;
        for (int i = 0; i < N; ++i)
            result[i] = WaveMultiPrefixBitOr(expr[i], mask);
        return result;
    }
}

/// @category wave
__generic<T : __BuiltinIntegerType>
__glsl_extension(GL_NV_shader_subgroup_partitioned)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
T WaveMultiPrefixBitXor(T expr, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixXor(_getMultiPrefixMask(($1).x), $0)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitXor";
    case glsl: __intrinsic_asm "subgroupPartitionedExclusiveXorNV";
    case spirv:
        return spirv_asm
        {
            OpExtension "SPV_NV_shader_subgroup_partitioned";
            OpCapability GroupNonUniformPartitionedNV;
            result:$$T = OpGroupNonUniformBitwiseXor Subgroup PartitionedExclusiveScanNV $expr $mask
        };
    }
}

__generic<T : __BuiltinIntegerType, let N : int>
__glsl_extension(GL_NV_shader_subgroup_partitioned)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
vector<T,N> WaveMultiPrefixBitXor(vector<T,N> expr, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixXorMultiple(_getMultiPrefixMask(($1).x), $0)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitXor";
    case glsl: __intrinsic_asm "subgroupPartitionedExclusiveXorNV";
    case spirv:
        return spirv_asm
        {
            OpExtension "SPV_NV_shader_subgroup_partitioned";
            OpCapability GroupNonUniformPartitionedNV;
            result:$$vector<T,N> = OpGroupNonUniformBitwiseXor Subgroup PartitionedExclusiveScanNV $expr $mask
        };
    }
}

__generic<T : __BuiltinIntegerType, let N : int, let M : int>
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
matrix<T,N,M> WaveMultiPrefixBitXor(matrix<T,N,M> expr, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixXorMultiple(_getMultiPrefixMask(($1).x), $0)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixBitXor";
    case glsl:
    case spirv:
        matrix<T, N, M> result;
        for (int i = 0; i < N; ++i)
            result[i] = WaveMultiPrefixBitXor(expr[i], mask);
        return result;
    }
}

/// @category wave
__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_NV_shader_subgroup_partitioned)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
T WaveMultiPrefixProduct(T value, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixProduct(_getMultiPrefixMask(($1).x), $0)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixProduct";
    case glsl: __intrinsic_asm "subgroupPartitionedExclusiveMulNV";
    case spirv:
        {
            spirv_asm
            {
                OpExtension "SPV_NV_shader_subgroup_partitioned";
                OpCapability GroupNonUniformPartitionedNV;
            };

            if (__isFloat<T>())
            {
                return spirv_asm
                {
                    result:$$T = OpGroupNonUniformFMul Subgroup PartitionedExclusiveScanNV $value $mask
                };
            }
            else
            {
                return spirv_asm
                {
                    result:$$T = OpGroupNonUniformIMul Subgroup PartitionedExclusiveScanNV $value $mask
                };
            }
        }
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_NV_shader_subgroup_partitioned)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
vector<T,N> WaveMultiPrefixProduct(vector<T,N> value, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixProductMultiple(_getMultiPrefixMask(($1).x), $0)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixProduct";
    case glsl: __intrinsic_asm "subgroupPartitionedExclusiveMulNV";
    case spirv:
        {
            spirv_asm
            {
                OpExtension "SPV_NV_shader_subgroup_partitioned";
                OpCapability GroupNonUniformPartitionedNV;
            };

            if (__isFloat<T>())
            {
                return spirv_asm
                {
                    result:$$vector<T,N> = OpGroupNonUniformFMul Subgroup PartitionedExclusiveScanNV $value $mask
                };
            }
            else
            {
                return spirv_asm
                {
                    result:$$vector<T,N> = OpGroupNonUniformIMul Subgroup PartitionedExclusiveScanNV $value $mask
                };
            }
        }
    }
}

__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
matrix<T,N,M> WaveMultiPrefixProduct(matrix<T,N,M> value, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixProductMultiple(_getMultiPrefixMask(($1).x), $0)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixProduct";
    case glsl:
    case spirv:
        matrix<T, N, M> result;
        for (int i = 0; i < N; ++i)
            result[i] = WaveMultiPrefixProduct(value[i], mask);
        return result;
    }
}

/// @category wave
__generic<T : __BuiltinArithmeticType>
__glsl_extension(GL_NV_shader_subgroup_partitioned)
__spirv_version(1.3)
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
T WaveMultiPrefixSum(T value, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixSum(_getMultiPrefixMask(($1).x), $0)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixSum";
    case glsl: __intrinsic_asm "subgroupPartitionedExclusiveAddNV";
    case spirv:
        {
            spirv_asm
            {
                OpExtension "SPV_NV_shader_subgroup_partitioned";
                OpCapability GroupNonUniformPartitionedNV;
            };

            if (__isFloat<T>())
            {
                return spirv_asm
                {
                    result:$$T = OpGroupNonUniformFAdd Subgroup PartitionedExclusiveScanNV $value $mask
                };
            }
            else
            {
                return spirv_asm
                {
                    result:$$T = OpGroupNonUniformIAdd Subgroup PartitionedExclusiveScanNV $value $mask
                };
            }
        }
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__glsl_extension(GL_NV_shader_subgroup_partitioned)
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
__spirv_version(1.3)
vector<T,N> WaveMultiPrefixSum(vector<T,N> value, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixSumMultiple(_getMultiPrefixMask(($1).x), $0 )";
    case hlsl: __intrinsic_asm "WaveMultiPrefixSum";
    case glsl: __intrinsic_asm "subgroupPartitionedExclusiveAddNV";
    case spirv:
        {
            spirv_asm
            {
                OpExtension "SPV_NV_shader_subgroup_partitioned";
                OpCapability GroupNonUniformPartitionedNV;
            };

            if (__isFloat<T>())
            {
                return spirv_asm
                {
                    result:$$vector<T,N> = OpGroupNonUniformFAdd Subgroup PartitionedExclusiveScanNV $value $mask
                };
            }
            else
            {
                return spirv_asm
                {
                    result:$$vector<T,N> = OpGroupNonUniformIAdd Subgroup PartitionedExclusiveScanNV $value $mask
                };
            }
        }
    }
}

__generic<T : __BuiltinArithmeticType, let N : int, let M : int>
[require(cuda_glsl_hlsl_spirv, wave_multi_prefix)]
matrix<T,N,M> WaveMultiPrefixSum(matrix<T,N,M> value, uint4 mask)
{
    __target_switch
    {
    case cuda: __intrinsic_asm "_wavePrefixSumMultiple(_getMultiPrefixMask(($1).x), $0)";
    case hlsl: __intrinsic_asm "WaveMultiPrefixSum";
    case glsl:
    case spirv:
        matrix<T, N, M> result;
        for (int i = 0; i < N; ++i)
            result[i] = WaveMultiPrefixSum(value[i], mask);
        return result;
    }
}

__glsl_extension(GL_EXT_demote_to_helper_invocation)
[ForceInline]
[require(glsl_hlsl_metal_spirv, helper_lane)]
bool IsHelperLane()
{
    __target_switch {
    case hlsl: __intrinsic_asm "IsHelperLane()";
    case glsl: __intrinsic_asm "gl_HelperInvocation";
    case metal: __intrinsic_asm "simd_is_helper_thread()";
    case spirv:
        return spirv_asm {
            OpExtension "SPV_EXT_demote_to_helper_invocation";
            OpCapability DemoteToHelperInvocationEXT;
            result:$$bool = OpIsHelperInvocationEXT
        };
    }
}

//@hidden:

__generic<T : __BuiltinType>
[ForceInline]
[require(glsl)]
void __requireGLSLShaderSubgroupTypeExtension()
{
    // the following is a seperate function call, since else the `__requireTargetExtension` and associated __intrinsic_asm is ignored if the calling function also calls an __intrinsic_asm
    if (__type_equals<T, half>()
        || __type_equals<T, float16_t>()
        ) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_float16");
    else if (__type_equals<T, uint8_t>()
        || __type_equals<T, int8_t>()
        ) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_int8");
    else if (__type_equals<T, uint16_t>()
        || __type_equals<T, int16_t>()
        ) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_int16");
    else if (__type_equals<T, uint64_t>()
        || __type_equals<T, int64_t>()
        ) __requireTargetExtension("GL_EXT_shader_subgroup_extended_types_int64");

    __intrinsic_asm "";
}

__generic<T : __BuiltinType>
[ForceInline]
[require(metal)]
void __checkMetalShaderSubgroupType()
{
    // These builtin types are not supported for Metal's `simd` operations.
    if (__type_equals<T, uint8_t>()
        || __type_equals<T, int8_t>()
        || __type_equals<T, uint64_t>()
        || __type_equals<T, int64_t>()
        || __isBool<T>()
        )
    {
        static_assert(false, "Unsupported type for subgroup operations in Metal. Valid types include scalars and vectors of uint/uint32_t, int/int32_t, uint16_t, int16_t, float, and half.");
    }
}

__generic<T : __BuiltinType>
void shader_subgroup_preamble()
{
    // checks needed for shader_subgroup functions; __requireTargetExtension does not work
    // (does not add the ext specified correctly to the compile output; using extended type
    // will result in error for using the type)
    __target_switch
    {
    case glsl:
        __requireGLSLShaderSubgroupTypeExtension<T>();
    case metal:
        __checkMetalShaderSubgroupType<T>();
    default:
        return;
    }
}

//@public:

//
// Wave Rotate intrinsics.
// These are Slang specific intrinsics to rotate values within a subgroup.
//

__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_rotate)
[require(glsl_metal_spirv, subgroup_rotate)]
T WaveRotate(T value, uint delta)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupRotate";
    case metal:
        __intrinsic_asm "simd_shuffle_rotate_down";
    case spirv:
        return spirv_asm
        {
            OpExtension "SPV_KHR_subgroup_rotate";
            OpCapability GroupNonUniformRotateKHR;
            result:$$T = OpGroupNonUniformRotateKHR Subgroup $value $delta;
        };
    }
}

__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_rotate)
[require(glsl_metal_spirv, subgroup_rotate)]
vector<T, N> WaveRotate(vector<T, N> value, uint delta)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupRotate";
    case metal:
        __intrinsic_asm "simd_shuffle_rotate_down";
    case spirv:
        return spirv_asm
        {
            OpExtension "SPV_KHR_subgroup_rotate";
            OpCapability GroupNonUniformRotateKHR;
            result:$$vector<T,N> = OpGroupNonUniformRotateKHR Subgroup $value $delta;
        };
    }
}

__generic<T : __BuiltinType>
__glsl_extension(GL_KHR_shader_subgroup_rotate)
[require(glsl_spirv, subgroup_rotate)]
T WaveClusteredRotate(T value, uint delta, constexpr uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredRotate";
    case spirv:
        return spirv_asm
        {
            OpExtension "SPV_KHR_subgroup_rotate";
            OpCapability GroupNonUniformRotateKHR;
            result:$$T = OpGroupNonUniformRotateKHR Subgroup $value $delta $clusterSize;
        };
    }
}

__generic<T : __BuiltinType, let N : int>
__glsl_extension(GL_KHR_shader_subgroup_rotate)
[require(glsl_spirv, subgroup_rotate)]
vector<T, N> WaveClusteredRotate(vector<T, N> value, uint delta, constexpr uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredRotate";
    case spirv:
        return spirv_asm
        {
            OpExtension "SPV_KHR_subgroup_rotate";
            OpCapability GroupNonUniformRotateKHR;
            result:$$vector<T,N> = OpGroupNonUniformRotateKHR Subgroup $value $delta $clusterSize;
        };
    }
}

//
// Quad Control intrinsics
//
// For SPIRV and GLSL targets, the behavior is taken from Vulkan's `VK_KHR_shader_quad_control` spec.
// QuadAny/QuadAll will map to OpGroupNonUniformQuadAny/All, and using either of these functions will 
// result in the QuadDerivativesKHR execution mode being used. If MaximallyReconvergesKHR is not already 
// specified by other means, it will be added when using either of QuadAny/QuadAll,
//

//@public:
/// Returns true if `expr` is true in any lane of the current quad.
__glsl_extension(GL_KHR_shader_subgroup_vote)
__glsl_extension(GL_EXT_maximal_reconvergence)
__glsl_extension(GL_EXT_shader_quad_control)
__spirv_version(1.3)
[ForceInline]
[require(glsl_hlsl_metal_spirv, quad_control)]
bool QuadAny(bool expr)
{
    __requireMaximallyReconverges();
    __requireQuadDerivatives();
    __target_switch
    {
    case hlsl: __intrinsic_asm "QuadAny";
    case glsl: __intrinsic_asm "subgroupQuadAny";
    case metal: __intrinsic_asm "quad_any";
    case spirv:
        return spirv_asm
        {
            result:$$bool = OpGroupNonUniformQuadAnyKHR $expr;
        };
    }
}

//@public:
/// Returns true if `expr` is true in all lanes of the current quad.
__glsl_extension(GL_KHR_shader_subgroup_vote)
__glsl_extension(GL_EXT_maximal_reconvergence)
__glsl_extension(GL_EXT_shader_quad_control)
__spirv_version(1.3)
[ForceInline]
[require(glsl_hlsl_metal_spirv, quad_control)]
bool QuadAll(bool expr)
{
    __requireMaximallyReconverges();
    __requireQuadDerivatives();
    __target_switch
    {
    case hlsl: __intrinsic_asm "QuadAll";
    case glsl: __intrinsic_asm "subgroupQuadAll";
    case metal: __intrinsic_asm "quad_all";
    case spirv:
        return spirv_asm
        {
            result:$$bool = OpGroupNonUniformQuadAllKHR $expr;
        };
    }
}

// `typedef`s to help with the fact that HLSL has been sorta-kinda case insensitive at various points
//@hidden:
typedef Texture2D texture2D;
//@public:

${{{{

// Buffer types

static const struct {
    char const*         name;
    SlangResourceAccess access;
} kBaseBufferAccessLevels[] = {
    { "",                   SLANG_RESOURCE_ACCESS_READ },
    { "RW",                 SLANG_RESOURCE_ACCESS_READ_WRITE },
    { "RasterizerOrdered",  SLANG_RESOURCE_ACCESS_RASTER_ORDERED },
};
static const int kBaseBufferAccessLevelCount = sizeof(kBaseBufferAccessLevels) / sizeof(kBaseBufferAccessLevels[0]);

for (int aa = 0; aa < kBaseBufferAccessLevelCount; ++aa)
{
    auto access = kBaseBufferAccessLevels[aa].access;
    sb << "/// @category texture_types\n";
    sb << "__generic<T:ITexelElement,let format:int=0>\n";
    sb << "typealias ";
    sb << kBaseBufferAccessLevels[aa].name;
    sb << "Buffer = _Texture<T, __ShapeBuffer, 0, 0, 0, " << aa << ", 0, 0, format>;\n";
    
    bool isReadOnly = aa == 0;

    char const* glslTextureSizeFunc = (isReadOnly) ? "textureSize" : "imageSize";
    char const* glslLoadFuncName = (isReadOnly) ? "texelFetch" : "imageLoad";
    char const* spvLoadInstName = (isReadOnly) ? "OpImageFetch" : "OpImageRead";
    char const* requireToSetQuery = (isReadOnly) ? "[require(glsl_hlsl_metal_spirv, texture_size)]" : "[require(glsl_hlsl_metal_spirv, image_size)]";
    char const* requireToSet = (isReadOnly) ? "[require(glsl_hlsl_metal_spirv, texture_sm_4_1)]" : "[require(glsl_hlsl_metal_spirv, texture_sm_4_1_compute_fragment)]";
    char const* requireToSet_onlyHLSL = (isReadOnly) ? "[require(hlsl, texture_sm_4_1)]" : "[require(hlsl, texture_sm_4_1_compute_fragment)]";
}}}}

__generic<T:ITexelElement, let format:int>
extension _Texture<T, __ShapeBuffer, 0, 0, 0, $(aa), 0, 0, format>
{
    [__readNone]
    $(requireToSetQuery)
    void GetDimensions(out uint dim)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".GetDimensions";
        case glsl:
            __requireTargetExtension("GL_EXT_samplerless_texture_functions");
            __intrinsic_asm "($1 = $(glslTextureSizeFunc)($0))";
        case metal: __intrinsic_asm "(*($1) = $0.get_width())";
        case spirv:
            dim = spirv_asm {
                OpCapability ImageQuery;
                result:$$uint = OpImageQuerySize $this;
            };
        }
    }

    $(isReadOnly?"[__readNone] ":"")
    $(requireToSet)
    T Load(int location)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load";
        case metal: __intrinsic_asm "$c$0.read(uint($1))$z";
        case glsl:
            __requireTargetExtension("GL_EXT_samplerless_texture_functions");
            __intrinsic_asm "$(glslLoadFuncName)($0, $1)$z";
        case spirv: return spirv_asm {
                %sampled:__sampledType(T) = $(spvLoadInstName) $this $location;
                __truncate $$T result __sampledType(T) %sampled;
            };
        }
    }

    $(isReadOnly?"[__readNone] ":"")
    $(requireToSet_onlyHLSL)
    T Load(int location, out uint status)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Load";
        }
    }

    __subscript(uint index) -> T {

        $(isReadOnly?"[__readNone] ":"")
        [ForceInline]
        $(requireToSet)
        get { return Load((int)index); }
${{{{
        if (access != SLANG_RESOURCE_ACCESS_READ) {
}}}}
            [nonmutating] 
            $(requireToSet)
            set
            {
                __target_switch
                {
                case hlsl: __intrinsic_asm "($0)[$1] = $2";
                case glsl: __intrinsic_asm "imageStore($0, int($1), $V2)";
                case metal: __intrinsic_asm "$0.write($2, $1)";
                case spirv: spirv_asm {
                        OpImageWrite $this $index __convertTexel(newValue);
                    };
                }
            }

            // If a 'Texture[index]' is referred to by a '__ref', call 'kIROp_ImageSubscript(index)'.
            // This allows call's to stay aware that the input is from a 'Texture'.
            __intrinsic_op($(kIROp_ImageSubscript))
            [nonmutating]
            ref;
${{{{
        } // access != SLANG_RESOURCE_ACCESS_READ
}}}}

        }
    

    };  // end extension
${{{{
}
}}}}


// DirectX Raytracing (DXR) Support
//
// The following is based on the experimental DXR SDK v0.09.01.
//
// Numbering follows the sections in the "D3D12 Raytracing Functional Spec" v0.09 (2018-03-12)
//

// 10.1.1 - Ray Flags

/// Flags that control ray traversal behavior and shader execution.
/// @category raytracing
typedef uint RAY_FLAG;

/// No special ray flags.
/// @category raytracing
static const RAY_FLAG RAY_FLAG_NONE = 0x00;

/// Forces all geometries to be treated as opaque, disabling any-hit shader execution.
/// @category raytracing
static const RAY_FLAG RAY_FLAG_FORCE_OPAQUE = 0x01;

/// Forces all geometries to be treated as non-opaque, enabling any-hit shader execution.
/// @category raytracing
static const RAY_FLAG RAY_FLAG_FORCE_NON_OPAQUE = 0x02;

/// Accepts the first intersection found and skips searching for closer hits.
/// @category raytracing
static const RAY_FLAG RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH = 0x04;

/// Skips execution of closest hit shaders, useful for shadow rays.
/// @category raytracing
static const RAY_FLAG RAY_FLAG_SKIP_CLOSEST_HIT_SHADER = 0x08;

/// Culls triangles facing away from the ray origin.
/// @category raytracing
static const RAY_FLAG RAY_FLAG_CULL_BACK_FACING_TRIANGLES = 0x10;

/// Culls triangles facing toward the ray origin.
/// @category raytracing
static const RAY_FLAG RAY_FLAG_CULL_FRONT_FACING_TRIANGLES = 0x20;

/// Skips intersections with opaque geometry.
/// @category raytracing
static const RAY_FLAG RAY_FLAG_CULL_OPAQUE = 0x40;

/// Skips intersections with non-opaque geometry.
/// @category raytracing
static const RAY_FLAG RAY_FLAG_CULL_NON_OPAQUE = 0x80;

/// Skips all triangle intersections.
/// @category raytracing
static const RAY_FLAG RAY_FLAG_SKIP_TRIANGLES = 0x100;

/// Skips all procedural primitive intersections.
/// @category raytracing
static const RAY_FLAG RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES = 0x200;

// 10.1.2 - Ray Description Structure
/// Describes a ray for traversal through an acceleration structure.
/// @category raytracing
__target_intrinsic(hlsl, RayDesc)
__target_intrinsic(cuda, RayDesc)
struct RayDesc
{
    /// Starting point of the ray in world space.
    __target_intrinsic(hlsl, Origin)
    __target_intrinsic(cuda, Origin)
    float3 Origin;

    /// Minimum distance along the ray to consider intersections.
    __target_intrinsic(hlsl, TMin)
    __target_intrinsic(cuda, TMin)
    float  TMin;

    /// Normalized direction vector of the ray in world space.
    __target_intrinsic(hlsl, Direction)
    __target_intrinsic(cuda, Direction)
    float3 Direction;

    /// Maximum distance along the ray to consider intersections.
    __target_intrinsic(hlsl, TMax)
    __target_intrinsic(cuda, TMax)
    float  TMax;
};

// 10.1.3 - Ray Acceleration Structure
/// Opaque type representing a ray-tracing acceleration structure.
/// @category raytracing
__builtin
__magic_type(RaytracingAccelerationStructureType)
__intrinsic_type($(kIROp_RaytracingAccelerationStructureType))
struct RaytracingAccelerationStructure
{
    [require(glsl_spirv, raytracing)]
    [__readNone]
    __init(uint64_t address)
    {
        __target_switch
        {
        case spirv:
            return spirv_asm {
                result: $$RaytracingAccelerationStructure = OpConvertUToAccelerationStructureKHR $address;
            };
        case glsl:
            __intrinsic_asm "accelerationStructureEXT($0)";
        }
    }
};

// 10.1.4 - Subobject Definitions

// TODO: We may decide to support these, but their reliance on C++ implicit
// constructor call syntax (`SomeType someVar(arg0, arg1);`) makes them
// annoying for the current Slang parsing strategy, and using global variables
// for this stuff comes across as a kludge rather than the best possible design.

// 10.1.5 - Intersection Attributes Structure
/// Built-in structure containing intersection attributes for triangle primitives.
/// @category raytracing
__target_intrinsic(hlsl, BuiltInTriangleIntersectionAttributes)
[require(cpp_cuda_glsl_hlsl_spirv, raytracing)]
struct BuiltInTriangleIntersectionAttributes
{
    /// Barycentric coordinates of the intersection point on the triangle.
    __target_intrinsic(hlsl, barycentrics)
    float2 barycentrics;
};

// 10.2 Shaders

// Right now new shader stages need to be added directly to the compiler
// implementation, rather than being something that can be declared in the core module.

// 10.3 - Intrinsics

// 10.3.1

// `executeCallableNV` is the GLSL intrinsic that will be used to implement
// `CallShader()` for GLSL-based targets.
//
[require(glsl, raytracing_raygen_closesthit_miss_callable)]
void __executeCallable(uint shaderIndex, int payloadLocation)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "executeCallableEXT";
    }
}

// Next is the custom intrinsic that will compute the payload location
// for a type being used in a `CallShader()` call for GLSL-based targets.
//
__generic<Payload>
[__readNone]
__intrinsic_op($(kIROp_GetVulkanRayTracingPayloadLocation))
int __callablePayloadLocation(__ref Payload payload);

// Now we provide a hard-coded definition of `CallShader()` for GLSL-based
// targets, which maps the generic HLSL operation into the non-generic
// GLSL equivalent.
//
/// Executes a callable shader with the specified payload.
/// @param shaderIndex Index of the callable shader to execute
/// @param payload Data structure to pass to and receive from the callable shader
/// @remarks Used to implement dynamic shader calls during ray tracing
/// @category raytracing
__generic<Payload>
[require(glsl_hlsl_spirv, raytracing_raygen_closesthit_miss_callable)]
void CallShader(uint shaderIndex, inout Payload payload)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "CallShader";
    case glsl:
        {
            [__vulkanCallablePayload]
            static Payload p;

            p = payload;
            __executeCallable(shaderIndex, __callablePayloadLocation(p));
            payload = p;
        }
    case spirv:
        {
            [__vulkanCallablePayload]
            static Payload p;

            p = payload;
            spirv_asm 
            {
                OpExecuteCallableKHR $shaderIndex &p
            };
            payload = p;
        }
    }
}

// 10.3.2

// Some functions only accept a "struct type" parameter. The 
// following function addresses this issue by transforming non-struct
// parameters into a struct. 
// side effect typed use locations (`inout`,`out`, etc.) are managed.
__generic<T>
__intrinsic_op($(kIROp_ForceVarIntoStructTemporarily))
Ref<T> __forceVarIntoStructTemporarily(inout T maybeStruct);

// Some functions require a struct type which is decorated with a [raypayload]
// attribute. This will do the same as __forceVarIntoStructTemporarily and also
// ensure that the struct type in question is decorated appropriately.
__generic<T>
__intrinsic_op($(kIROp_ForceVarIntoRayPayloadStructTemporarily))
Ref<T> __forceVarIntoRayPayloadStructTemporarily(inout T maybeStruct);

__generic<payload_t>
[require(hlsl, raytracing)]
void __traceRayHLSL(
        RaytracingAccelerationStructure AccelerationStructure,
        uint RayFlags,
        uint InstanceInclusionMask,
        uint RayContributionToHitGroupIndex,
        uint MultiplierForGeometryContributionToHitGroupIndex,
        uint MissShaderIndex,
        RayDesc Ray,
        inout payload_t Payload)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "TraceRay";
    }
}

[require(glsl, raytracing_raygen_closesthit_miss)]
void __traceRay(
    RaytracingAccelerationStructure AccelerationStructure,
    uint                            RayFlags,
    uint                            InstanceInclusionMask,
    uint                            RayContributionToHitGroupIndex,
    uint                            MultiplierForGeometryContributionToHitGroupIndex,
    uint                            MissShaderIndex,
    float3                          Origin,
    float                           TMin,
    float3                          Direction,
    float                           TMax,
    int                             PayloadLocation)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "traceRayEXT";
    }
}

// TODO: Slang's parsing logic currently puts modifiers on
// the `GenericDecl` rather than the inner decl when
// using our default syntax, which seems wrong. We need
// to fix this, but for now using the expanded `__generic`
// syntax works in a pinch.
//
__generic<Payload>
[__readNone]
__intrinsic_op($(kIROp_GetVulkanRayTracingPayloadLocation))
int __rayPayloadLocation(__ref Payload payload);

/// Traces a ray through the acceleration structure.
/// @param AccelerationStructure The acceleration structure to traverse
/// @param RayFlags Flags controlling ray behavior
/// @param InstanceInclusionMask Mask for filtering instance visibility
/// @param RayContributionToHitGroupIndex Offset for hit group indexing
/// @param MultiplierForGeometryContributionToHitGroupIndex Multiplier for geometry-based hit group indexing
/// @param MissShaderIndex Index of the miss shader to execute if no hit is found
/// @param Ray Description of the ray to trace
/// @param Payload Structure for passing data between shaders
/// @remarks Core ray tracing function for initiating traversal
/// @category raytracing
[ForceInline]
__generic<payload_t>
[require(cuda_glsl_hlsl_spirv, raytracing_raygen_closesthit_miss)]
void TraceRay(
    RaytracingAccelerationStructure AccelerationStructure,
    uint                            RayFlags,
    uint                            InstanceInclusionMask,
    uint                            RayContributionToHitGroupIndex,
    uint                            MultiplierForGeometryContributionToHitGroupIndex,
    uint                            MissShaderIndex,
    RayDesc                         Ray,
    inout payload_t                 Payload)
{
    __target_switch
    {
    case hlsl:
        __traceRayHLSL(
            AccelerationStructure,
            RayFlags,
            InstanceInclusionMask,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            Ray,
            __forceVarIntoRayPayloadStructTemporarily(Payload));
        return;
    case cuda: __intrinsic_asm "traceOptiXRay";
    case glsl:
    {
        [__vulkanRayPayload]
        static payload_t p;

        p = Payload;
        __traceRay(
            AccelerationStructure,
            RayFlags,
            InstanceInclusionMask,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            Ray.Origin,
            Ray.TMin,
            Ray.Direction,
            Ray.TMax,
            __rayPayloadLocation(p));
        Payload = p;
    }
    case spirv:
    {
        [__vulkanRayPayload]
        static payload_t p;

        p = Payload;
        let origin = Ray.Origin;
        let direction = Ray.Direction;
        let tmin = Ray.TMin;
        let tmax = Ray.TMax;
        spirv_asm 
        {
            OpTraceRayKHR 
                /**/ $AccelerationStructure
                /**/ $RayFlags
                /**/ $InstanceInclusionMask
                /**/ $RayContributionToHitGroupIndex
                /**/ $MultiplierForGeometryContributionToHitGroupIndex
                /**/ $MissShaderIndex
                /**/ $origin
                /**/ $tmin
                /**/ $direction
                /**/ $tmax
                /**/ &p;
        };
        Payload = p;
    }
    }
}

// NOTE!
// The name of the following functions may change when DXR supports
// a feature similar to the `GL_NV_ray_tracing_motion_blur` extension
//
// https://github.com/KhronosGroup/GLSL/blob/master/extensions/nv/GLSL_NV_ray_tracing_motion_blur.txt

__generic<payload_t>
[require(hlsl, raytracing_motionblur)]
void __traceMotionRayHLSL(
        RaytracingAccelerationStructure AccelerationStructure,
        uint RayFlags,
        uint InstanceInclusionMask,
        uint RayContributionToHitGroupIndex,
        uint MultiplierForGeometryContributionToHitGroupIndex,
        uint MissShaderIndex,
        RayDesc Ray,
        float CurrentTime,
        inout payload_t Payload)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "TraceMotionRay";
    }
}

__glsl_extension(GL_NV_ray_tracing_motion_blur)
[require(glsl, raytracing_motionblur_raygen_closesthit_miss)]
void __traceMotionRay(
    RaytracingAccelerationStructure AccelerationStructure,
    uint                            RayFlags,
    uint                            InstanceInclusionMask,
    uint                            RayContributionToHitGroupIndex,
    uint                            MultiplierForGeometryContributionToHitGroupIndex,
    uint                            MissShaderIndex,
    float3                          Origin,
    float                           TMin,
    float3                          Direction,
    float                           TMax,
    float                           CurrentTime,
    int                             PayloadLocation)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "traceRayMotionNV";
    }
}

/// Traces a ray with motion blur support through the acceleration structure.
/// @param AccelerationStructure The acceleration structure to traverse
/// @param RayFlags Flags controlling ray behavior
/// @param InstanceInclusionMask Mask for filtering instance visibility
/// @param RayContributionToHitGroupIndex Offset for hit group indexing
/// @param MultiplierForGeometryContributionToHitGroupIndex Multiplier for geometry-based hit group indexing
/// @param MissShaderIndex Index of the miss shader to execute if no hit is found
/// @param Ray Description of the ray to trace
/// @param CurrentTime Time value for motion blur interpolation
/// @param Payload Structure for passing data between shaders
/// @remarks Extended version of TraceRay with motion blur support
/// @category raytracing
[ForceInline]
[require(glsl_hlsl_spirv, raytracing_motionblur_raygen_closesthit_miss)]
__generic<payload_t>
void TraceMotionRay(
    RaytracingAccelerationStructure AccelerationStructure,
    uint                            RayFlags,
    uint                            InstanceInclusionMask,
    uint                            RayContributionToHitGroupIndex,
    uint                            MultiplierForGeometryContributionToHitGroupIndex,
    uint                            MissShaderIndex,
    RayDesc                         Ray,
    float                           CurrentTime,
    inout payload_t                 Payload)
{
    __target_switch
    {
    case hlsl:
        __traceMotionRayHLSL(
            AccelerationStructure,
            RayFlags,
            InstanceInclusionMask,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            Ray,
            CurrentTime,
            __forceVarIntoRayPayloadStructTemporarily(Payload));
        return;
    case glsl:
    {
        [__vulkanRayPayload]
        static payload_t p;

        p = Payload;
        __traceMotionRay(
            AccelerationStructure,
            RayFlags,
            InstanceInclusionMask,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            Ray.Origin,
            Ray.TMin,
            Ray.Direction,
            Ray.TMax,
            CurrentTime,
            __rayPayloadLocation(p));
        Payload = p;
    }
    case spirv:
    {
        [__vulkanRayPayload]
        static payload_t p;
        
        let origin = Ray.Origin;
        let direction = Ray.Direction;
        let tmin = Ray.TMin;
        let tmax = Ray.TMax;

        p = Payload;
        spirv_asm 
        {
            OpCapability RayTracingMotionBlurNV;
            OpExtension "SPV_NV_ray_tracing_motion_blur";

            OpTraceRayMotionNV
                /**/ $AccelerationStructure
                /**/ $RayFlags
                /**/ $InstanceInclusionMask
                /**/ $RayContributionToHitGroupIndex
                /**/ $MultiplierForGeometryContributionToHitGroupIndex
                /**/ $MissShaderIndex
                /**/ $origin
                /**/ $tmin
                /**/ $direction
                /**/ $tmax
                /**/ $CurrentTime
                /**/ &p;
        };
        Payload = p;
    }
    }
}

// 10.3.3

[require(glsl_spirv, raytracing_intersection)]
bool __reportIntersection(float tHit, uint hitKind)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "reportIntersectionEXT";
    case spirv:
        return spirv_asm 
        {
            result:$$bool = OpReportIntersectionKHR $tHit $hitKind;
        };
    }
}

/// Reports a hit from an intersection shader.
/// @param tHit Distance along the ray where the intersection occurred
/// @param hitKind User-defined value identifying the type of hit
/// @param attributes Custom attributes for the intersection
/// @return true if the hit was accepted, false if rejected
/// @remarks Used in custom intersection shaders to report primitive intersections
/// @category raytracing
__generic<A>
[ForceInline]
[require(glsl_hlsl_spirv, raytracing_intersection)]
bool ReportHit(float tHit, uint hitKind, A attributes)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "ReportHit($0, $1, $2)";
    case glsl:
    case spirv:
        [__vulkanHitAttributes]
        static A a;
        a = attributes;
        return __reportIntersection(tHit, hitKind);
    }
}

/// Reports a hit optimized for OptiX.
/// @param tHit Distance along the ray where the intersection occurred
/// @param hitKind User-defined value identifying the type of hit
/// @param attribs Attribute values for the intersection
/// @return true if the hit was accepted, false if rejected
/// @remarks OptiX-specific version of ReportHit with optimized attribute handling
/// @category raytracing
__generic<each T : __BuiltinIntegerType>
[ForceInline]
[require(cuda_glsl_hlsl_spirv, raytracing_intersection)]
bool ReportHitOptix(float tHit, uint hitKind, expand each T attribs)
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "optixReportIntersection";
    default:
        return ReportHit(tHit, hitKind, makeTuple(expand each attribs));
    }
}

// 10.3.4
/// Ignores the current intersection and continues traversal.
/// @remarks Used in any-hit shaders to reject potential intersections
/// @category raytracing
[require(cuda_glsl_hlsl_spirv, raytracing_anyhit)]
void IgnoreHit()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "IgnoreHit";
    case glsl: __intrinsic_asm "ignoreIntersectionEXT;";
    case cuda: __intrinsic_asm "optixIgnoreIntersection";
    case spirv: 
        spirv_asm 
        { 
            OpIgnoreIntersectionKHR; %_ = OpLabel 
        };
    }
}

// 10.3.5
/// Accepts the current intersection and terminates further traversal.
/// @remarks Used in any-hit shaders to immediately accept an intersection
/// @category raytracing
[require(cuda_glsl_hlsl_spirv, raytracing_anyhit)]
void AcceptHitAndEndSearch()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "AcceptHitAndEndSearch";
    case glsl: __intrinsic_asm "terminateRayEXT;";
    case cuda: __intrinsic_asm "optixTerminateRay";
    case spirv: 
        spirv_asm 
        { 
            OpTerminateRayKHR; %_ = OpLabel 
        };
    }
}

// 10.4 - System Values and Special Semantics

// TODO: Many of these functions need to be restricted so that
// they can only be accessed from specific stages.

// 10.4.1 - Ray Dispatch System Values

/// Returns the current ray dispatch coordinates.
/// @return 3D index of the current ray being processed
/// @remarks Available in all ray tracing shader stages
/// @category raytracing
[NonUniformReturn]
[require(cuda_glsl_hlsl_spirv, raytracing_allstages)]
uint3 DispatchRaysIndex()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "DispatchRaysIndex";
    case glsl: __intrinsic_asm "(gl_LaunchIDEXT)";
    case cuda: __intrinsic_asm "optixGetLaunchIndex";
    case spirv:
        return spirv_asm 
        {
            result:$$uint3 = OpLoad builtin(LaunchIdKHR:uint3);
        };
    }
}

/// Returns the dimensions of the ray dispatch.
/// @return 3D dimensions of the ray dispatch grid
/// @remarks Available in all ray tracing shader stages
/// @category raytracing
[require(cuda_glsl_hlsl_spirv, raytracing_allstages)]
uint3 DispatchRaysDimensions()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "DispatchRaysDimensions";
    case glsl: __intrinsic_asm "(gl_LaunchSizeEXT)";
    case cuda: __intrinsic_asm "optixGetLaunchDimensions";
    case spirv:
        return spirv_asm 
        {
            result:$$uint3 = OpLoad builtin(LaunchSizeKHR:uint3);
        };
    }
}

// 10.4.2 - Ray System Values

/// Returns the origin of the current ray in world space.
/// @return World-space position where the ray originated
/// @remarks Available in any-hit, closest-hit, intersection, and miss shaders
/// @category raytracing
[NonUniformReturn]
[require(cuda_glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection_miss)]
float3 WorldRayOrigin()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "WorldRayOrigin";
    case glsl: __intrinsic_asm "(gl_WorldRayOriginEXT)";
    case cuda: __intrinsic_asm "optixGetWorldRayOrigin";
    case spirv:
        return spirv_asm 
        {
            result:$$float3 = OpLoad builtin(WorldRayOriginKHR:float3);
        };
    }
}

/// Returns the direction of the current ray in world space.
/// @return Normalized world-space direction vector of the ray
/// @remarks Available in any-hit, closest-hit, intersection, and miss shaders
/// @category raytracing
[NonUniformReturn]
[require(cuda_glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection_miss)]
float3 WorldRayDirection()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "WorldRayDirection";
    case glsl: __intrinsic_asm "(gl_WorldRayDirectionEXT)";
    case cuda: __intrinsic_asm "optixGetWorldRayDirection";
    case spirv:
        return spirv_asm 
        {
            result:$$float3 = OpLoad builtin(WorldRayDirectionKHR:float3);
        };
    }
}

/// Returns the minimum valid intersection distance for the current ray.
/// @return Minimum distance along the ray where intersections are considered
/// @remarks Used to prevent self-intersections and near-plane clipping
/// @category raytracing
[NonUniformReturn]
[require(cuda_glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection_miss)]
float RayTMin()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "RayTMin";
    case glsl: __intrinsic_asm "(gl_RayTminEXT)";
    case cuda: __intrinsic_asm "optixGetRayTmin";
    case spirv:
        return spirv_asm 
        {
            result:$$float = OpLoad builtin(RayTminKHR:float);
        };
    }
}

// Note: The `RayTCurrent()` intrinsic should translate to
// either `gl_HitTNV` (for hit shaders) or `gl_RayTmaxNV`
// (for intersection shaders). Right now we are handling this
// during code emission, for simplicity.
//
// TODO: Once the compiler supports a more refined concept
// of profiles/capabilities and overloading based on them,
// we should simply provide two overloads here, specialized
// to the appropriate Vulkan stages.
//

/// Returns the current intersection distance or maximum ray distance.
/// @return Current t-value for hit shaders or maximum distance for intersection shaders
/// @remarks Interpretation depends on shader stage (hit vs. intersection)
/// @category raytracing
[NonUniformReturn]
[require(cuda_glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection_miss)]
float RayTCurrent()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "RayTCurrent";
    case glsl: __intrinsic_asm "(gl_RayTmaxEXT)";
    case cuda: __intrinsic_asm "optixGetRayTmax";
    case spirv:
        return spirv_asm 
        {
            result:$$float = OpLoad builtin(RayTmaxKHR:float);
        };
    }
}

/// Returns the flags used when tracing the current ray.
/// @return Combination of RAY_FLAG values used for this ray
/// @remarks Allows shaders to modify behavior based on ray trace flags
/// @category raytracing
[require(cuda_glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection_miss)]
uint RayFlags()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "RayFlags";
    case glsl: __intrinsic_asm "(gl_IncomingRayFlagsEXT)";
    case cuda: __intrinsic_asm "optixGetRayFlags";
    case spirv:
        return spirv_asm 
        {
            result:$$uint = OpLoad builtin(IncomingRayFlagsKHR:uint);
        };
    }
}

// 10.4.3 - Primitive/Object Space System Values

/// Returns the index of the current instance in the acceleration structure.
/// @return Zero-based index of the current instance
/// @remarks Available in any-hit, closest-hit, and intersection shaders
/// @category raytracing
[NonUniformReturn]
[require(cuda_glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection)]
uint InstanceIndex()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "InstanceIndex";
    case glsl: __intrinsic_asm "(gl_InstanceID)";
    case cuda: __intrinsic_asm "optixGetInstanceIndex";
    case spirv:
        return spirv_asm 
        {
            result:$$uint = OpLoad builtin(InstanceId:uint);
        };
    }
}

/// Returns the user-provided ID of the current instance.
/// @return Custom instance identifier set during acceleration structure build
/// @remarks Used for instance-specific shader behavior
/// @category raytracing
[NonUniformReturn]
[require(cuda_glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection)]
uint InstanceID()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "InstanceID";
    case glsl: __intrinsic_asm "(gl_InstanceCustomIndexEXT)";
    case cuda: __intrinsic_asm "optixGetInstanceId";
    case spirv:
        return spirv_asm 
        {
            result:$$uint = OpLoad builtin(InstanceCustomIndexKHR:uint);
        };
    }
}

/// Returns the index of the current primitive within its geometry.
/// @return Zero-based index of the intersected primitive
/// @remarks Available in any-hit, closest-hit, and intersection shaders
/// @category raytracing
[NonUniformReturn]
[require(cuda_glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection)]
uint PrimitiveIndex()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "PrimitiveIndex";
    case glsl: __intrinsic_asm "(gl_PrimitiveID)";
    case cuda: __intrinsic_asm "optixGetPrimitiveIndex";
    case spirv:
        return spirv_asm 
        {
            result:$$uint = OpLoad builtin(PrimitiveId:uint);
        };
    }
}

/// Returns the ray origin in object space of the current instance.
/// @return Object-space position where the ray originated
/// @remarks Transformed by the inverse of the instance transform
/// @category raytracing
[NonUniformReturn]
[require(cuda_glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection)]
float3 ObjectRayOrigin()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "ObjectRayOrigin";
    case glsl: __intrinsic_asm "(gl_ObjectRayOriginEXT)";
    case cuda: __intrinsic_asm "optixGetObjectRayOrigin";
    case spirv:
        return spirv_asm 
        {
            result:$$float3 = OpLoad builtin(ObjectRayOriginKHR:float3);
        };
    }
}

/// Returns the ray direction in object space of the current instance.
/// @return Object-space direction vector of the ray
/// @remarks Transformed by the inverse of the instance transform
/// @category raytracing
[NonUniformReturn]
[require(cuda_glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection)]
float3 ObjectRayDirection()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "ObjectRayDirection";
    case glsl: __intrinsic_asm "(gl_ObjectRayDirectionEXT)";
    case cuda: __intrinsic_asm "optixGetObjectRayDirection";
    case spirv:
        return spirv_asm 
        {
            result:$$float3 = OpLoad builtin(ObjectRayDirectionKHR:float3);
        };
    }
}

// TODO: optix has an optixGetObjectToWorldTransformMatrix function that returns 12
// floats by reference.
/// Returns the object-to-world transformation matrix (3x4).
/// @return 3x4 matrix transforming from object to world space
/// @remarks Includes position and orientation of the current instance
/// @category raytracing
[NonUniformReturn]
[require(glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection)]
float3x4 ObjectToWorld3x4()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "ObjectToWorld3x4";
    case glsl: __intrinsic_asm "transpose(gl_ObjectToWorldEXT)";
    case spirv:
        return spirv_asm 
        {
            %mat:$$float4x3 = OpLoad builtin(ObjectToWorldKHR:float4x3);
            result:$$float3x4 = OpTranspose %mat;
        };
    }
}

/// Returns the world-to-object transformation matrix (3x4).
/// @return 3x4 matrix transforming from world to object space
/// @remarks Inverse of the object-to-world transform
/// @category raytracing
[NonUniformReturn]
[require(glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection)]
float3x4 WorldToObject3x4()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "WorldToObject3x4";
    case glsl: __intrinsic_asm "transpose(gl_WorldToObjectEXT)";
    case spirv:
        return spirv_asm 
        {
            %mat:$$float4x3 = OpLoad builtin(WorldToObjectKHR:float4x3);
            result:$$float3x4 = OpTranspose %mat;
        };
    }
}

/// Returns the object-to-world transformation matrix (4x3).
/// @return 4x3 matrix transforming from object to world space
/// @remarks Transposed version of ObjectToWorld3x4
/// @category raytracing
[NonUniformReturn]
[require(glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection)]
float4x3 ObjectToWorld4x3()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "ObjectToWorld4x3";
    case glsl: __intrinsic_asm "(gl_ObjectToWorldEXT)";
    case spirv:
        return spirv_asm 
        {
            result:$$float4x3 = OpLoad builtin(ObjectToWorldKHR:float4x3);
        };
    }
}

/// Returns the world-to-object transformation matrix (4x3).
/// @return 4x3 matrix transforming from world to object space
/// @remarks Transposed version of WorldToObject3x4
/// @category raytracing
[NonUniformReturn]
[require(glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection)]
float4x3 WorldToObject4x3()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "WorldToObject4x3";
    case glsl: __intrinsic_asm "(gl_WorldToObjectEXT)";
    case spirv:
        return spirv_asm 
        {
            result:$$float4x3 = OpLoad builtin(WorldToObjectKHR:float4x3);
        };
    }
}

// NOTE!
// The name of the following functions may change when DXR supports
// a feature similar to the `GL_NV_ray_tracing_motion_blur` extension

/// Returns the current time value for motion blur.
/// @return Time value between 0 and 1 for motion blur interpolation
/// @remarks Available when motion blur extension is enabled
/// @category raytracing
__glsl_extension(GL_NV_ray_tracing_motion_blur)
__glsl_extension(GL_EXT_ray_tracing)
[NonUniformReturn]
[require(glsl_hlsl_spirv, raytracing_motionblur_anyhit_closesthit_intersection_miss)]
float RayCurrentTime()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "RayCurrentTime";
    case glsl:  __intrinsic_asm "(gl_CurrentRayTimeNV)";
    case spirv:
        return spirv_asm 
        {
            result:$$float = OpLoad builtin(CurrentRayTimeNV:float);
        };
    }
}

// Note: The provisional DXR spec included these unadorned
// `ObjectToWorld()` and `WorldToObject()` functions, so
// we will forward them to the new names as a convience
// for users who are porting their code.
//
// TODO: Should we provide a deprecation warning on these
// declarations, so that users can know they aren't coding
// against the final spec?
//
/// Alias for ObjectToWorld3x4.
/// @category raytracing
[NonUniformReturn] float3x4 ObjectToWorld() { return ObjectToWorld3x4(); }
/// Alias for WorldToObject3x4.
/// @category raytracing
[NonUniformReturn] float3x4 WorldToObject() { return WorldToObject3x4(); }

// 10.4.4 - Hit Specific System values
/// Returns the type of intersection that was found.
/// @return Hit kind value (HIT_KIND_TRIANGLE_FRONT_FACE, HIT_KIND_TRIANGLE_BACK_FACE, or custom value)
/// @remarks Available in any-hit and closest-hit shaders
/// @category raytracing
[NonUniformReturn]
[require(cuda_glsl_hlsl_spirv, raytracing_anyhit_closesthit)]
uint HitKind()
{
    __target_switch
    {
    case hlsl:  __intrinsic_asm "HitKind";
    case glsl: __intrinsic_asm "(gl_HitKindEXT)";
    case cuda:  __intrinsic_asm "optixGetHitKind";
    case spirv:
        return spirv_asm 
        {
            result:$$uint = OpLoad builtin(HitKindKHR:uint);
        };
    }
}

// Pre-defined hit kinds (not documented explicitly)
/// Predefined hit kind value for front-facing triangle intersections.
/// @category raytracing
static const uint HIT_KIND_TRIANGLE_FRONT_FACE = 254;

/// Predefined hit kind value for back-facing triangle intersections.
/// @category raytracing
static const uint HIT_KIND_TRIANGLE_BACK_FACE = 255;

//
// Shader Model 6.4
// @public:
//

/// Treats `x` and `y` as 4-component vectors of `UInt8` and computes `dot(x, y) + acc`
/// @category math
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_6_4)]
uint dot4add_u8packed(uint x, uint y, uint acc)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "dot4add_u8packed";
    case wgsl: __intrinsic_asm "(dot4U8Packed($0, $1) + $2)";
    case spirv:
        // OpUDotAccSat cannot be used as there should not be any saturation.
        return spirv_asm
        {
            OpCapability DotProduct;
            OpCapability DotProductInput4x8BitPacked;
            OpExtension "SPV_KHR_integer_dot_product";
            %dotResult = OpUDot $$uint $x $y 0;
            result:$$uint = OpIAdd %dotResult $acc;
        };
    default:
        uint4 vecX = unpackUint4x8ToUint32(x);
        uint4 vecY = unpackUint4x8ToUint32(y);
        return dot(vecX, vecY) + acc;
    }
}

/// Treats `x` and `y` as 4-component vectors of `int8` and computes `dot(x, y) + acc`
/// @category math
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_6_4)]
int dot4add_i8packed(uint x, uint y, int acc)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "dot4add_i8packed";
    case wgsl: __intrinsic_asm "(dot4I8Packed($0, $1) + $2)";
    case spirv:
        // OpSDottAccSat cannot be used as there should not be any saturation.
        return spirv_asm
        {
            OpCapability DotProduct;
            OpCapability DotProductInput4x8BitPacked;
            OpExtension "SPV_KHR_integer_dot_product";
            %dotResult = OpSDot $$int $x $y 0;
            result:$$int = OpIAdd %dotResult $acc;
        };
    default:
        int4 vecX = unpackInt4x8ToInt32(x);
        int4 vecY = unpackInt4x8ToInt32(y);
        return dot(vecX, vecY) + acc;
    }
}

/// Computes `dot(x, y) + acc`.
/// May not produce infinities or NaNs for intermediate results that overflow the range of `half`
/// @category math
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, sm_6_4)]
float dot2add(half2 x, half2 y, float acc)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "dot2add";
    default:
        return float(dot(x, y)) + acc;
    }
}

//
// Shader Model 6.5
//

//
// Mesh Shaders
//

/// Set the number of output vertices and primitives for a mesh shader invocation.
/// @category meshshading Mesh shading
__glsl_extension(GL_EXT_mesh_shader)
__glsl_version(450)
[require(glsl_hlsl_metal_spirv, meshshading)]
[noRefInline]
void SetMeshOutputCounts(uint vertexCount, uint primitiveCount)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "SetMeshOutputCounts";
    case glsl:
        __intrinsic_asm "SetMeshOutputsEXT";
    case metal:
        __intrinsic_asm "_slang_mesh.set_primitive_count($1)";
    case spirv:
        return spirv_asm
        {
            OpCapability MeshShadingEXT;
            OpExtension "SPV_EXT_mesh_shader";
            OpSetMeshOutputsEXT $vertexCount $primitiveCount;
        };
    }
}

/// Specify the number of downstream mesh shader thread groups to invoke from an amplification shader,
/// and provide the values for per-mesh payload parameters.
/// @return This function doesn't return.
/// @category meshshading
[KnownBuiltin("DispatchMesh")]
[require(glsl_hlsl_metal_spirv, meshshading)]
[noRefInline]
void DispatchMesh<P>(uint threadGroupCountX, uint threadGroupCountY, uint threadGroupCountZ, __ref P meshPayload)
{
    // This function cannot be inlined due to a legalization pass happening mid-way through processing
    // and later more processing happening to the function which requires eventual inlining.
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "DispatchMesh";
    case glsl:
        // This intrinsic doesn't take into account writing meshPayload. That
        // is dealt with separately by 'legalizeDispatchMeshPayloadForGLSL'.
        __intrinsic_asm "EmitMeshTasksEXT($0, $1, $2)";
    case metal:
        __intrinsic_asm "*_slang_mesh_payload = *$3; _slang_mgp.set_threadgroups_per_grid(uint3($0, $1, $2)); return;";
    case spirv:
        return spirv_asm
        {
            OpCapability MeshShadingEXT;
            OpExtension "SPV_EXT_mesh_shader";
            OpEmitMeshTasksEXT $threadGroupCountX $threadGroupCountY $threadGroupCountZ &meshPayload;
            // OpEmitMeshTasksExt is a terminator, so we need to start a new
            // block to hold whatever comes after this intrinsic
            %_ = OpLabel
        };
    }
}

//
// "Sampler feedback" types `FeedbackTexture2D` and `FeedbackTexture2DArray`.
//

// https://microsoft.github.io/DirectX-Specs/d3d/SamplerFeedback.html

// The docs describe these as 'types' but their syntax makes them seem enum like, and enum is a simpler way to implement them
// But slang enums are always 'enum class like', so I use an empty struct type here

[sealed]
[builtin]
interface __BuiltinSamplerFeedbackType {};

/// @category texture_types
[sealed]
__magic_type(FeedbackType, $(int(FeedbackType::Kind::MinMip)))
__target_intrinsic(hlsl, SAMPLER_FEEDBACK_MIN_MIP)
struct SAMPLER_FEEDBACK_MIN_MIP : __BuiltinSamplerFeedbackType, ITexelElement {
    typealias Element = float;
    static const int elementCount = 1;
    __intrinsic_op(0) __init(float x){}
};

/// @category texture_types
[sealed]
__magic_type(FeedbackType, $(int(FeedbackType::Kind::MipRegionUsed)))
__target_intrinsic(hlsl, SAMPLER_FEEDBACK_MIP_REGION_USED)
struct SAMPLER_FEEDBACK_MIP_REGION_USED : __BuiltinSamplerFeedbackType, ITexelElement
{
    typealias Element = float;
    static const int elementCount = 1;
    __intrinsic_op(0) __init(float x){}
};

// All of these objects are write-only resources that point to a special kind of unordered access view meant for sampler feedback.

extension<T> _Texture<T,__Shape2D, 0, 0, 0, $(kCoreModule_ResourceAccessFeedback), 0, 0, 0>
    where T:ITexelElement
    where T:__BuiltinSamplerFeedbackType
{
    // With Clamp

    [require(cpp_hlsl)]
    void WriteSamplerFeedback<S:ITexelElement>(Texture2D<S> tex, SamplerState samp, float2 location, float clamp)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "($0).WriteSamplerFeedback($1, $2, $3, $4)";
        case hlsl: __intrinsic_asm "($0).WriteSamplerFeedback($1, $2, $3, $4)";
        }
    }

    [require(cpp_hlsl)]
    void WriteSamplerFeedbackBias<S:ITexelElement>(Texture2D<S> tex, SamplerState samp, float2 location, float bias, float clamp)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "($0).WriteSamplerFeedbackBias($1, $2, $3, $4, $5)";
        case hlsl: __intrinsic_asm "($0).WriteSamplerFeedbackBias($1, $2, $3, $4, $5)";
        }
    }

    [require(cpp_hlsl)]
    void WriteSamplerFeedbackGrad<S:ITexelElement>(Texture2D<S> tex, SamplerState samp, float2 location, float2 ddx, float2 ddy, float clamp)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "($0).WriteSamplerFeedbackGrad($1, $2, $3, $4, $5, $6)";
        case hlsl: __intrinsic_asm "($0).WriteSamplerFeedbackGrad($1, $2, $3, $4, $5, $6)";
        }
    }

    // Level

    [require(cpp_hlsl)]
    void WriteSamplerFeedbackLevel<S:ITexelElement>(Texture2D<S> tex, SamplerState samp, float2 location, float lod)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "($0).WriteSamplerFeedbackLevel($1, $2, $3, $4)";
        case hlsl: __intrinsic_asm "($0).WriteSamplerFeedbackLevel($1, $2, $3, $4)";
        }
    }

    // Without Clamp

    [require(cpp_hlsl)]
    void WriteSamplerFeedback<S:ITexelElement>(Texture2D<S> tex, SamplerState samp, float2 location)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "($0).WriteSamplerFeedback($1, $2, $3)";
        case hlsl: __intrinsic_asm "($0).WriteSamplerFeedback($1, $2, $3)";
        }
    }

    [require(cpp_hlsl)]
    void WriteSamplerFeedbackBias<S:ITexelElement>(Texture2D<S> tex, SamplerState samp, float2 location, float bias)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "($0).WriteSamplerFeedbackBias($1, $2, $3, $4)";
        case hlsl: __intrinsic_asm "($0).WriteSamplerFeedbackBias($1, $2, $3, $4)";
        }
    }

    [require(cpp_hlsl)]
    void WriteSamplerFeedbackGrad<S:ITexelElement>(Texture2D<S> tex, SamplerState samp, float2 location, float2 ddx, float2 ddy)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "($0).WriteSamplerFeedbackGrad($1, $2, $3, $4, $5)";
        case hlsl: __intrinsic_asm "($0).WriteSamplerFeedbackGrad($1, $2, $3, $4, $5)";
        }
    }
};

extension<T> _Texture<T,__Shape2D, 1, 0, 0, $(kCoreModule_ResourceAccessFeedback), 0, 0, 0>
    where T:__BuiltinSamplerFeedbackType
    where T:ITexelElement
{
    // With Clamp

    [require(cpp_hlsl)]
    void WriteSamplerFeedback<S:ITexelElement>(Texture2DArray<S> texArray, SamplerState samp, float3 location, float clamp)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "($0).WriteSamplerFeedback($1, $2, $3, $4)";
        case hlsl: __intrinsic_asm "($0).WriteSamplerFeedback($1, $2, $3, $4)";
        }
    }

    [require(cpp_hlsl)]
    void WriteSamplerFeedbackBias<S:ITexelElement>(Texture2DArray<S> texArray, SamplerState samp, float3 location, float bias, float clamp)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "($0).WriteSamplerFeedbackBias($1, $2, $3, $4, $5)";
        case hlsl: __intrinsic_asm "($0).WriteSamplerFeedbackBias($1, $2, $3, $4, $5)";
        }
    }

    [require(cpp_hlsl)]
    void WriteSamplerFeedbackGrad<S:ITexelElement>(Texture2DArray<S> texArray, SamplerState samp, float3 location, float3 ddx, float3 ddy, float clamp)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "($0).WriteSamplerFeedbackGrad($1, $2, $3, $4, $5, $6)";
        case hlsl: __intrinsic_asm "($0).WriteSamplerFeedbackGrad($1, $2, $3, $4, $5, $6)";
        }
    }

    // Level

    [require(cpp_hlsl)]
    void WriteSamplerFeedbackLevel<S:ITexelElement>(Texture2DArray<S> texArray, SamplerState samp, float3 location, float lod)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "($0).WriteSamplerFeedbackLevel($1, $2, $3, $4)";
        case hlsl: __intrinsic_asm "($0).WriteSamplerFeedbackLevel($1, $2, $3, $4)";
        }
    }

    // Without Clamp

    [require(cpp_hlsl)]
    void WriteSamplerFeedback<S:ITexelElement>(Texture2DArray<S> texArray, SamplerState samp, float3 location)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "($0).WriteSamplerFeedback($1, $2, $3)";
        case hlsl: __intrinsic_asm "($0).WriteSamplerFeedback($1, $2, $3)";
        }
    }

    [require(cpp_hlsl)]
    void WriteSamplerFeedbackBias<S:ITexelElement>(Texture2DArray<S> texArray, SamplerState samp, float3 location, float bias)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "($0).WriteSamplerFeedbackBias($1, $2, $3, $4)";
        case hlsl: __intrinsic_asm "($0).WriteSamplerFeedbackBias($1, $2, $3, $4)";
        }
    }

    [require(cpp_hlsl)]
    void WriteSamplerFeedbackGrad<S:ITexelElement>(Texture2DArray<S> texArray, SamplerState samp, float3 location, float3 ddx, float3 ddy)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "($0).WriteSamplerFeedbackGrad($1, $2, $3, $4, $5)";
        case hlsl: __intrinsic_asm "($0).WriteSamplerFeedbackGrad($1, $2, $3, $4, $5)";
        }
    }
};

//
// DXR 1.1 and `TraceRayInline` support
//

/// Returns the index of the geometry that was hit in an intersection, any-hit, or closest-hit shader.
/// @return Zero-based index of the geometry in the current instance
/// @remarks Available in intersection, any-hit, and closest-hit shaders
/// @category raytracing
__glsl_extension(GL_EXT_ray_tracing)
[NonUniformReturn]
[require(glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection)]
uint GeometryIndex()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "GeometryIndex";
    case glsl: __intrinsic_asm "(gl_GeometryIndexEXT)";
    case spirv: return spirv_asm {
            result:$$uint = OpLoad builtin(RayGeometryIndexKHR:uint);
        };
    }
}

/// Get the vertex positions of the currently hit triangle in any-hit or closest-hit shader.
/// https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_ray_tracing_position_fetch.txt
/// @param index Index of the vertex (0-2)
/// @return World-space position of the specified vertex
/// @remarks Requires ray tracing position fetch extension
/// @see GL_EXT_ray_tracing_position_fetch
/// @category raytracing
__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_EXT_ray_tracing_position_fetch)
[ForceInline]
[require(glsl_spirv, raytracing_position)]
float3 HitTriangleVertexPosition(uint index)
{
    __target_switch
    {
        case glsl:
            __intrinsic_asm "gl_HitTriangleVertexPositionsEXT[$0]";
        case spirv:
            return spirv_asm {
                OpCapability RayTracingKHR;
                OpCapability RayTracingPositionFetchKHR;
                OpExtension "SPV_KHR_ray_tracing";
                OpExtension "SPV_KHR_ray_tracing_position_fetch";
                %_ptr_Input_v3float = OpTypePointer Input $$float3;
                %addr : %_ptr_Input_v3float = OpAccessChain builtin(HitTriangleVertexPositionsKHR:float3[3]) $index;
                result:$$float3 = OpLoad %addr;
            };
    }
}

/// Status indicating whether and what type of hit has been committed in a RayQuery.
/// @category raytracing
typedef uint COMMITTED_STATUS;

/// Indicates no hit has been committed yet.
/// @category raytracing
static const COMMITTED_STATUS COMMITTED_NOTHING = 0;

/// Closest hit is a triangle.
/// This could be an opaque triangle hit found by the fixed-function
/// traversal and intersection implementation, or a non-opaque
/// triangle hit committed by user code with `RayQuery.CommitNonOpaqueTriangleHit`.
/// @category raytracing
static const COMMITTED_STATUS COMMITTED_TRIANGLE_HIT = 1;

/// Closest hit is a procedural primitive.
/// A procedural hit primitive is committed using `RayQuery.CommitProceduralPrimitiveHit`.
/// @category raytracing
static const COMMITTED_STATUS COMMITTED_PROCEDURAL_PRIMITIVE_HIT = 2;

/// Type of candidate hit that a `RayQuery` is pausing at.
/// A `RayQuery` can automatically commit hits with opaque triangles,
/// but yields to user code for other hits to allow them to be
/// dismissed or committed.
/// @category raytracing
typedef uint CANDIDATE_TYPE;

/// Candidate hit is a non-opaque triangle.
/// @category raytracing
static const CANDIDATE_TYPE CANDIDATE_NON_OPAQUE_TRIANGLE = 0;

/// Candidate hit is a procedural primitive.
/// @category raytracing
static const CANDIDATE_TYPE CANDIDATE_PROCEDURAL_PRIMITIVE = 1;

/// Handle to state of an in-progress ray-tracing query.
/// The ray query is effectively a coroutine that user shader
/// code can resume to continue tracing the ray, and which yields
/// back to the user code at interesting events along the ray.
//
/// Note: The treatment of the `RayQuery` type in Slang does not
/// perfectly match its semantics in vanilla HLSL in some corner
/// cases. Specifically, a `RayQuery` in vanilla HLSL is an
/// opaque handle to mutable storage, and assigning a `RayQuery`
/// or passing one as a parameter will only copy the *handle*,
/// potentially resulting in aliasing of the underlying mutable
/// storage.
///
/// In contrast, Slang considers a `RayQuery` to own its mutable
/// state, and (because the API does not support cloning of queries),
/// `RayQuery` values are non-copyable (aka "move-only").
///
/// The main place where this arises as a consideration is when
/// passing a `RayQuery` down into a function that will perform
/// mutating operations on it (e.g., `TraceRay` or `Proceed`):
/// ```
///      void myFunc( inout RayQuery<FLAGS> q )
///      {
///          q.Proceed();
///      }
/// ```
/// In Slang, a parameter like `q` above should be declared `inout`.
/// HLSL does not care about whether `q` is declared `inout` or not.
///
///cannot use a cap for struct with unequal target support
///since it will propegate rules to children.
/// @category raytracing Ray-tracing
__glsl_extension(GL_EXT_ray_query)
[__NonCopyableType]
__intrinsic_type($(kIROp_RayQueryType))
struct RayQuery <let rayFlagsGeneric : RAY_FLAG = RAY_FLAG_NONE>
{
    /// Initialize a new ray query in its default state.
    __intrinsic_op($(kIROp_AllocateOpaqueHandle))
    __init();


    __glsl_extension(GL_EXT_ray_query)
    [require(glsl_spirv, rayquery)]
    [mutating]
    void __rayQueryInitializeEXT(
        RaytracingAccelerationStructure accelerationStructure,
        RAY_FLAG                        rayFlags,
        uint                            instanceInclusionMask,
        float3                          origin,
        float                           tMin,
        float3                          direction,
        float                           tMax)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryInitializeEXT($0, $1, $2, $3, $4, $5, $6, $7)";
        case spirv:
            spirv_asm {
                OpRayQueryInitializeKHR &this $accelerationStructure $rayFlags $instanceInclusionMask $origin $tMin $direction $tMax;
            };
        }
    }

    /// Initialize a ray-tracing query.
    ///
    /// This method may be called on a "fresh" ray query, or
    /// on one that is already tracing a ray. In the latter
    /// case any state related to the ray previously being
    /// traced is overwritten.
    ///
    /// The `rayFlags` here will be bitwise ORed with
    /// the `rayFlags` passed as a generic argument to
    /// `RayQuery` to get the effective ray flags, which
    /// must obey any API-imposed restrictions.
    ///
    /// @param accelerationStructure Acceleration structure to traverse
    /// @param rayFlags Additional flags for this trace (combined with rayFlagsGeneric)
    /// @param instanceInclusionMask Mask for filtering instance visibility
    /// @param ray Description of ray parameters (origin, direction, tMin, tMax)
    [__unsafeForceInlineEarly]
    [mutating]
    [require(glsl_hlsl_spirv, rayquery)]
    void TraceRayInline(
        RaytracingAccelerationStructure accelerationStructure,
        RAY_FLAG                        rayFlags,
        uint                            instanceInclusionMask,
        RayDesc                         ray)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".TraceRayInline";
        case glsl:
        case spirv:
            __rayQueryInitializeEXT(
                accelerationStructure,
                rayFlags | rayFlagsGeneric,
                instanceInclusionMask,
                ray.Origin,
                ray.TMin,
                ray.Direction,
                ray.TMax);
        }
    }
    
    /// Resume the ray query coroutine.
    ///
    /// If the coroutine suspends because of encountering
    /// a candidate hit that cannot be resolved with fixed-funciton
    /// logic, this function returns `true`, and the `Candidate*()`
    /// functions should be used by application code to resolve
    /// the candidate hit (by either committing or ignoring it).
    ///
    /// If the coroutine terminates because traversal is
    /// complete (or has been aborted), this function returns
    /// `false`, and application code should use the `Committed*()`
    /// functions to appropriately handle the closest hit (it any)
    /// that was found.
    ///
    /// @return true if a candidate hit needs evaluation, false if traversal is complete
    /// @remarks When true is returned, use Candidate* methods to evaluate the hit
    __glsl_extension(GL_EXT_ray_query)
    [mutating]
    [require(glsl_hlsl_spirv, rayquery)]
    bool Proceed()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Proceed";
        case glsl: __intrinsic_asm "rayQueryProceedEXT";
        case spirv: return spirv_asm
            {
                result:$$bool = OpRayQueryProceedKHR &this
            };
        }
    }

    /// Terminate ray traversal immediately.
    /// @remarks Causes subsequent Proceed() calls to return false
    __glsl_extension(GL_EXT_ray_query)
    [mutating]
    [require(glsl_hlsl_spirv, rayquery)]
    void Abort()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".Abort";
        case glsl: __intrinsic_asm "rayQueryTerminateEXT";
        case spirv: spirv_asm { OpRayQueryTerminateKHR &this };
        }
    }

    /// Commit the current non-opaque triangle hit as the closest hit.
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [mutating]
    [require(glsl_hlsl_spirv, rayquery)]
    void CommitNonOpaqueTriangleHit()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".CommitNonOpaqueTriangleHit";
        case glsl: __intrinsic_asm "rayQueryConfirmIntersectionEXT";
        case spirv: spirv_asm { OpRayQueryConfirmIntersectionKHR &this };
        }
    }

    /// Commit a procedural primitive hit at the specified distance.
    /// @param t Distance along the ray where the hit occurred
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [mutating]
    [require(glsl_hlsl_spirv, rayquery)]
    void CommitProceduralPrimitiveHit(float t)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".CommitProceduralPrimitiveHit";
        case glsl: __intrinsic_asm "rayQueryGenerateIntersectionEXT";
        case spirv: spirv_asm { OpRayQueryGenerateIntersectionKHR &this $t };
        }
    }

    /// Get the type of candidate hit being considered.
    ///
    /// The ray query coroutine will suspend when it encounters
    /// a hit that cannot be resolved with fixed-function logic
    /// (either a non-opaque triangle or a procedural primitive).
    /// In either of those cases, `CandidateType()` will return
    /// the kind of candidate hit that must be resolved by
    /// user code.
    ///
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [NonUniformReturn]
    [require(glsl_hlsl_spirv, rayquery)]
    CANDIDATE_TYPE CandidateType()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".CandidateType";
        case glsl: __intrinsic_asm "rayQueryGetIntersectionTypeEXT($0, false)";
        case spirv:
            uint RayQueryCandidateIntersectionKHR = 0;
            return spirv_asm {
                result:$$CANDIDATE_TYPE = OpRayQueryGetIntersectionTypeKHR &this $RayQueryCandidateIntersectionKHR;
            };
        }
    }

    /// Returns the status of the committed (closest) hit.
    /// @return COMMITTED_STATUS indicating type of committed hit, if any
    /// @remarks Valid after traversal is complete (Proceed() returns false)
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [NonUniformReturn]
    [require(glsl_hlsl_spirv, rayquery)]
    COMMITTED_STATUS CommittedStatus()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".CommittedStatus";
        case glsl: __intrinsic_asm "rayQueryGetIntersectionTypeEXT($0, true)";
        case spirv:
            uint RayQueryCommittedIntersectionKHR = 1;
            return spirv_asm
            {
                result:$$COMMITTED_STATUS = OpRayQueryGetIntersectionTypeKHR &this $RayQueryCommittedIntersectionKHR;
            };
        }
    }

    /// Checks if the candidate procedural primitive is non-opaque.
    /// @return true if the primitive is non-opaque and requires shader evaluation
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [NonUniformReturn]
    [require(glsl_hlsl_spirv, rayquery)]
    bool CandidateProceduralPrimitiveNonOpaque()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".CandidateProceduralPrimitiveNonOpaque";
        case glsl: __intrinsic_asm "(!rayQueryGetIntersectionCandidateAABBOpaqueEXT($0))";
        case spirv:
            uint iCandidateOrCommitted = 0;
            return spirv_asm
            {
                %rr:$$bool = OpRayQueryGetIntersectionCandidateAABBOpaqueKHR &this;
                result:$$bool = OpLogicalNot %rr;
            };
        }
    }

    /// Gets the distance to the candidate triangle hit.
    /// @return t-value along the ray where intersection occurred
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [NonUniformReturn]
    [require(glsl_hlsl_spirv, rayquery)]
    float CandidateTriangleRayT()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".CandidateTriangleRayT";
        case glsl: __intrinsic_asm "rayQueryGetIntersectionTEXT($0, false)";
        case spirv:
            uint iCandidateOrCommitted = 0;
            return spirv_asm
            {
                result:$$float = OpRayQueryGetIntersectionTKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the distance to the committed (closest) hit.
    /// @return t-value along the ray where the closest hit occurred
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [NonUniformReturn]
    [require(glsl_hlsl_spirv, rayquery)]
    float CommittedRayT()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".CommittedRayT";
        case glsl: __intrinsic_asm "rayQueryGetIntersectionTEXT($0, true)";
        case spirv:
            uint iCandidateOrCommitted = 1;
            return spirv_asm
            {
                result:$$float = OpRayQueryGetIntersectionTKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the custom index of the instance containing the candidate hit.
    /// @return User-provided instance identifier
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    int CandidateRayInstanceCustomIndex()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionInstanceCustomIndexEXT($0, false)";
        case spirv:
            uint iCandidateOrCommitted = 0;
            return spirv_asm
            {
                result:$$int = OpRayQueryGetIntersectionInstanceCustomIndexKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the custom index of the instance containing the committed hit.
    /// @return User-provided instance identifier
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    int CommittedRayInstanceCustomIndex()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionInstanceCustomIndexEXT($0, true)";
        case spirv:
            uint iCandidateOrCommitted = 1;
            return spirv_asm
            {
                result:$$int = OpRayQueryGetIntersectionInstanceCustomIndexKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the instance ID of the candidate hit.
    /// @return System-assigned instance identifier
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    int CandidateRayInstanceId()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionInstanceIdEXT($0, false)";
        case spirv:
            uint iCandidateOrCommitted = 0;
            return spirv_asm
            {
                result:$$int = OpRayQueryGetIntersectionInstanceIdKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the instance ID of the committed hit.
    /// @return System-assigned instance identifier
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    int CommittedRayInstanceId()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionInstanceIdEXT($0, true)";
        case spirv:
            uint iCandidateOrCommitted = 1;
            return spirv_asm
            {
                result:$$int = OpRayQueryGetIntersectionInstanceIdKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the shader binding table offset for the instance containing the candidate hit.
    /// @return Offset into the shader binding table for hit group selection
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    uint CandidateRayInstanceShaderBindingTableRecordOffset()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT($0, false)";
        case spirv:
            uint iCandidateOrCommitted = 0;
            return spirv_asm
            {
                result:$$uint = OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the shader binding table offset for the instance containing the committed hit.
    /// @return Offset into the shader binding table for hit group selection
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    uint CommittedRayInstanceShaderBindingTableRecordOffset()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT($0, true)";
        case spirv:
            uint iCandidateOrCommitted = 1;
            return spirv_asm
            {
                result:$$uint = OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the geometry index for the candidate hit.
    /// @return Zero-based index of the geometry in the instance
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    int CandidateRayGeometryIndex()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionGeometryIndexEXT($0, false)";
        case spirv:
            uint iCandidateOrCommitted = 0;
            return spirv_asm
            {
                result:$$int = OpRayQueryGetIntersectionGeometryIndexKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the geometry index for the committed hit.
    /// @return Zero-based index of the geometry in the instance
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    int CommittedRayGeometryIndex()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionGeometryIndexEXT($0, true)";
        case spirv:
            uint iCandidateOrCommitted = 1;
            return spirv_asm
            {
                result:$$int = OpRayQueryGetIntersectionGeometryIndexKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the primitive index for the candidate hit.
    /// @return Zero-based index of the primitive in the geometry
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    int CandidateRayPrimitiveIndex()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionPrimitiveIndexEXT($0, false)";
        case spirv:
            uint iCandidateOrCommitted = 0;
            return spirv_asm
            {
                result:$$int = OpRayQueryGetIntersectionPrimitiveIndexKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the primitive index for the committed hit.
    /// @return Zero-based index of the primitive in the geometry
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    int CommittedRayPrimitiveIndex()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionPrimitiveIndexEXT($0, true)";
        case spirv:
            uint iCandidateOrCommitted = 1;
            return spirv_asm
            {
                result:$$int = OpRayQueryGetIntersectionPrimitiveIndexKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the barycentric coordinates of the candidate hit point.
    /// @return UV barycentric coordinates on the triangle
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    float2 CandidateRayBarycentrics()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionBarycentricsEXT($0, false)";
        case spirv:
            uint iCandidateOrCommitted = 0;
            return spirv_asm
            {
                result:$$float2 = OpRayQueryGetIntersectionBarycentricsKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the barycentric coordinates of the committed hit point.
    /// @return UV barycentric coordinates on the triangle
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    float2 CommittedRayBarycentrics()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionBarycentricsEXT($0, true)";
        case spirv:
            uint iCandidateOrCommitted = 1;
            return spirv_asm
            {
                result:$$float2 = OpRayQueryGetIntersectionBarycentricsKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Checks if the candidate hit is on the front face of a triangle.
    /// @return true if hit is on triangle front face
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    bool CandidateRayFrontFace()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionFrontFaceEXT($0, false)";
        case spirv:
            uint iCandidateOrCommitted = 0;
            return spirv_asm
            {
                result:$$bool = OpRayQueryGetIntersectionFrontFaceKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Checks if the committed hit is on the front face of a triangle.
    /// @return true if hit is on triangle front face
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    bool CommittedRayFrontFace()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionFrontFaceEXT($0, true)";
        case spirv:
            uint iCandidateOrCommitted = 1;
            return spirv_asm
            {
                result:$$bool = OpRayQueryGetIntersectionFrontFaceKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the ray direction in object space for the candidate hit.
    /// @return Direction vector transformed into instance's object space
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    float3 CandidateRayObjectRayDirection()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionObjectRayDirectionEXT($0, false)";
        case spirv:
            uint iCandidateOrCommitted = 0;
            return spirv_asm
            {
                result:$$float3 = OpRayQueryGetIntersectionObjectRayDirectionKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the ray direction in object space for the committed hit.
    /// @return Direction vector transformed into instance's object space
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    float3 CommittedRayObjectRayDirection()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionObjectRayDirectionEXT($0, true)";
        case spirv:
            uint iCandidateOrCommitted = 1;
            return spirv_asm
            {
                result:$$float3 = OpRayQueryGetIntersectionObjectRayDirectionKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the ray origin in object space for the candidate hit.
    /// @return Origin point transformed into instance's object space
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    float3 CandidateRayObjectRayOrigin()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionObjectRayOriginEXT($0, false)";
        case spirv:
            uint iCandidateOrCommitted = 0;
            return spirv_asm
            {
                result:$$float3 = OpRayQueryGetIntersectionObjectRayOriginKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the ray origin in object space for the committed hit.
    /// @return Origin point transformed into instance's object space
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    float3 CommittedRayObjectRayOrigin()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionObjectRayOriginEXT($0, true)";
        case spirv:
            uint iCandidateOrCommitted = 1;
            return spirv_asm
            {
                result:$$float3 = OpRayQueryGetIntersectionObjectRayOriginKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the object-to-world transform matrix for the candidate hit instance.
    /// @return 4x3 matrix transforming from object to world space
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    float4x3 CandidateRayObjectToWorld()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionObjectToWorldEXT($0, false)";
        case spirv:
            uint iCandidateOrCommitted = 0;
            return spirv_asm
            {
                result:$$float4x3 = OpRayQueryGetIntersectionObjectToWorldKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the object-to-world transform matrix for the committed hit instance.
    /// @return 4x3 matrix transforming from object to world space
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    float4x3 CommittedRayObjectToWorld()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionObjectToWorldEXT($0, true)";
        case spirv:
            uint iCandidateOrCommitted = 1;
            return spirv_asm
            {
                result:$$float4x3 = OpRayQueryGetIntersectionObjectToWorldKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the world-to-object transform matrix for the candidate hit instance.
    /// @return 4x3 matrix transforming from world to object space
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    float4x3 CandidateRayWorldToObject()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionWorldToObjectEXT($0, false)";
        case spirv:
            uint iCandidateOrCommitted = 0;
            return spirv_asm
            {
                result:$$float4x3 = OpRayQueryGetIntersectionWorldToObjectKHR &this $iCandidateOrCommitted;
            };
        }
    }

    /// Gets the world-to-object transform matrix for the committed hit instance.
    /// @return 4x3 matrix transforming from world to object space
    /// @remarks GLSL/SPIRV only
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [require(glsl_spirv, rayquery)]
    float4x3 CommittedRayWorldToObject()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersectionWorldToObjectEXT($0, true)";
        case spirv:
            uint iCandidateOrCommitted = 1;
            return spirv_asm
            {
                OpRayQueryGetIntersectionWorldToObjectKHR $$float4x3 result &this $iCandidateOrCommitted;
            };
        }
    }
///~

${{{{
    const char* kCandidateCommitted[] = {"Candidate", "Committed"};

    // Access Candidate and Committed Matrices.
    for (uint32_t candidateOrCommitted = 0; candidateOrCommitted < 2; candidateOrCommitted++)
    {
        auto ccName = kCandidateCommitted[candidateOrCommitted];
        auto ccTF = candidateOrCommitted == 0 ? "false" : "true";
}}}}

    __glsl_extension(GL_EXT_ray_query)
    __glsl_extension(GL_EXT_ray_tracing_position_fetch)
    [require(glsl, rayquery_position)]
    [__NoSideEffect]
    void __glslGetIntersectionTriangleVertexPositions$(ccName)(out float3 arr[3])
    {
        __intrinsic_asm "rayQueryGetIntersectionTriangleVertexPositionsEXT($0, $(ccTF), $1)";
    };

    /// Gets the triangle vertex positions for an intersection.
    /// @return Array of three vertex positions in world space
    /// @remarks Requires ray query position fetch extension
    __glsl_extension(GL_EXT_ray_query)
    [require(glsl, rayquery_position)]
    [require(spirv, rayquery_position)]
    [__NoSideEffect]
    float3[3] $(ccName)GetIntersectionTriangleVertexPositions()
    {
        typedef float3[3] float3Arr3;
        __target_switch
        {
        case glsl:
            float3 output[3];
            __glslGetIntersectionTriangleVertexPositions$(ccName)(output);
            return output;
        case spirv:
            uint iCandidateOrCommitted = $(candidateOrCommitted);
            return spirv_asm
            {
                OpCapability RayQueryKHR;
                OpCapability RayQueryPositionFetchKHR;
                OpExtension "SPV_KHR_ray_query";
                OpExtension "SPV_KHR_ray_tracing_position_fetch";
                result: $$float3Arr3 = OpRayQueryGetIntersectionTriangleVertexPositionsKHR &this $iCandidateOrCommitted;
            };
        }
    };

    // CandidateObjectToWorld3x4, CandidateWorldToObject4x3
    // CommittedObjectToWorld3x4, CommittedObjectToWorld4x3
    ${{{{
    const char* kRayQueryMatrixNames[] = {"ObjectToWorld", "WorldToObject"};
    for (auto matName : kRayQueryMatrixNames) {
    }}}}

    /// Gets the object-to-world transform as a 3x4 matrix.
    /// @return 3x4 matrix transforming from object to world space
    /// @remarks Available for both candidate and committed hits
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [NonUniformReturn]
    [require(glsl_hlsl_spirv, rayquery)]
    float3x4 $(ccName)$(matName)3x4()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "transpose(rayQueryGetIntersection$(matName)EXT($0, $(ccTF)))";
        case hlsl: __intrinsic_asm ".$(ccName)$(matName)3x4";
        case spirv:
            uint iCandidateOrCommitted = $(candidateOrCommitted);
            return spirv_asm 
            {
                %m:$$float4x3 = OpRayQueryGetIntersection$(matName)KHR &this $iCandidateOrCommitted;
                result:$$float3x4 = OpTranspose %m;
            };
        }
    }

    /// Gets the world-to-object transform as a 4x3 matrix.
    /// @return 4x3 matrix transforming from world to object space
    /// @remarks Available for both candidate and committed hits
    __glsl_extension(GL_EXT_ray_query)
    [__readNone]
    [NonUniformReturn]
    [require(glsl_hlsl_spirv, rayquery)]
    float4x3 $(ccName)$(matName)4x3()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "rayQueryGetIntersection$(matName)EXT($0, $(ccTF))";
        case hlsl: __intrinsic_asm ".$(ccName)$(matName)4x3";
        case spirv:
            uint iCandidateOrCommitted = $(candidateOrCommitted);
            return spirv_asm 
            {
                result:$$float4x3 = OpRayQueryGetIntersection$(matName)KHR &this $iCandidateOrCommitted;
            };
        }
    }

${{{{
    } // ObjectToWorld/WorldToObject.

    // Access Candidate and Committed properties.
    struct RayQueryMethodEntry
    {
        const char* type;
        const char* hlslName;
        const char* glslName;
    };
    const RayQueryMethodEntry rayQueryMethods[] = {
        {"uint", "InstanceIndex", "InstanceId"},
        {"uint", "InstanceID", "InstanceCustomIndex"},
        {"uint", "PrimitiveIndex", "PrimitiveIndex"},
        {"uint", "GeometryIndex", "GeometryIndex"},
        {"uint", "InstanceContributionToHitGroupIndex", "InstanceShaderBindingTableRecordOffset"},
        {"float3", "ObjectRayOrigin", "ObjectRayOrigin"},
        {"float3", "ObjectRayDirection", "ObjectRayDirection"},
        {"bool", "TriangleFrontFace", "FrontFace"},
        {"float2", "TriangleBarycentrics", "Barycentrics"},
    };
    for (auto method : rayQueryMethods) {
}}}}

    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [NonUniformReturn]
    [require(glsl_hlsl_spirv, rayquery)]
    $(method.type) $(ccName)$(method.hlslName)()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".$(ccName)$(method.hlslName)";
        case glsl: __intrinsic_asm "rayQueryGetIntersection$(method.glslName)EXT($0, $(ccTF))";
        case spirv:
            uint iCandidateOrCommitted = $(candidateOrCommitted);
            return spirv_asm 
            {
                result:$$$(method.type) = OpRayQueryGetIntersection$(method.glslName)KHR &this $iCandidateOrCommitted;
            };
        }
    }
${{{{
    } // Candidate/Committed properties.
    } // for ("Candidate", "Committed")
}}}}

    // Access properties of the ray being traced.

    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [NonUniformReturn]
    [require(glsl_hlsl_spirv, rayquery)]
    uint RayFlags()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".RayFlags";
        case glsl: __intrinsic_asm "rayQueryGetRayFlagsEXT";
        case spirv:
            return spirv_asm 
            {
                result:$$uint = OpRayQueryGetRayFlagsKHR &this;
            };
        }
    }

    /// Gets the world-space origin of the ray.
    /// @return Starting point of the ray in world space
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [NonUniformReturn]
    [require(glsl_hlsl_spirv, rayquery)]
    float3 WorldRayOrigin()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".WorldRayOrigin";
        case glsl: __intrinsic_asm "rayQueryGetWorldRayOriginEXT";
        case spirv:
            return spirv_asm 
            {
                result:$$float3 = OpRayQueryGetWorldRayOriginKHR &this;
            };
        }
    }

    /// Gets the world-space direction of the ray.
    /// @return Normalized direction vector in world space
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [NonUniformReturn]
    [require(glsl_hlsl_spirv, rayquery)]
    float3 WorldRayDirection()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".WorldRayDirection";
        case glsl: __intrinsic_asm "rayQueryGetWorldRayDirectionEXT";
        case spirv:
            return spirv_asm 
            {
                result:$$float3 = OpRayQueryGetWorldRayDirectionKHR &this;
            };
        }
    }

    /// Gets the minimum valid distance along the ray.
    /// @return Minimum t-value for considering intersections
    /// @remarks Used to prevent self-intersections
    __glsl_extension(GL_EXT_ray_query)
    [__NoSideEffect]
    [NonUniformReturn]
    [require(glsl_hlsl_spirv, rayquery)]
    float RayTMin()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".RayTMin";
        case glsl: __intrinsic_asm "rayQueryGetRayTMinEXT";
        case spirv:
            return spirv_asm 
            {
                result:$$float = OpRayQueryGetRayTMinKHR &this;
            };
        }
    };
}

//
// SubpassInput
//

/// @category stage_io
__magic_type(SubpassInputType)
__intrinsic_type($(kIROp_SubpassInputType))
[require(glsl_hlsl_spirv, subpass)]
struct __SubpassImpl<T, let isMS:int>
{
}

/// @category stage_io
__generic<T = float4, let isMS:int=0>
typealias SubpassInput = __SubpassImpl<T, isMS>;

__generic<T>
extension __SubpassImpl<T, 0>
{
    [ForceInline]
    [require(glsl_hlsl_spirv, subpass)]
    T SubpassLoad()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "$0.SubpassLoad()";
        case glsl: __intrinsic_asm "subpassLoad($0)";
        case spirv:
        {
            let zeroVec = int2(0);
            return spirv_asm
            {
                OpCapability StorageImageReadWithoutFormat;
                result:$$T = OpImageRead $this $zeroVec
            };
        }
        }
    }
}

/// @category stage_io
__generic<T = float4, let isMS:int=1>
typealias SubpassInputMS = __SubpassImpl<T, isMS>;

__generic<T>
extension __SubpassImpl<T, 1>
{
    [ForceInline]
    [require(glsl_hlsl_spirv, subpass)]
    T SubpassLoad(int sample)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "$0.SubpassLoad($1)";
        case glsl: __intrinsic_asm "subpassLoad($0, $1)";
        case spirv:
        {
            let zeroVec = int2(0);
            return spirv_asm
            {
                OpCapability StorageImageReadWithoutFormat;
                result:$$T = OpImageRead $this $zeroVec Sample $sample
            };
        }
        }
    }
}

///
/// Shader Execution Reordering (SER)
///
/// NOTE! This API is currently experimental and may change in the future as SER is made available
/// in different APIs and downstream compilers.
///
/// Based on the NVAPI on D3D12 only currently.
///
/// White paper on SER on NVAPI https://developer.nvidia.com/sites/default/files/akamai/gameworks/ser-whitepaper.pdf
///
/// The NVAPI headers (R520) required for this functionality to work can be found here...
///
/// https://developer.nvidia.com/rtx/path-tracing/nvapi/get-started
///
/// For VK the specification is currently in this PR
///
/// https://github.com/KhronosGroup/GLSL/pull/196/files

/// Internal helper functions

// This is a bit of a hack for GLSL HitObjectAttributes
// It relies on [ForceInline] removing the surrounding function and just inserting the *contained* `t` as a global
// The __ref should indicate the desire for the returned value to not be a copy of t, but *t*.
// In practive __ref doesn't have this effect in practice.
// 
// We need this to be able access the payload outside of a function (which is all that TraceRay for example needs)
// We access the HitObjectAttributes via this function for the desired type, and it acts *as if* it's just an access
// to the global t.
[ForceInline]
Ref<T> __hitObjectAttributes<T>()
{
    [__vulkanHitObjectAttributes]   
    static T t;
    return t;
}
[ForceInline]
__Addr<T> __allocHitObjectAttributes<T>()
{
    [__vulkanHitObjectAttributes]
    static T t;
    return __get_addr(t);
}

// Next is the custom intrinsic that will compute the hitObjectAttributes location
// for GLSL-based targets.
//
__generic<Attributes>
__intrinsic_op($(kIROp_GetVulkanRayTracingPayloadLocation))
int __hitObjectAttributesLocation(__ref Attributes attributes);

/// Immutable data type representing a ray hit or a miss. Can be used to invoke hit or miss shading,
/// or as a key in ReorderThread. Created by one of several methods described below. HitObject
/// and its related functions are available in raytracing shader types only.
/// @category raytracing Ray-tracing
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GL_EXT_ray_tracing)
[__NonCopyableType]
__intrinsic_type($(kIROp_HitObjectType))
struct HitObject
{
    __intrinsic_op($(kIROp_AllocateOpaqueHandle))
    __init();

        /// Executes ray traversal (including anyhit and intersection shaders) like TraceRay, but returns the
        /// resulting hit information as a HitObject and does not trigger closesthit or miss shaders.
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    static HitObject TraceRay<payload_t>(
        RaytracingAccelerationStructure AccelerationStructure,
        uint RayFlags,
        uint InstanceInclusionMask,
        uint RayContributionToHitGroupIndex,
        uint MultiplierForGeometryContributionToHitGroupIndex,
        uint MissShaderIndex,
        RayDesc Ray,
        inout payload_t Payload)
    {
        __target_switch
        {
        case hlsl:
            {
                HitObject hitObj;
                __hlslTraceRay(
                    AccelerationStructure, 
                    RayFlags, 
                    InstanceInclusionMask, 
                    RayContributionToHitGroupIndex, 
                    MultiplierForGeometryContributionToHitGroupIndex, 
                    MissShaderIndex, 
                    Ray, 
                    __forceVarIntoRayPayloadStructTemporarily(Payload),
                    hitObj);
                return hitObj;
            }
        case glsl:
            {
                [__vulkanRayPayload]
                static payload_t p;

                // Save the payload
                p = Payload;

                __glslTraceRay(
                    __return_val,
                    AccelerationStructure,
                    RayFlags,                                           // Assumes D3D/VK have some RayFlags values
                    InstanceInclusionMask,                              // cullMask
                    RayContributionToHitGroupIndex,                     // sbtRecordOffset
                    MultiplierForGeometryContributionToHitGroupIndex,   // sbtRecordStride
                    MissShaderIndex,
                    Ray.Origin,
                    Ray.TMin,
                    Ray.Direction, 
                    Ray.TMax,
                    __rayPayloadLocation(p));
        
                // Write the payload out
                Payload = p;
            }
        case spirv:
            {
                [__vulkanRayPayload]
                static payload_t p;

                // Save the payload
                p = Payload;

                let origin = Ray.Origin;
                let direction = Ray.Direction;
                let tmin = Ray.TMin;
                let tmax = Ray.TMax;
                spirv_asm 
                {
                    OpExtension "SPV_NV_shader_invocation_reorder";
                    OpCapability ShaderInvocationReorderNV;
                    OpHitObjectTraceRayNV
                        /**/ &__return_val
                        /**/ $AccelerationStructure
                        /**/ $RayFlags
                        /**/ $InstanceInclusionMask
                        /**/ $RayContributionToHitGroupIndex
                        /**/ $MultiplierForGeometryContributionToHitGroupIndex
                        /**/ $MissShaderIndex
                        /**/ $origin
                        /**/ $tmin
                        /**/ $direction
                        /**/ $tmax
                        /**/ &p;
                };

                // Write the payload out
                Payload = p;
            }
        }
    }

        /// Executes motion ray traversal (including anyhit and intersection shaders) like TraceRay, but returns the
        /// resulting hit information as a HitObject and does not trigger closesthit or miss shaders.
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_motion_raygen_closesthit_miss)]
    static HitObject TraceMotionRay<payload_t>( 
        RaytracingAccelerationStructure AccelerationStructure, 
        uint RayFlags, 
        uint InstanceInclusionMask, 
        uint RayContributionToHitGroupIndex, 
        uint MultiplierForGeometryContributionToHitGroupIndex, 
        uint MissShaderIndex, 
        RayDesc Ray,
        float CurrentTime,
        inout payload_t Payload)
    {
        __target_switch
        {
        case hlsl:
            __traceMotionRayHLSL(
                AccelerationStructure,
                RayFlags,
                InstanceInclusionMask,
                RayContributionToHitGroupIndex,
                MultiplierForGeometryContributionToHitGroupIndex,
                MissShaderIndex,
                Ray,
                CurrentTime,
                __forceVarIntoRayPayloadStructTemporarily(Payload));
        case glsl:
            {
                [__vulkanRayPayload]
                static payload_t p;

                // Save the payload
                p = Payload;

                __glslTraceMotionRay(
                    __return_val,
                    AccelerationStructure,
                    RayFlags,                                           // Assumes D3D/VK have some RayFlags values
                    InstanceInclusionMask,                              // cullMask
                    RayContributionToHitGroupIndex,                     // sbtRecordOffset
                    MultiplierForGeometryContributionToHitGroupIndex,   // sbtRecordStride
                    MissShaderIndex,
                    Ray.Origin,
                    Ray.TMin,
                    Ray.Direction, 
                    Ray.TMax,
                    CurrentTime,
                    __rayPayloadLocation(p));
        
                // Write the payload out
                Payload = p;
            }
        case spirv:
            {
                [__vulkanRayPayload]
                static payload_t p;

                // Save the payload
                p = Payload;

                let origin = Ray.Origin;
                let direction = Ray.Direction;
                let tmin = Ray.TMin;
                let tmax = Ray.TMax;
                spirv_asm 
                {
                    OpExtension "SPV_NV_shader_invocation_reorder";
                    OpCapability ShaderInvocationReorderNV;
                    OpExtension "SPV_NV_ray_tracing_motion_blur";
                    OpCapability RayTracingMotionBlurNV;
                    OpHitObjectTraceRayMotionNV
                        /**/ &__return_val
                        /**/ $AccelerationStructure
                        /**/ $RayFlags
                        /**/ $InstanceInclusionMask
                        /**/ $RayContributionToHitGroupIndex
                        /**/ $MultiplierForGeometryContributionToHitGroupIndex
                        /**/ $MissShaderIndex
                        /**/ $origin
                        /**/ $tmin
                        /**/ $direction
                        /**/ $tmax
                        /**/ $CurrentTime
                        /**/ &p;
                };
        
                // Write the payload out
                Payload = p;
            }
        }
        
    }

        /// Creates a HitObject representing a hit based on values explicitly passed as arguments, without
        /// tracing a ray. The primitive specified by AccelerationStructure, InstanceIndex, GeometryIndex,
        /// and PrimitiveIndex must exist. The shader table index is computed using the formula used with
        /// TraceRay. The computed index must reference a valid hit group record in the shader table. The
        /// Attributes parameter must either be an attribute struct, such as
        /// BuiltInTriangleIntersectionAttributes, or another HitObject to copy the attributes from.
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    static HitObject MakeHit<attr_t>(
        RaytracingAccelerationStructure AccelerationStructure,
        uint InstanceIndex,
        uint GeometryIndex,
        uint PrimitiveIndex,
        uint HitKind,
        uint RayContributionToHitGroupIndex,
        uint MultiplierForGeometryContributionToHitGroupIndex,
        RayDesc Ray,
        attr_t attributes)
    {
        __target_switch
        {
        case hlsl:
            HitObject hitObj;
            __hlslMakeHit(
                AccelerationStructure, 
                InstanceIndex,
                GeometryIndex,
                PrimitiveIndex,
                HitKind,
                RayContributionToHitGroupIndex,
                MultiplierForGeometryContributionToHitGroupIndex,
                Ray,
                attributes,
                hitObj);
            return hitObj;
        case glsl:
            {
                // Save the attributes
                __hitObjectAttributes<attr_t>() = attributes;

                __glslMakeHit(
                    __return_val,
                    AccelerationStructure,
                    InstanceIndex,
                    PrimitiveIndex,
                    GeometryIndex,
                    HitKind,
                    RayContributionToHitGroupIndex,                         /// sbtRecordOffset?
                    MultiplierForGeometryContributionToHitGroupIndex,       /// sbtRecordStride?
                    Ray.Origin,
                    Ray.TMin,
                    Ray.Direction, 
                    Ray.TMax,
                    __hitObjectAttributesLocation(__hitObjectAttributes<attr_t>()));
            }
        case spirv:
            {
                // Save the attributes
                __Addr<attr_t> attr = __allocHitObjectAttributes<attr_t>();

                *attr = attributes;

                let origin = Ray.Origin;
                let direction = Ray.Direction;
                let tmin = Ray.TMin;
                let tmax = Ray.TMax;
                spirv_asm 
                {
                    OpExtension "SPV_NV_shader_invocation_reorder";
                    OpCapability ShaderInvocationReorderNV;
                    OpHitObjectRecordHitNV
                        /**/ &__return_val
                        /**/ $AccelerationStructure
                        /**/ $InstanceIndex
                        /**/ $PrimitiveIndex
                        /**/ $GeometryIndex
                        /**/ $HitKind
                        /**/ $RayContributionToHitGroupIndex
                        /**/ $MultiplierForGeometryContributionToHitGroupIndex
                        /**/ $origin
                        /**/ $tmin
                        /**/ $direction
                        /**/ $tmax
                        /**/ $attr;
                };
            }
        }
    }

        /// See MakeHit but handles Motion 
        /// Currently only supported on VK
    [ForceInline]    
    [require(glsl_hlsl_spirv, ser_motion_raygen_closesthit_miss)]
    static HitObject MakeMotionHit<attr_t>( 
        RaytracingAccelerationStructure AccelerationStructure, 
        uint InstanceIndex, 
        uint GeometryIndex, 
        uint PrimitiveIndex, 
        uint HitKind, 
        uint RayContributionToHitGroupIndex, 
        uint MultiplierForGeometryContributionToHitGroupIndex, 
        RayDesc Ray,
        float CurrentTime,
        attr_t attributes)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "MakeMotionHit";
        case glsl:
        {
            // Save the attributes
            __hitObjectAttributes<attr_t>() = attributes;

            __glslMakeMotionHit(
                __return_val,
                AccelerationStructure,
                InstanceIndex,
                PrimitiveIndex,
                GeometryIndex,
                HitKind,
                RayContributionToHitGroupIndex,                         /// sbtRecordOffset?
                MultiplierForGeometryContributionToHitGroupIndex,       /// sbtRecordStride?
                Ray.Origin,
                Ray.TMin,
                Ray.Direction, 
                Ray.TMax,
                CurrentTime,
                __hitObjectAttributesLocation(__hitObjectAttributes<attr_t>()));
        }
        case spirv:
        {
            // Save the attributes
            __Addr<attr_t> attr = __allocHitObjectAttributes<attr_t>();

            *attr = attributes;

            let origin = Ray.Origin;
            let direction = Ray.Direction;
            let tmin = Ray.TMin;
            let tmax = Ray.TMax;
            spirv_asm 
            {
                OpExtension "SPV_NV_ray_tracing_motion_blur";
                OpCapability RayTracingMotionBlurNV;
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                OpHitObjectRecordHitMotionNV
                    /**/ &__return_val
                    /**/ $AccelerationStructure
                    /**/ $InstanceIndex
                    /**/ $PrimitiveIndex
                    /**/ $GeometryIndex
                    /**/ $HitKind
                    /**/ $RayContributionToHitGroupIndex
                    /**/ $MultiplierForGeometryContributionToHitGroupIndex
                    /**/ $origin
                    /**/ $tmin
                    /**/ $direction
                    /**/ $tmax
                    /**/ $CurrentTime
                    /**/ $attr;
            };
        }
        }
    }

        /// Creates a HitObject representing a hit based on values explicitly passed as arguments, without
        /// tracing a ray. The primitive specified by AccelerationStructure, InstanceIndex, GeometryIndex,
        /// and PrimitiveIndex must exist. The shader table index is explicitly provided as an argument
        /// instead of being computed from the indexing formula used in TraceRay. The provided index must
        /// reference a valid hit group record in the shader table. The Attributes parameter must either be an
        /// attribute struct, such as BuiltInTriangleIntersectionAttributes, or another HitObject to copy the
        /// attributes from.
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    static HitObject MakeHit<attr_t>(
        uint HitGroupRecordIndex,
        RaytracingAccelerationStructure AccelerationStructure,
        uint InstanceIndex,
        uint GeometryIndex,
        uint PrimitiveIndex,
        uint HitKind,
        RayDesc Ray,
        attr_t attributes)
    {
        __target_switch
        {
        case hlsl:
            HitObject hitObj;
            __hlslMakeHitWithRecordIndex(
                HitGroupRecordIndex, 
                AccelerationStructure, 
                InstanceIndex,
                GeometryIndex,
                PrimitiveIndex,
                HitKind,
                Ray,
                attributes,
                hitObj);
            return hitObj;
        case glsl:
        {
            // Save the attributes
            __hitObjectAttributes<attr_t>() = attributes;

            __glslMakeHitWithIndex(
                __return_val,
                AccelerationStructure,
                InstanceIndex,              ///? Same as instanceid ?
                PrimitiveIndex,
                GeometryIndex,
                HitKind,                    /// Assuming HitKinds are compatible
                HitGroupRecordIndex,        /// sbtRecordIndex
                Ray.Origin,
                Ray.TMin,
                Ray.Direction, 
                Ray.TMax,
                __hitObjectAttributesLocation(__hitObjectAttributes<attr_t>()));
        }
        case spirv:
        {
            // Save the attributes
            __Addr<attr_t> attr = __allocHitObjectAttributes<attr_t>();
            *attr = attributes;
            let origin = Ray.Origin;
            let direction = Ray.Direction;
            let tmin = Ray.TMin;
            let tmax = Ray.TMax;
            spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                OpHitObjectRecordHitWithIndexNV
                    /**/ &__return_val
                    /**/ $AccelerationStructure
                    /**/ $InstanceIndex
                    /**/ $PrimitiveIndex
                    /**/ $GeometryIndex
                    /**/ $HitKind
                    /**/ $HitGroupRecordIndex
                    /**/ $origin
                    /**/ $tmin
                    /**/ $direction
                    /**/ $tmax
                    /**/ $attr;
            };
        }
        }
    }
        /// See MakeHit but handles Motion 
        /// Currently only supported on VK
    [ForceInline]
    [require(glsl_spirv, ser_motion_raygen_closesthit_miss)]
    static HitObject MakeMotionHit<attr_t>( 
        uint HitGroupRecordIndex, 
        RaytracingAccelerationStructure AccelerationStructure, 
        uint InstanceIndex, 
        uint GeometryIndex, 
        uint PrimitiveIndex, 
        uint HitKind, 
        RayDesc Ray, 
        float CurrentTime,
        attr_t attributes)
    {
        __target_switch
        {
        case glsl:
        {
            // Save the attributes
            __hitObjectAttributes<attr_t>() = attributes;

            __glslMakeMotionHitWithIndex(
                __return_val,
                AccelerationStructure,
                InstanceIndex,              ///? Same as instanceid ?
                PrimitiveIndex, 
                GeometryIndex,
                HitKind,                    /// Assuming HitKinds are compatible
                HitGroupRecordIndex,        /// sbtRecordIndex
                Ray.Origin,
                Ray.TMin,
                Ray.Direction, 
                Ray.TMax,
                CurrentTime,
                __hitObjectAttributesLocation(__hitObjectAttributes<attr_t>()));
        }
        case spirv:
        {
            // Save the attributes
            __Addr<attr_t> attr = __allocHitObjectAttributes<attr_t>();
            *attr = attributes;
            let origin = Ray.Origin;
            let direction = Ray.Direction;
            let tmin = Ray.TMin;
            let tmax = Ray.TMax;
            spirv_asm 
            {
                OpExtension "SPV_NV_ray_tracing_motion_blur";
                OpCapability RayTracingMotionBlurNV;
                OpHitObjectRecordHitWithIndexMotionNV
                    /**/ &__return_val
                    /**/ $AccelerationStructure
                    /**/ $InstanceIndex
                    /**/ $PrimitiveIndex
                    /**/ $GeometryIndex
                    /**/ $HitKind
                    /**/ $HitGroupRecordIndex
                    /**/ $origin
                    /**/ $tmin
                    /**/ $direction
                    /**/ $tmax
                    /**/ $CurrentTime
                    /**/ $attr;
            };
        }
        }
    }

        /// Creates a HitObject representing a miss based on values explicitly passed as arguments, without
        /// tracing a ray. The provided shader table index must reference a valid miss record in the shader
        /// table.
    [__requiresNVAPI]
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    static HitObject MakeMiss( 
        uint MissShaderIndex, 
        RayDesc Ray)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "($2=NvMakeMiss($0,$1))";
        case glsl:
            __glslMakeMiss(__return_val, MissShaderIndex, Ray.Origin, Ray.TMin, Ray.Direction, Ray.TMax);
        case spirv:
            {
                let origin = Ray.Origin;
                let direction = Ray.Direction;
                let tmin = Ray.TMin;
                let tmax = Ray.TMax;
                spirv_asm 
                {
                    OpExtension "SPV_NV_shader_invocation_reorder";
                    OpCapability ShaderInvocationReorderNV;
                    OpHitObjectRecordMissNV
                        /**/ &__return_val
                        /**/ $MissShaderIndex
                        /**/ $origin
                        /**/ $tmin
                        /**/ $direction
                        /**/ $tmax;
                };
            }
        }
    }

        /// See MakeMiss but handles Motion 
        /// Currently only supported on VK
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_motion_raygen_closesthit_miss)]
    static HitObject MakeMotionMiss( 
        uint MissShaderIndex, 
        RayDesc Ray,
        float CurrentTime)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "($3=NvMakeMotionMiss($0,$1,$2))";
        case glsl:
            __glslMakeMotionMiss(__return_val, MissShaderIndex, Ray.Origin, Ray.TMin, Ray.Direction, Ray.TMax, CurrentTime);
        case spirv:
            {
                let origin = Ray.Origin;
                let direction = Ray.Direction;
                let tmin = Ray.TMin;
                let tmax = Ray.TMax;
                spirv_asm 
                {
                    OpExtension "SPV_NV_ray_tracing_motion_blur";
                    OpCapability RayTracingMotionBlurNV;
                    OpExtension "SPV_NV_shader_invocation_reorder";
                    OpCapability ShaderInvocationReorderNV;
                    OpHitObjectRecordMissMotionNV
                        /**/ &__return_val
                        /**/ $MissShaderIndex
                        /**/ $origin
                        /**/ $tmin
                        /**/ $direction
                        /**/ $tmax
                        /**/ $CurrentTime;
                };
            }
        }
    }

        /// Creates a HitObject representing “NOP” (no operation) which is neither a hit nor a miss. Invoking a
        /// NOP hit object using HitObject::Invoke has no effect. Reordering by hit objects using
        /// ReorderThread will group NOP hit objects together. This can be useful in some reordering
        /// scenarios where future control flow for some threads is known to process neither a hit nor a
        /// miss.
    [__requiresNVAPI]
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    static HitObject MakeNop()
    {
        __target_switch
        {
        case hlsl:
            __intrinsic_asm "($0 = NvMakeNop())";
        case glsl:
            __glslMakeNop(__return_val);
        case spirv:
            spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                OpHitObjectRecordEmptyNV
                    /**/ &__return_val;
            };
        }
    }

    [require(hlsl, ser)]
    __generic<payload_t>
    static void __InvokeHLSL(
        RaytracingAccelerationStructure AccelerationStructure,
        HitObject HitOrMiss,
        inout payload_t Payload)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "NvInvokeHitObject";
        }
    }

        /// Invokes closesthit or miss shading for the specified hit object. In case of a NOP HitObject, no
        /// shader is invoked.
    [__requiresNVAPI]
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    static void Invoke<payload_t>(
        RaytracingAccelerationStructure AccelerationStructure,
        HitObject HitOrMiss,
        inout payload_t Payload)
    {
        __target_switch
        {
        case hlsl:
            __InvokeHLSL(
                AccelerationStructure,
                HitOrMiss,
                __forceVarIntoRayPayloadStructTemporarily(Payload));
        case glsl:
            {
                [__vulkanRayPayload]
                static payload_t p;

                // Save the payload
                p = Payload;

                __glslInvoke(HitOrMiss, __rayPayloadLocation(p));

                // Write payload result
                Payload = p;
            }
        case spirv:
            {
                [__vulkanRayPayload]
                static payload_t p;

                // Save the payload
                p = Payload;

                spirv_asm 
                {     
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;   
                    OpHitObjectExecuteShaderNV
                        /**/ &HitOrMiss
                        /**/ &p;
                };

                // Write payload result
                Payload = p;
            }
        }
    }

        /// Returns true if the HitObject encodes a miss, otherwise returns false.
    [__requiresNVAPI]
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    bool IsMiss()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".IsMiss";
        case glsl: __intrinsic_asm "hitObjectIsMissNV($0)";
        case spirv:
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$bool = OpHitObjectIsMissNV &this;
            };
        }
    }

        /// Returns true if the HitObject encodes a hit, otherwise returns false.
    [__requiresNVAPI]
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    bool IsHit()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".IsHit";
        case glsl: __intrinsic_asm "hitObjectIsHitNV($0)";
        case spirv: 
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$bool = OpHitObjectIsHitNV &this;
            };
        }
    }

        /// Returns true if the HitObject encodes a nop, otherwise returns false.
    [__requiresNVAPI]
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    bool IsNop()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".IsNop";
        case glsl: __intrinsic_asm "hitObjectIsEmptyNV($0)";
        case spirv: 
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$bool = OpHitObjectIsEmptyNV &this;
            };
        }
    }

        /// Queries ray properties from HitObject. Valid if the hit object represents a hit or a miss.
    [__requiresNVAPI]
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    RayDesc GetRayDesc()
    {
        __target_switch
        {
        case hlsl:
            __intrinsic_asm ".GetRayDesc";
        case glsl:
            {
                RayDesc ray = { __glslGetRayWorldOrigin(), __glslGetTMin(), __glslGetRayWorldDirection(), __glslGetTMax() };
                return ray;
            }
        case spirv:
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                %origin:$$float3 = OpHitObjectGetWorldRayOriginNV &this;
                %tmin:$$float = OpHitObjectGetRayTMinNV &this;
                %direction:$$float3 = OpHitObjectGetWorldRayDirectionNV &this;
                %tmax:$$float = OpHitObjectGetRayTMaxNV &this;
                result:$$RayDesc = OpCompositeConstruct %origin %tmin %direction %tmax;
            };
        }
    }

        /// Queries shader table index from HitObject. Valid if the hit object represents a hit or a miss.
    [__requiresNVAPI]
    __glsl_extension(GL_EXT_ray_tracing)
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    uint GetShaderTableIndex()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".GetShaderTableIndex";
        case glsl: __intrinsic_asm "hitObjectGetShaderBindingTableRecordIndexNV($0)";
        case spirv: 
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$uint = OpHitObjectGetShaderBindingTableRecordIndexNV &this;
            };
        }
    }

        /// Returns the instance index of a hit. Valid if the hit object represents a hit.
    [__requiresNVAPI]
    __glsl_extension(GL_EXT_ray_tracing)
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    uint GetInstanceIndex()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".GetInstanceIndex";
        case glsl: __intrinsic_asm "hitObjectGetInstanceIdNV($0)";
        case spirv: 
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$uint = OpHitObjectGetInstanceIdNV &this;
            };
        }
    }

        /// Returns the instance ID of a hit. Valid if the hit object represents a hit.
    [__requiresNVAPI]
    __glsl_extension(GL_EXT_ray_tracing)
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    uint GetInstanceID()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".GetInstanceID";
        case glsl: __intrinsic_asm "hitObjectGetInstanceCustomIndexNV($0)";
        case spirv:
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$uint = OpHitObjectGetInstanceCustomIndexNV &this;
            };
        }
    }

        /// Returns the geometry index of a hit. Valid if the hit object represents a hit.
    [__requiresNVAPI]
    __glsl_extension(GL_EXT_ray_tracing)
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    uint GetGeometryIndex()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".GetGeometryIndex";
        case glsl: __intrinsic_asm "hitObjectGetGeometryIndexNV($0)";
        case spirv: 
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$uint = OpHitObjectGetGeometryIndexNV &this;
            };
        }
    }

        /// Returns the primitive index of a hit. Valid if the hit object represents a hit.
    [__requiresNVAPI]
    __glsl_extension(GL_EXT_ray_tracing)
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    uint GetPrimitiveIndex()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".GetPrimitiveIndex";
        case glsl: __intrinsic_asm "hitObjectGetPrimitiveIndexNV($0)";
        case spirv: 
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$uint = OpHitObjectGetPrimitiveIndexNV &this;
            };
        }
    }

        /// Returns the hit kind. Valid if the hit object represents a hit.
    [__requiresNVAPI]
    __glsl_extension(GL_EXT_ray_tracing)
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    uint GetHitKind()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".GetHitKind";
        case glsl: __intrinsic_asm "hitObjectGetHitKindNV($0)";
        case spirv: 
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$uint = OpHitObjectGetHitKindNV &this;
            };
        }
    }

    [__requiresNVAPI]
    __glsl_extension(GL_EXT_ray_tracing)
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    float4x3 GetWorldToObject()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".GetWorldToObject";
        case glsl: __intrinsic_asm "hitObjectGetWorldToObjectNV($0)";
        case spirv: 
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$float4x3 = OpHitObjectGetWorldToObjectNV &this;
            };
        }
    }

    [__requiresNVAPI]
    __glsl_extension(GL_EXT_ray_tracing)
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    float4x3 GetObjectToWorld()
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".GetObjectToWorld";
        case glsl: __intrinsic_asm "hitObjectGetObjectToWorldNV($0)";
        case spirv: 
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$float4x3 = OpHitObjectGetObjectToWorldNV &this;
            };
        }
    }

    [ForceInline]
    [require(glsl_spirv, ser_raygen_closesthit_miss)]
    float GetCurrentTime() {
        __target_switch
        {
        case glsl:
            __intrinsic_asm "hitObjectGetCurrentTimeNV($0)";
        case spirv:
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$float = OpHitObjectGetCurrentTimeNV &this
            };
        }
    }

    [ForceInline]
    [require(glsl_spirv, ser_raygen_closesthit_miss)]
    float3 GetObjectRayOrigin() {
        __target_switch
        {
        case glsl:
            __intrinsic_asm "hitObjectGetObjectRayOriginNV($0)";
        case spirv:
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$float3 = OpHitObjectGetObjectRayOriginNV &this
            };
        }
    }

    [ForceInline]
    [require(glsl_spirv, ser_raygen_closesthit_miss)]
    float3 GetObjectRayDirection() {
        __target_switch
        {
        case glsl:
            __intrinsic_asm "hitObjectGetObjectRayDirectionNV($0)";
        case spirv:
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$float3 = OpHitObjectGetObjectRayDirectionNV &this
            };
        }
    }

    [ForceInline]
    [require(glsl_spirv, ser_raygen_closesthit_miss)]
    uint2 GetShaderRecordBufferHandle() {
        __target_switch
        {
        case glsl:
            __intrinsic_asm "hitObjectGetShaderRecordBufferHandleNV($0)";
        case spirv:
            return spirv_asm 
            {
                OpExtension "SPV_NV_shader_invocation_reorder";
                OpCapability ShaderInvocationReorderNV;
                result:$$uint2 = OpHitObjectGetShaderRecordBufferHandleNV &this
            };
        }
    }

    /// Returns the attributes of a hit. Valid if the hit object represents a hit or a miss.
    [ForceInline]
    [require(glsl_hlsl_spirv, ser_raygen_closesthit_miss)]
    attr_t GetAttributes<attr_t>()
    {
        __target_switch
        {
        case hlsl:
            {
                attr_t v;
                __hlslGetAttributesFromHitObject(v);
                return v;
            }
        case glsl:
            {
                // Work out the location
                int attributeLocation = __hitObjectAttributesLocation(__hitObjectAttributes<attr_t>());

                // Load the attributes from the location
                __glslGetAttributes(attributeLocation);

                // Return the attributes
                return __hitObjectAttributes<attr_t>();
            }
        case spirv:
            {
                __Addr<attr_t> attr = __allocHitObjectAttributes<attr_t>();
                spirv_asm 
                {
                    OpExtension "SPV_NV_shader_invocation_reorder";
                    OpCapability ShaderInvocationReorderNV;
                    OpHitObjectGetAttributesNV &this $attr;
                };
                return *attr;
            }
        }
    }
        /// Loads a root constant from the local root table referenced by the hit object. Valid if the hit object
        /// represents a hit or a miss. RootConstantOffsetInBytes must be a multiple of 4.
    [__requiresNVAPI]
    [require(hlsl, ser)]
    uint LoadLocalRootTableConstant(uint RootConstantOffsetInBytes)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".LoadLocalRootTableConstant";
        }
    }

    /// 
    /// !!!! Internal NVAPI HLSL impl. Not part of interface! !!!!!!!!!!!!
    /// 

    [__requiresNVAPI]
    [require(hlsl, ser_raygen_closesthit_miss)]
    void __hlslGetAttributesFromHitObject<T>(out T t)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "NvGetAttributesFromHitObject($0, $1)";
        }
    }

    [__requiresNVAPI]
    [require(hlsl, ser_raygen_closesthit_miss)]
    static void __hlslMakeHitWithRecordIndex<attr_t>(
        uint HitGroupRecordIndex, 
        RaytracingAccelerationStructure AccelerationStructure, 
        uint InstanceIndex, 
        uint GeometryIndex, 
        uint PrimitiveIndex, 
        uint HitKind, 
        RayDesc Ray, 
        attr_t attributes, 
        out HitObject hitObj)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "NvMakeHitWithRecordIndex";
        }
    }

    [__requiresNVAPI]
    [require(hlsl, ser_raygen_closesthit_miss)]
    static void __hlslMakeHit<attr_t>(RaytracingAccelerationStructure AccelerationStructure, 
        uint InstanceIndex, 
        uint GeometryIndex, 
        uint PrimitiveIndex, 
        uint HitKind, 
        uint RayContributionToHitGroupIndex, 
        uint MultiplierForGeometryContributionToHitGroupIndex, 
        RayDesc Ray, 
        attr_t attributes, 
        out HitObject hitObj)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "NvMakeHit";
        }
    }

    [__requiresNVAPI]
    [require(hlsl, ser_raygen_closesthit_miss)]
    static void __hlslTraceRay<payload_t>( 
        RaytracingAccelerationStructure AccelerationStructure, 
        uint RayFlags, 
        uint InstanceInclusionMask, 
        uint RayContributionToHitGroupIndex, 
        uint MultiplierForGeometryContributionToHitGroupIndex, 
        uint MissShaderIndex, 
        RayDesc Ray, 
        inout payload_t Payload,
        out HitObject hitObj)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "NvTraceRayHitObject";
        }
    }

    /// 
    /// !!!! Internal GLSL GL_NV_shader_invocation_reorder impl. Not part of interface! !!!!!!!!!!!!
    /// 

    __glsl_extension(GL_NV_shader_invocation_reorder)
    __glsl_extension(GL_EXT_ray_tracing)
    [require(glsl, ser_raygen_closesthit_miss)]
    static void __glslMakeMiss(
        out HitObject hitObj,
        uint MissShaderIndex,
        float3 Origin,
        float TMin,
        float3 Direction,
        float TMax)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectRecordMissNV";
        }
    }

    // "void hitObjectRecordMissNV(hitObjectNV, uint, vec3, float, vec3, float);"
    __glsl_extension(GL_NV_shader_invocation_reorder)
    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_ray_tracing_motion_blur)    
    [require(glsl, ser_motion_raygen_closesthit_miss)]
    static void __glslMakeMotionMiss(
        out HitObject hitObj,
        uint MissShaderIndex,
        float3 Origin,
        float TMin,
        float3 Direction,
        float TMax, 
        float CurrentTime)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectRecordMissMotionNV";
        }
    }

    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_shader_invocation_reorder)
    [require(glsl, ser_raygen_closesthit_miss)]
    static void __glslMakeNop(out HitObject hitObj)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectRecordEmptyNV";
        }
    }

    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_shader_invocation_reorder)
    [require(glsl, ser_raygen_closesthit_miss)]
    float3 __glslGetRayDirection()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectGetObjectRayDirectionNV($0)";
        }
    }
    
    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_shader_invocation_reorder)
    [require(glsl, ser_raygen_closesthit_miss)]
    float3 __glslGetRayWorldDirection()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectGetWorldRayDirectionNV($0)";
        }
    }

    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_shader_invocation_reorder)
    [require(glsl, ser_raygen_closesthit_miss)]
    float3 __glslGetRayWorldOrigin()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectGetWorldRayOriginNV($0)";
        }
    }

    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_shader_invocation_reorder)
    [require(glsl, ser_raygen_closesthit_miss)]
    float __glslGetTMax()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectGetRayTMaxNV($0)";
        }
    }

    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_shader_invocation_reorder)
    [require(glsl, ser_raygen_closesthit_miss)]
    float __glslGetTMin()
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectGetRayTMinNV($0)";
        }
    }

    // "void hitObjectRecordHitWithIndexNV(hitObjectNV, accelerationStructureEXT,int,int,int,uint,uint,vec3,float,vec3,float,int);"
    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_shader_invocation_reorder)
    [require(glsl, ser_raygen_closesthit_miss)]
    static void __glslMakeHitWithIndex(
        out HitObject hitObj,
        RaytracingAccelerationStructure accelerationStructure,
        int instanceid,
        int primitiveid,
        int geometryindex,
        uint hitKind,
        uint sbtRecordIndex,
        float3 origin,
        float Tmin,
        float3 direction,
        float Tmax,
        int attributeLocation)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectRecordHitWithIndexNV";
        }
    }

    //  "void hitObjectRecordHitWithIndexMotionNV(hitObjectNV, accelerationStructureEXT,int,int,int,uint,uint,vec3,float,vec3,float,float,int);"
    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_shader_invocation_reorder)
    __glsl_extension(GL_NV_ray_tracing_motion_blur)
    [require(glsl, ser_motion_raygen_closesthit_miss)]
    static void __glslMakeMotionHitWithIndex(
        out HitObject hitObj,
        RaytracingAccelerationStructure accelerationStructure,
        int instanceid,
        int primitiveid,
        int geometryindex,
        uint hitKind,
        uint sbtRecordIndex,
        float3 origin,
        float Tmin,
        float3 direction,
        float Tmax,
        float CurrentTime,
        int attributeLocation)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectRecordHitWithIndexMotionNV";
        }
    }

    // "void hitObjectRecordHitNV(hitObjectNV,accelerationStructureEXT,int,int,int,uint,uint,uint,vec3,float,vec3,float,int);"
    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_shader_invocation_reorder)
    [require(glsl, ser_raygen_closesthit_miss)]
    static void __glslMakeHit(
        out HitObject hitObj,
        RaytracingAccelerationStructure accelerationStructure,
        int instanceid,
        int primitiveid,
        int geometryindex,
        uint hitKind,
        uint sbtRecordOffset,
        uint sbtRecordStride,
        float3 origin,
        float Tmin,
        float3 direction,
        float Tmax,
        int attributeLocation)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectRecordHitNV";
        }
    }

        // "void hitObjectRecordHitMotionNV(hitObjectNV,accelerationStructureEXT,int,int,int,uint,uint,uint,vec3,float,vec3,float,float,int);"
    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_shader_invocation_reorder)
    __glsl_extension(GL_NV_ray_tracing_motion_blur)
    [require(glsl, ser_motion_raygen_closesthit_miss)]
    static void __glslMakeMotionHit(
        out HitObject hitObj,
        RaytracingAccelerationStructure accelerationStructure,
        int instanceid,
        int primitiveid,
        int geometryindex,
        uint hitKind,
        uint sbtRecordOffset,
        uint sbtRecordStride,
        float3 origin,
        float Tmin,
        float3 direction,
        float Tmax,
        float CurrentTime,
        int attributeLocation)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectRecordHitMotionNV";
        }
    }

    
    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_shader_invocation_reorder)
    [require(glsl, ser_raygen_closesthit_miss)]
    void __glslGetAttributes(int attributeLocation)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectGetAttributesNV($0, $1)";
        }
    }

    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_shader_invocation_reorder)
    [require(glsl, ser_raygen_closesthit_miss)]
    static void __glslTraceRay(
        out HitObject hitObject,
        RaytracingAccelerationStructure accelerationStructure,
        uint rayFlags,
        uint cullMask,
        uint sbtRecordOffset,
        uint sbtRecordStride,
        uint missIndex,
        float3 origin,
        float Tmin,
        float3 direction,
        float Tmax,
        int payload)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectTraceRayNV";
        }
    }

    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_shader_invocation_reorder)
    __glsl_extension(GL_NV_ray_tracing_motion_blur)
    [require(glsl, ser_motion_raygen_closesthit_miss)]
    static void __glslTraceMotionRay(
        out HitObject hitObject,
        RaytracingAccelerationStructure accelerationStructure,
        uint rayFlags,
        uint cullMask,
        uint sbtRecordOffset,
        uint sbtRecordStride,
        uint missIndex,
        float3 origin,
        float Tmin,
        float3 direction,
        float Tmax,
        float currentTime,
        int payload)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectTraceRayMotionNV";
        }
    }

    __glsl_extension(GL_EXT_ray_tracing)
    __glsl_extension(GL_NV_shader_invocation_reorder)
    [require(glsl, ser_raygen_closesthit_miss)]
    static void __glslInvoke(
        HitObject hitObj,
        int payload)
    {
        __target_switch
        {
        case glsl: __intrinsic_asm "hitObjectExecuteShaderNV";
        }
    }
};

    /// Reorders threads based on a coherence hint value. NumCoherenceHintBits indicates how many of
    /// the least significant bits of CoherenceHint should be considered during reordering (max: 16).
    /// Applications should set this to the lowest value required to represent all possible values in
    /// CoherenceHint. For best performance, all threads should provide the same value for
    /// NumCoherenceHintBits.
    /// Where possible, reordering will also attempt to retain locality in the thread’s launch indices
    /// (DispatchRaysIndex in DXR).
[__requiresNVAPI]
__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
[ForceInline]
[require(glsl_hlsl_spirv, ser_raygen)]
void ReorderThread( uint CoherenceHint, uint NumCoherenceHintBitsFromLSB )
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "NvReorderThread";
    case glsl: __intrinsic_asm "reorderThreadNV";
    case spirv:
        spirv_asm 
        {
            OpExtension "SPV_NV_shader_invocation_reorder";
            OpCapability ShaderInvocationReorderNV;
            OpReorderThreadWithHintNV $CoherenceHint $NumCoherenceHintBitsFromLSB;
        };
    }
}

    /// Reorders threads based on a hit object, optionally extended by a coherence hint value. Coherence
    /// hints behave as described in the generic variant of ReorderThread. The maximum number of
    /// coherence hint bits in this variant of ReorderThread is 8. If no coherence hint is desired, set
    /// NumCoherenceHitBits to zero.
    /// Reordering will consider information in the HitObject and coherence hint with the following
    /// priority:
    ///
    /// 1. Shader ID stored in the HitObject
    /// 2. Coherence hint, with the most significant hint bit having highest priority
    /// 3. Spatial information stored in the HitObject
    ///
    /// That is, ReorderThread will first attempt to group threads whose HitObject references the
    /// same shader ID. (Miss shaders and NOP HitObjects are grouped separately). Within each of these
    /// groups, it will attempt to order threads by the value of their coherence hints. And within ranges
    /// of equal coherence hints, it will attempt to maximize locality in 3D space of the ray hit (if any).

[__requiresNVAPI]
__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
[ForceInline]
[require(glsl_hlsl_spirv, ser_raygen)]
void ReorderThread( HitObject HitOrMiss, uint CoherenceHint, uint NumCoherenceHintBitsFromLSB )
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "NvReorderThread";
    case glsl: __intrinsic_asm "reorderThreadNV";
    case spirv:
        spirv_asm 
        {
            OpExtension "SPV_NV_shader_invocation_reorder";
            OpCapability ShaderInvocationReorderNV;
            OpReorderThreadWithHitObjectNV &HitOrMiss $CoherenceHint $NumCoherenceHintBitsFromLSB;
        };
    }
}

    // Is equivalent to
    // ```
    // void ReorderThread( HitObject HitOrMiss, uint CoherenceHint, uint NumCoherenceHintBitsFromLSB );
    // ```
    // With CoherenceHint and NumCoherenceHintBitsFromLSB as 0, meaning they are ignored.

[__requiresNVAPI]
__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
[ForceInline]
[require(glsl_hlsl_spirv, ser_raygen)]
void ReorderThread( HitObject HitOrMiss )
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "NvReorderThread";
    case glsl: __intrinsic_asm "reorderThreadNV";
    case spirv:
        spirv_asm 
        {
            OpExtension "SPV_NV_shader_invocation_reorder";
            OpCapability ShaderInvocationReorderNV;
            OpReorderThreadWithHitObjectNV &HitOrMiss;
        };
    }
}

///
/// DebugBreak support 
///
/// There doesn't appear to be an equivalent for debugBreak for HLSL

[require(glsl)]
__specialized_for_target(glsl)
[[vk::spirv_instruction(1, "NonSemantic.DebugBreak")]]
void __glslDebugBreak();

[ForceInline]
[require(cpp_cuda_glsl_hlsl, breakpoint)]
void debugBreak() 
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "/* debugBreak() not currently supported for HLSL */";
    case cuda: __intrinsic_asm "__brkpt()";
    case cpp: __intrinsic_asm "SLANG_BREAKPOINT(0)";
    case glsl:
        __glslDebugBreak();
        return;
    }
}


// 
// Realtime Clock support
//

// https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_realtime_clock.txt

[__requiresNVAPI]
__glsl_extension(GL_EXT_shader_realtime_clock)
[NonUniformReturn]
[require(cpp_cuda_glsl_hlsl_spirv, shaderclock)]
uint getRealtimeClockLow()
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "NvGetSpecial( NV_SPECIALOP_GLOBAL_TIMER_LO)";
    case glsl:
        return getRealtimeClock().x;
    case cuda:
        __intrinsic_asm "clock";
    case spirv:
        return getRealtimeClock().x;
    case cpp:
        __intrinsic_asm "(uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count()";
    }
}

[NonUniformReturn]
[require(cpp_cuda, shaderclock)]
int64_t __cudaCppGetRealtimeClock()
{
    __target_switch
    {
    case cpp: __intrinsic_asm "std::chrono::high_resolution_clock::now().time_since_epoch().count()";
    case cuda: __intrinsic_asm "clock64";
    }
}

[__requiresNVAPI]
__glsl_extension(GL_EXT_shader_realtime_clock)
[NonUniformReturn]
[require(cpp_cuda_glsl_hlsl_spirv, shaderclock)]
uint2 getRealtimeClock()
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "uint2(NvGetSpecial(NV_SPECIALOP_GLOBAL_TIMER_LO), NvGetSpecial( NV_SPECIALOP_GLOBAL_TIMER_HI))";
    case glsl:
        __intrinsic_asm "clockRealtime2x32EXT()";
    case cuda:
    case cpp:
        int64_t ticks = __cudaCppGetRealtimeClock();
        return uint2(uint(ticks), uint(uint64_t(ticks) >> 32));
    case spirv:
        return spirv_asm
        {
            OpCapability ShaderClockKHR;
            OpExtension "SPV_KHR_shader_clock";
            result : $$uint2 = OpReadClockKHR Device
        };
    }
}

// 
// CUDA specific 
// 

[__readNone]
[NonUniformReturn]
[require(cuda)]
uint3 cudaThreadIdx()
{
    __target_switch
    {
    case cuda: __intrinsic_asm "(threadIdx)";
    }
}

[__readNone]
[NonUniformReturn]
[require(cuda)]
uint3 cudaBlockIdx()
{
    __target_switch
    {
    case cuda: __intrinsic_asm "(blockIdx)";
    }
}

[__readNone]
[NonUniformReturn]
[require(cuda)]
uint3 cudaBlockDim()
{
    __target_switch
    {
    case cuda: __intrinsic_asm "(blockDim)";
    }
}

//
// Workgroup cooperation
//

//
// `saturated_cooperation(c, f, s, u)` will call `f(s, u)` if not all lanes in the
// workgroup are currently executing. however if all lanes are saturated, then
// for each unique `s` across all the active lanes `c(s, u)` is called. The
// return value is the one corresponding to the input `s` from this lane.
//
// Adjacent calls to saturated_cooperation are subject to fusion, i.e.
//      saturated_cooperation(c1, f1, s, u1);
//      saturated_cooperation(c2, f2, s, u2);
// will be transformed to:
//      saturated_cooperation(c1c2, f1f2, s, u1u2);
// where
//      c1c2 is a function which calls c1(s, u1) and then c2(s, u2);
//      f1f2 is a function which calls f1(s, u1) and then f2(s, u2);
//
// When the input differs, calls are fused
//      saturated_cooperation(c1, f1, s1, u1);
//      saturated_cooperation(c2, f2, s2, u2);
// will be transformed to:
//      saturated_cooperation(c1c2, f1f2, s1s2, u1u2);
// where
//      s1s2 is a tuple of s1 and s2
//      c1c2 is a function which calls c1(s1, u1) and then c2(s2, u2);
//      f1f2 is a function which calls f1(s1, u1) and then f2(s2, u2);
// Note that in this case, we will make a call to c1c2 for every unique pair
// s1s2 across all lanes
//
// (This fusion takes place in the fuse-satcoop pass, and as such any changes to
// the signature or behavior of this function should be adjusted for there).
//
//@hidden:
[KnownBuiltin("saturated_cooperation")]
func saturated_cooperation<A : __BuiltinType, B, C>(
    cooperate : functype (A, B) -> C,
    fallback : functype (A, B) -> C,
    A input,
    B otherArg)
    -> C
{
    return saturated_cooperation_using(cooperate, fallback, __WaveMatchBuitin<A>, __WaveReadLaneAtBuiltin<A>, input, otherArg);
}

// These two functions are a temporary (circa May 2023) workaround to the fact
// that we can't deduce which overload to pass to saturated_cooperation_using
// in the call above
[__unsafeForceInlineEarly]
func __WaveMatchBuitin<T : __BuiltinType>(T t) -> uint4
{
    return WaveMatch(t);
}
[__unsafeForceInlineEarly]
func __WaveReadLaneAtBuiltin<T : __BuiltinType>(T t, int i) -> T
{
    return WaveReadLaneAt(t, i);
}

//
// saturated_cooperation, but you're able to specify manually the functions:
//
// waveMatch: a function to return a mask of lanes with the same input as this one
// broadcast: a function which returns the value passed into it on the specified lane
//
[KnownBuiltin("saturated_cooperation_using")]
func saturated_cooperation_using<A, B, C>(
    cooperate : functype (A, B) -> C,
    fallback : functype (A, B) -> C,
    waveMatch : functype (A) -> uint4,
    broadcast : functype (A, int) -> A,
    A input,
    B otherArg)
    -> C
{
    const bool isWaveSaturated = WaveActiveCountBits(true) == WaveGetLaneCount();
    if(isWaveSaturated)
    {
        let lanesWithSameInput = waveMatch(input).x;
        // Keep least significant lane in our set
        let ourRepresentative = lanesWithSameInput & -lanesWithSameInput;
        // The representative lanes for all lanes
        var allRepresentatives = WaveActiveBitOr(ourRepresentative);

        C ret;

        // Iterate over set bits in mask from low to high.
        // In each iteration the lowest bit is cleared.
        while(bool(allRepresentatives))
        {
            // Broadcast input across warp.
            let laneIdx = firstbitlow(allRepresentatives);
            let uniformInput = broadcast(input, int(laneIdx));

            // All lanes perform some cooperative computation with dynamic
            // uniform input
            C c = cooperate(uniformInput, otherArg);

            // Update our return value until it
            if(bool(allRepresentatives & ourRepresentative))
                ret = c;

            // Clear the lowest bit
            allRepresentatives &= allRepresentatives - 1;
        }

        return ret;
    }
    else
    {
        return fallback(input, otherArg);
    }
}


${
// The NVAPI operations are defined to take the space/register
// indices of their texture and sampler parameters, rather than
// taking the texture/sampler objects directly.
//
// In order to support this approach, we need intrinsics that
// can magically fetch the binding information for a resource.
//
// TODO: These operations are kind of *screaming* for us to
// have a built-in `interface` that all of the opaque resource
// types conform to, so that we can define builtins that work
// for any resource type.
}

__intrinsic_op($(kIROp_GetRegisterSpace)) uint __getRegisterSpace<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let isMS:int, let sampleCount:int, let access:int, let isShadow:int, let isCombined:int, let format:int>(_Texture<T,Shape,isArray,isMS,sampleCount,access,isShadow,isCombined,format> texture);
__intrinsic_op($(kIROp_GetRegisterSpace)) uint __getRegisterSpace(SamplerState sampler);

__intrinsic_op($(kIROp_GetRegisterIndex)) uint __getRegisterIndex<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let isMS:int, let sampleCount:int, let access:int, let isShadow:int, let isCombined:int, let format:int>(_Texture<T,Shape,isArray,isMS,sampleCount,access,isShadow,isCombined,format> texture);
__intrinsic_op($(kIROp_GetRegisterIndex)) uint __getRegisterIndex(SamplerState sampler);

//@public:

${{{{
//
// Texture Footprint Queries
//
// This section introduces the types and methods related
// to the `GL_NV_shader_texture_footprint` GLSL extension,
// and the matching NVAPI operations.
//
// Footprint queries are allowed on both 2D and 3D textures,
// and are structurally similar for the two, so we will
// use a meta-loop to deduplicate the code for the two
// cases.
//

// A footprint query yields a data structure
// that describes blocks of texels that
// conservatively cover the data that might
// be fetched in the query.
//
// A given sampling operation might access two
// mip levels of a texture when, e.g., trilinear
// filtering is on. A footprint query may ask for
// a footprint in either the coarse or fine level
// of the pair.
//
// We first define a `struct` type that closely maps
// to how a footprint is defined for each of the
// implementations we support, and then wrap that
// in a derived `struct` that includes the extra
// data that is returned by the GLSL API via the
// function reuslt.
//
}}}}

[__NoSideEffect]
[__requiresNVAPI]
[require(glsl_hlsl_spirv, texturefootprint)]
vector<uint, ND> __textureFootprintGetAnchor<let ND:int>(__TextureFootprintData<ND> data, int nd)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "NvFootprintExtractAnchorTileLoc$!1D($0)";
    case glsl:
        __intrinsic_asm "$0.anchor";
    case spirv:
        return spirv_asm {
            result:$$vector<uint,ND> = OpCompositeExtract $data 1;
        };
    }
}

[__NoSideEffect]
[__requiresNVAPI]
[require(glsl_hlsl_spirv, texturefootprint)]
vector<uint, ND> __textureFootprintGetOffset<let ND:int>(__TextureFootprintData<ND> data, int nd)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "NvFootprintExtractOffset$!1D($0)";
    case glsl:
        __intrinsic_asm "$0.offset";
    case spirv:
        return spirv_asm {
            result:$$vector<uint,ND> = OpCompositeExtract $data 2;
        };
    }
}

//@public:
__intrinsic_type($(kIROp_TextureFootprintType))
[require(glsl_hlsl_spirv, texturefootprint)]
struct __TextureFootprintData<let ND:int>
{
    typealias Anchor        = vector<uint, ND>;
    typealias Offset        = vector<uint, ND>;
    typealias Mask          = uint2;
    typealias LOD           = uint;
    typealias Granularity   = uint;

    property anchor : Anchor
    {
        [__NoSideEffect]
        [__requiresNVAPI]
        [ForceInline]
        get { return __textureFootprintGetAnchor(this, ND); }
    }

    property offset : Offset
    {
        [__NoSideEffect]
        [__requiresNVAPI]
        [ForceInline]
        get { return __textureFootprintGetOffset(this, ND); }
    }

    property mask : Mask
    {
        [__NoSideEffect]
        [__requiresNVAPI]
        get
        {
            __target_switch
            {
            case hlsl:
                __intrinsic_asm "NvFootprintExtractBitmask";
            case glsl:
                __intrinsic_asm "$0.mask";
            case spirv:
                return spirv_asm {
                    result:$$Mask = OpCompositeExtract $this 3;
                };
            }
        }
    }

    property lod : LOD
    {
        [__NoSideEffect]
        [__requiresNVAPI]
        get
        {
            __target_switch
            {
            case hlsl:
                __intrinsic_asm "NvFootprintExtractLOD";
            case glsl:
                __intrinsic_asm "$0.lod";
            case spirv:
                return spirv_asm {
                    result:$$LOD = OpCompositeExtract $this 4;
                };
            }
        }
    }

    property granularity : Granularity
    {
        [__NoSideEffect]
        [__requiresNVAPI]
        get
        {
            __target_switch
            {
            case hlsl:
                __intrinsic_asm "NvFootprintExtractReturnGran";
            case glsl:
                __intrinsic_asm "$0.granularity";
            case spirv:
                return spirv_asm {
                    result:$$Granularity = OpCompositeExtract $this 5;
                };
            }
        }
    }
}

///@category stage_io
struct TextureFootprint<let ND:int> : __TextureFootprintData<ND>
{
    bool _isSingleLevel;

    property isSingleLevel : bool
    {
        [__NoSideEffect]
        get
        {
            return _isSingleLevel;
        }
    }
}

///@category stage_io
typealias TextureFootprint2D = TextureFootprint<2>;

///@category stage_io
typealias TextureFootprint3D = TextureFootprint<3>;

${
// We define the new operations via an `extension`
// on the relevant texture type(s), rather than
// further clutter the original type declarations.
}

__generic<T:ITexelElement, Shape: __ITextureShape, let sampleCount:int, let isShadow:int, let format:int>
extension _Texture<T,Shape,0,0,sampleCount,0,isShadow,0,format>
{
${
// We introduce a few convenience type aliases here,
// which both keep our declarations simpler and easier
// to understand, but which might *also* be useful to
// users of the standard module, so that they can write things
// like `Texture2D.Footprint`, and also have auto-complete
// help them find such members.
//
// TODO: The `Coords` type really ought to be something
// defined on the base texture types, rather than via
// this `extension`.
}
    typealias Coords = vector<float, Shape.dimensions>;
    typealias Footprint = TextureFootprint<Shape.dimensions>;
    typealias __FootprintData = __TextureFootprintData<Shape.dimensions>;
    typealias FootprintGranularity = Footprint.Granularity;

${
// For the GLSL extension, the choice between the
// coarse and fine level is modeled as a `bool`
// parameter to the query operation(s). We define
// the GLSL functions here as intrinsics, so that
// we can refer to them later in the definitions
// of our standard module operaitons; not just in glsl module.
//
// Note: despite the GLSL extension defining the `granularity`
// member of the query result as having type `uint`, the
// function signatures all take `int` parameters for the
// granularity instead.
//
}

    [__NoSideEffect]
    __glsl_version(450)
    __glsl_extension(GL_NV_shader_texture_footprint)
    [require(glsl_spirv, texturefootprint)]
    bool __queryFootprintGLSL(
            SamplerState    sampler,
            Coords          coords,
            int             granularity,
            bool            useCoarseLevel,
            out __FootprintData footprint)
    {
        __target_switch
        {
        case glsl:
            __intrinsic_asm "textureFootprintNV($p, $*2)";
        case spirv:
            return spirv_asm {
                OpCapability ImageFootprintNV;
                OpExtension "SPV_NV_shader_image_footprint";
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $sampler;
                %resultVal:$$__FootprintData = OpImageSampleFootprintNV %sampledImage $coords $granularity $useCoarseLevel;
                OpStore &footprint %resultVal;
                result:$$bool = OpCompositeExtract %resultVal 0;
            };
        }
    }

    [__NoSideEffect]
    __glsl_version(450)
    __glsl_extension(GL_NV_shader_texture_footprint)
    [require(glsl_spirv, texturefootprint)]
    bool __queryFootprintGLSL(
            SamplerState    sampler,
            Coords          coords,
            int             granularity,
            bool            useCoarseLevel,
            out __FootprintData footprint,
            float           bias)
    {
        __target_switch
        {
        case glsl:
            __intrinsic_asm "textureFootprintNV($p, $*2)";
        case spirv:
            return spirv_asm {
                OpCapability ImageFootprintNV;
                OpExtension "SPV_NV_shader_image_footprint";
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $sampler;
                %resultVal:$$__FootprintData = OpImageSampleFootprintNV %sampledImage $coords $granularity $useCoarseLevel Bias $bias;
                OpStore &footprint %resultVal;
                result:$$bool = OpCompositeExtract %resultVal 0;
            };
        }
    }

    [__NoSideEffect]
    __glsl_version(450)
    __glsl_extension(GL_NV_shader_texture_footprint)
    __glsl_extension(GL_ARB_sparse_texture_clamp)
    [require(glsl_spirv, texturefootprintclamp)]
    bool __queryFootprintClampGLSL(
            SamplerState    sampler,
            Coords          coords,
            float           lodClamp,
            int             granularity,
            bool            useCoarseLevel,
            out __FootprintData footprint)
    {
        __target_switch
        {
        case glsl:
            __intrinsic_asm "textureFootprintClampNV($p, $*2)";
        case spirv:
            return spirv_asm {
                OpCapability ImageFootprintNV;
                OpCapability MinLod;
                OpExtension "SPV_NV_shader_image_footprint";
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $sampler;
                %resultVal:$$__FootprintData = OpImageSampleFootprintNV %sampledImage $coords $granularity $useCoarseLevel MinLod $lodClamp;
                OpStore &footprint %resultVal;
                result:$$bool = OpCompositeExtract %resultVal 0;
            };
        }
    }

    [__NoSideEffect]
    __glsl_version(450)
    __glsl_extension(GL_NV_shader_texture_footprint)
    __glsl_extension(GL_ARB_sparse_texture_clamp)
    [require(glsl_spirv, texturefootprintclamp)]
    bool __queryFootprintClampGLSL(
            SamplerState    sampler,
            Coords          coords,
            float           lodClamp,
            int             granularity,
            bool            useCoarseLevel,
            out __FootprintData footprint,
            float           bias)
    {
        __target_switch
        {
        case glsl:
            __intrinsic_asm "textureFootprintClampNV($p, $*2)";
        case spirv:
            return spirv_asm {
                OpCapability ImageFootprintNV;
                OpCapability MinLod;
                OpExtension "SPV_NV_shader_image_footprint";
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $sampler;
                %resultVal:$$__FootprintData = OpImageSampleFootprintNV %sampledImage $coords $granularity $useCoarseLevel Bias|MinLod $bias $lodClamp;
                OpStore &footprint %resultVal;
                result:$$bool = OpCompositeExtract %resultVal 0;
            };
        }
    }

    [__NoSideEffect]
    __glsl_version(450)
    __glsl_extension(GL_NV_shader_texture_footprint)
    [__requiresNVAPI]
    [require(glsl_spirv, texturefootprint)]
    bool __queryFootprintLodGLSL(
            SamplerState            sampler,
            Coords                  coords,
            float                   lod,
            int                     granularity,
            bool                    useCoarseLevel,
            out __FootprintData         footprint)
    {
        __target_switch
        {
        case glsl:
            __intrinsic_asm "textureFootprintLodNV($p, $*2)";
        case spirv:
            return spirv_asm {
                OpCapability ImageFootprintNV;
                OpExtension "SPV_NV_shader_image_footprint";
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $sampler;
                %resultVal:$$__FootprintData = OpImageSampleFootprintNV %sampledImage $coords $granularity $useCoarseLevel Lod $lod;
                OpStore &footprint %resultVal;
                result:$$bool = OpCompositeExtract %resultVal 0;
            };
        }
    }


${{{
    // Texture sampling with gradient is only available for 2D textures.
}}}
    [__NoSideEffect]
    __glsl_version(450)
    __glsl_extension(GL_NV_shader_texture_footprint)
    [__requiresNVAPI]
    [require(glsl_spirv, texturefootprint)]
    bool __queryFootprintGradGLSL(
            SamplerState    sampler,
            Coords          coords,
            Coords          dx,
            Coords          dy,
            int             granularity,
            bool            useCoarseLevel,
            out __FootprintData footprint)
    {
        __target_switch
        {
        case glsl:
            __intrinsic_asm "textureFootprintGradNV($p, $*2)";
        case spirv:
            return spirv_asm {
                OpCapability ImageFootprintNV;
                OpExtension "SPV_NV_shader_image_footprint";
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $sampler;
                %resultVal:$$__FootprintData = OpImageSampleFootprintNV %sampledImage $coords $granularity $useCoarseLevel Grad $dx $dy;
                OpStore &footprint %resultVal;
                result:$$bool = OpCompositeExtract %resultVal 0;
            };
        }
    }

    [__NoSideEffect]
    __glsl_version(450)
    __glsl_extension(GL_NV_shader_texture_footprint)
    __glsl_extension(GL_ARB_sparse_texture_clamp)
    [require(glsl_spirv, texturefootprintclamp)]
    bool __queryFootprintGradClampGLSL(
            SamplerState    sampler,
            Coords          coords,
            Coords          dx,
            Coords          dy,
            float           lodClamp,
            int             granularity,
            bool            useCoarseLevel,
            out __FootprintData footprint)
    {
        __target_switch
        {
        case glsl:
            __intrinsic_asm "textureFootprintGradClampNV($p, $*2)";
        case spirv:
            return spirv_asm {
                OpCapability ImageFootprintNV;
                OpCapability MinLod;
                OpExtension "SPV_NV_shader_image_footprint";
                %sampledImage : __sampledImageType(this) = OpSampledImage $this $sampler;
                %resultVal:$$__FootprintData = OpImageSampleFootprintNV %sampledImage $coords $granularity $useCoarseLevel Grad|MinLod $dx $dy $lodClamp;
                OpStore &footprint %resultVal;
                result:$$bool = OpCompositeExtract %resultVal 0;
            };
        }
    }
${{{
    // End texture2D specific functions.
}}}


${{{{
// The NVAPI texture query operations encode the choice
// between coarse and fine levels as part of the function
// name, and so we are forced to match this convention
// if we want to provide a more portable API.
//
// TODO: We could conceivably define the functions to use
// a parameter for the coarse/fine choice, which is required
// to be `constexpr` for the HLSL/NVAPI target.
//
static const struct LevelChoice
{
char const* name;
char const* isCoarseVal;
} kLevelChoices[] =
{
    { "Coarse", "true" },
    { "Fine", "false" },
};
for(auto levelChoice : kLevelChoices)
{
    auto CoarseOrFine = levelChoice.name;
    auto isCoarseVal = levelChoice.isCoarseVal;

// We now go ahead and define the intrinsics provided by NVAPI,
// which have a very different signature from the GLSL ones.
//
// Note: the NVAPI functions also support an optional texel
// offset parameter. For now we are not including overloads
// with that parameter, since they have no equivalent in
// the GLSL extension.
//
}}}}

    [__NoSideEffect]
    [__requiresNVAPI]
    [require(hlsl, texturefootprint)]
    static __FootprintData __queryFootprint$(CoarseOrFine)NVAPI(
        int                     nd,
        uint                    textureSpace,
        uint                    textureIndex,
        uint                    samplerSpace,
        uint                    samplerIndex,
        float3                  coords,
        FootprintGranularity    granularity, 
        out uint                isSingleLod)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "NvFootprint$(CoarseOrFine)($1, $2, $3, $4, NV_EXTN_TEXTURE_$!0D, $*5)";
        }
    }

    [__NoSideEffect]
    [__requiresNVAPI]
    [require(hlsl, texturefootprint)]
    static __FootprintData __queryFootprint$(CoarseOrFine)BiasNVAPI(
        int                     nd,
        uint                    textureSpace,
        uint                    textureIndex,
        uint                    samplerSpace,
        uint                    samplerIndex,
        float3                  coords,
        FootprintGranularity    granularity,
        float                   lodBias, 
        out uint                isSingleLod)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "NvFootprint$(CoarseOrFine)Bias($1, $2, $3, $4, NV_EXTN_TEXTURE_$!0D, $*5)";
        }
    }

    [__NoSideEffect]
    [__requiresNVAPI]
    [require(hlsl, texturefootprint)]
    static __FootprintData __queryFootprint$(CoarseOrFine)LevelNVAPI(
        int                     nd,
        uint                    textureSpace,
        uint                    textureIndex,
        uint                    samplerSpace,
        uint                    samplerIndex,
        float3                  coords,
        FootprintGranularity    granularity,
        float                   lod, 
        out uint                isSingleLod)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "NvFootprint$(CoarseOrFine)Level($1, $2, $3, $4, NV_EXTN_TEXTURE_$!0D, $*5)";
        }
    }

    [__NoSideEffect]
    [__requiresNVAPI]
    [require(hlsl, texturefootprint)]
    static __FootprintData __queryFootprint$(CoarseOrFine)GradNVAPI(
        int                     nd,
        uint                    textureSpace,
        uint                    textureIndex,
        uint                    samplerSpace,
        uint                    samplerIndex,
        float3                  coords,
        FootprintGranularity    granularity,
        float3                  dx,
        float3                  dy, 
        out uint                isSingleLod)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm "NvFootprint$(CoarseOrFine)Grad($1, $2, $3, $4, NV_EXTN_TEXTURE_$!0D, $*5)";
        }
    }

${
// We now define the portable operations that will be officially
// supported by the standard module. For each operation, we
// need to provide both a version that maps to the GLSL extension,
// and a version that uses the NVAPI functions.
//
// Some function variations are only available with one extension
// or the other, so we try our best to only define them where
// each is available.
//
// Note that these functions cannot be marked as [ForceInline] for now
// because the texture resource may get removed after DCE, since the only
// use of those resources are done through __GetRegisterIndex/Space, which is
// replaced early with their binding slot in the compilation process.
// Not inlining these function is a quick way to make sure the texture always
// has live uses.
//
}

    /// Query the footprint that would be accessed by a texture sampling operation.
    ///
    /// This operation queries the footprint that would be accessed
    /// by a comparable call to:
    ///
    ///     t.Sample(sampler, coords);
    ///
    [__NoSideEffect]
    Footprint queryFootprint$(CoarseOrFine)(
            FootprintGranularity    granularity,
            SamplerState            sampler,
            Coords                  coords)
    {
        __target_switch
        {
        case glsl:
        case spirv:
            Footprint footprint;
            footprint._isSingleLevel = __queryFootprintGLSL(sampler, coords, granularity, $(isCoarseVal), footprint);
            return footprint;

        case hlsl:
            uint isSingleLod = 0;
            __queryFootprint$(CoarseOrFine)NVAPI(
                Shape.dimensions,
                __getRegisterSpace(this), __getRegisterIndex(this),
                __getRegisterSpace(sampler), __getRegisterIndex(sampler),
                __vectorReshape<3>(coords), granularity, /* out */isSingleLod);
            Footprint footprint = {false};
            footprint._isSingleLevel = (isSingleLod != 0);
            return footprint;
        }
    }

    /// Query the footprint that would be accessed by a texture sampling operation.
    ///
    /// This operation queries the footprint that would be accessed
    /// by a comparable call to:
    ///
    ///     t.SampleBias(sampler, coords, lodBias);
    ///
    [__NoSideEffect]
    Footprint queryFootprint$(CoarseOrFine)Bias(
            FootprintGranularity    granularity,
            SamplerState            sampler,
            Coords                  coords,
            float                   lodBias)
    {
        __target_switch
        {
        case glsl:
        case spirv:
            Footprint footprint;
            footprint._isSingleLevel = __queryFootprintGLSL(sampler, coords, granularity, $(isCoarseVal), footprint, lodBias);
            return footprint;
        case hlsl:
            uint isSingleLod = 0;
            __queryFootprint$(CoarseOrFine)BiasNVAPI(
                Shape.dimensions,
                __getRegisterSpace(this), __getRegisterIndex(this),
                __getRegisterSpace(sampler), __getRegisterIndex(sampler),
                __vectorReshape<3>(coords), granularity, lodBias, /* out */isSingleLod);
            Footprint footprint = {false};
            footprint._isSingleLevel = (isSingleLod != 0);
            return footprint;
        }
    }

    /// Query the footprint that would be accessed by a texture sampling operation.
    ///
    /// This operation queries the footprint that would be accessed
    /// by a comparable call to:
    ///
    ///     t.SampleClamp(sampler, coords, lodClamp);
    ///
    [__NoSideEffect]
    Footprint queryFootprint$(CoarseOrFine)Clamp(
            FootprintGranularity    granularity,
            SamplerState            sampler,
            Coords                  coords,
            float                   lodClamp)
    {
        __target_switch
        {
        case glsl:
        case spirv:
            Footprint footprint;
            footprint._isSingleLevel = __queryFootprintClampGLSL(sampler, coords, lodClamp, granularity, $(isCoarseVal), footprint);
            return footprint;
        }
    }

    /// Query the footprint that would be accessed by a texture sampling operation.
    ///
    /// This operation queries the footprint that would be accessed
    /// by a comparable call to:
    ///
    ///     t.SampleBiasClamp(sampler, coords, lodBias, lodClamp);
    ///
    [__NoSideEffect]
    Footprint queryFootprint$(CoarseOrFine)BiasClamp(
            FootprintGranularity    granularity,
            SamplerState            sampler,
            Coords                  coords,
            float                   lodBias,
            float                   lodClamp)
    {
        __target_switch
        {
        case glsl:
        case spirv:
            Footprint footprint;
            footprint._isSingleLevel = __queryFootprintClampGLSL(sampler, coords, lodClamp, granularity, $(isCoarseVal), footprint, lodBias);
            return footprint;
        }
    }

    /// Query the footprint that would be accessed by a texture sampling operation.
    ///
    /// This operation queries the footprint that would be accessed
    /// by a comparable call to:
    ///
    ///     t.SampleLevel(sampler, coords, lod);
    ///
    [__NoSideEffect]
    Footprint queryFootprint$(CoarseOrFine)Level(
            FootprintGranularity    granularity,
            SamplerState            sampler,
            Coords                  coords,
            float                   lod)
    {
        __target_switch
        {
        case glsl:
        case spirv:
            Footprint footprint;
            footprint._isSingleLevel = __queryFootprintLodGLSL(sampler, coords, lod, granularity, $(isCoarseVal), footprint);
            return footprint;
        case hlsl:
            uint isSingleLod = 0;
            __queryFootprint$(CoarseOrFine)LevelNVAPI(
                Shape.dimensions,
                __getRegisterSpace(this), __getRegisterIndex(this),
                __getRegisterSpace(sampler), __getRegisterIndex(sampler),
                __vectorReshape<3>(coords), granularity, lod, /* out */isSingleLod);
            Footprint footprint = {false};
            footprint._isSingleLevel = (isSingleLod != 0);
            return footprint;
        }
    }

${{{
    // TODO: Texture sampling with gradient is only available for 2D textures.
}}}

    /// Query the footprint that would be accessed by a texture sampling operation.
    ///
    /// This operation queries the footprint that would be accessed
    /// by a comparable call to:
    ///
    ///     t.SampleGrad(sampler, coords, dx, dy);
    ///
    [__NoSideEffect] [ForceInline]
    Footprint queryFootprint$(CoarseOrFine)Grad(
            FootprintGranularity    granularity,
            SamplerState            sampler,
            Coords                  coords,
            Coords                  dx,
            Coords                  dy)
    {
        __target_switch
        {
        case glsl:
        case spirv:
            Footprint footprint;
            footprint._isSingleLevel = __queryFootprintGradGLSL(sampler, coords, dx, dy, granularity, $(isCoarseVal), footprint);
            return footprint;
        case hlsl:
            uint isSingleLod = 0;
            __queryFootprint$(CoarseOrFine)GradNVAPI(
                Shape.dimensions,
                __getRegisterSpace(this), __getRegisterIndex(this),
                __getRegisterSpace(sampler), __getRegisterIndex(sampler),
                __vectorReshape<3>(coords), granularity, __vectorReshape<3>(dx), __vectorReshape<3>(dy), /* out */isSingleLod);
            Footprint footprint = {false};
            footprint._isSingleLevel = (isSingleLod != 0);
            return footprint;
        }
    }

    /// Query the footprint that would be accessed by a texture sampling operation.
    ///
    /// This operation queries the footprint that would be accessed
    /// by a comparable call to:
    ///
    ///     t.SampleGradClamp(sampler, coords, dx, dy, lodClamp);
    ///
    [__NoSideEffect][ForceInline]
    Footprint queryFootprint$(CoarseOrFine)GradClamp(
            FootprintGranularity    granularity,
            SamplerState            sampler,
            Coords                  coords,
            Coords                  dx,
            Coords                  dy,
            float                   lodClamp)
    {
        __target_switch
        {
        case glsl:
        case spirv:
            Footprint footprint;
            footprint._isSingleLevel = __queryFootprintGradClampGLSL(sampler, coords, dx, dy, lodClamp, granularity, $(isCoarseVal), footprint);
            return footprint;
        }
    }

${{{
    // TODO: end texture2D specific functions.
}}}

${{{{
}
}}}}

} // extension

//<T, Shape: __ITextureShape, let isArray:int, let isMS:int, let sampleCount:int, let access:int, let isShadow:int, let isCombined:int, let format:int>
__generic<Shape:__ITextureShape1D2D3D, let format : int>
extension _Texture<float, Shape, 0, 0, 0, $(kCoreModule_ResourceAccessReadWrite), 0, 0, format>
{
    [__requiresNVAPI]
    [ForceInline]
    __glsl_extension(GL_EXT_shader_atomic_float)
    [require(glsl_hlsl_spirv, atomic_glsl_hlsl_nvapi_cuda_metal_float1)]
    void InterlockedAddF32(vector<uint, Shape.dimensions> coord, float value, out float originalValue)
    {
        __target_switch
        {
        default:
            originalValue = __atomic_add(this[coord], value);
            return;
        case hlsl:
            __intrinsic_asm "$3 = NvInterlockedAddFp32($0, $1, $2)";
        case glsl:
            __intrinsic_asm "$3 = imageAtomicAdd($0, $1, $2)";
        }
    }

    [ForceInline]
    float InterlockedAddF32(vector<uint, Shape.dimensions> coord, float value)
    {
        float originalValue;
        InterlockedAddF32(coord, value, originalValue);
        return originalValue;
    }
}

// Buffer Pointer

//@hidden:

namespace vk
{
    // Partial implementation of the vk::buffer_ref proposal:
    // https://github.com/microsoft/hlsl-specs/blob/main/proposals/0010-vk-buffer-ref.md
    struct BufferPointer<T, let Alignment : int = 0>
    {
        T *_ptr;
        [ForceInline] __init(T *ptr) { _ptr = ptr; }
        [ForceInline] __init(uint64_t val) { _ptr = (T *)val; }
        [ForceInline] Ref<T> Get() { return *_ptr; }
        [ForceInline] T *getPtr() { return _ptr;}
    }
    [ForceInline]
    BufferPointer<U, alignment> static_pointer_cast<U, let alignment : int = 0, T, let a : int>(BufferPointer<T, a> src)
    {
        return BufferPointer<U, alignment>((U*)(src.getPtr()));
    }
    [ForceInline]
    BufferPointer<U, alignment> reinterpret_pointer_cast<U, let alignment : int = 0, T, let a : int>(BufferPointer<T, a> src)
    {
        return BufferPointer<U, alignment>((U *)(src.getPtr()));
    }
}

attribute_syntax[vk_aliased_pointer] : VkAliasedPointerAttribute;
attribute_syntax[vk_restrict_pointer] : VkRestrictPointerAttribute;

extension uint64_t
{
    __init<T, let alignment : int>(vk::BufferPointer<T, alignment> ptr)
    {
        this = (uint64_t)ptr._ptr;
    }
}

struct ConstBufferPointer<T, int alignment = 16>
{
    T *_ptr;

    [ForceInline] T get() { return loadAligned<alignment>(_ptr); }

    __subscript(int index) -> T
    {
        [ForceInline]
        get { return _ptr[index]; }
    }

    [ForceInline] T* getPtr() { return _ptr; }

    [ForceInline]
    static ConstBufferPointer<T> fromUInt(uint64_t val)
    {
        return {(T*)val};
    }

    [ForceInline]
    uint64_t toUInt()
    {
        return (uint64_t)_ptr;
    }

    [ForceInline]
    bool isValid()
    {
        return _ptr != nullptr;
    }
}

//
// HLSL-like dynamic resources
// https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_DynamicResources.html
//
// For Khronos targets, `__DynamicResource` can be used to declare "untyped" global bindings as
// usual (e.g. unsized arrays for descriptor indexing), which will then be materialized into 
// new aliased bindings for each distinct cast type.
//

//@public:

/// Represent the kind of a descriptor type.
enum DescriptorKind
{
    Unknown, /// Unknown descriptor kind.
    Texture, /// A texture descriptor.
    CombinedTextureSampler, /// A combined texture and sampler state descriptor.
    Buffer, /// A buffer descriptor.
    Sampler, /// A sampler state descriptor.
    AccelerationStructure, /// A ray tracing acceleration structure descriptor.
}

enum DescriptorAccess
{
    Unknown = -1,
    Read = $(kCoreModule_ResourceAccessReadOnly),
    Write = $(kCoreModule_ResourceAccessWriteOnly),
    ReadWrite = $(kCoreModule_ResourceAccessReadWrite),
    RasterizerOrdered = $(kCoreModule_ResourceAccessRasterizerOrdered),
    Feedback = $(kCoreModule_ResourceAccessFeedback),
}

/// Represents an opaque descriptor type, such as textures, samplers, and buffers etc,
/// whose size may be undefined and can't be directly accessed as ordinary data.
[sealed]
[builtin]
interface IOpaqueDescriptor
{
    /// The kind of the descriptor.
    static const DescriptorKind kind;
    static const DescriptorAccess descriptorAccess;
}

__magic_type(DynamicResourceType)
__intrinsic_type($(kIROp_DynamicResourceType))
struct __DynamicResource<let kind = __DynamicResourceKind.General>
{
    __intrinsic_op($(kIROp_CastDynamicResource))
    T as<T : __IDynamicResourceCastable<kind>>();

    __intrinsic_op($(kIROp_CastDynamicResource))
    T asOpaqueDescriptor<T : IOpaqueDescriptor>();
}
interface __IDynamicResourceCastable<let kind = __DynamicResourceKind.General> : IOpaqueDescriptor
{
}

enum __DynamicResourceKind
{
    General = 0, // CBV_SRV_UAV
    Sampler = 1
}

__generic<T:ITexelElement, Shape : __ITextureShape, let isArray : int, let isMS : int, let sampleCount : int, let access : int, let isShadow : int, let isCombined : int, let format : int>
extension _Texture<T, Shape, isArray, isMS, sampleCount, access, isShadow, isCombined, format> : __IDynamicResourceCastable<__DynamicResourceKind.General>
{
    __intrinsic_op($(kIROp_CastDynamicResource))
    __implicit_conversion($(kConversionCost_GenericParamUpcast))
    __init(__DynamicResource res);

    typealias Handle = DescriptorHandle<This>;

    static const DescriptorKind kind = isCombined != 0 ? DescriptorKind.CombinedTextureSampler : DescriptorKind.Texture;
    static const DescriptorAccess descriptorAccess = (DescriptorAccess)access;

    __implicit_conversion($(kConversionCost_ImplicitDereference))
    [ForceInline]
    __init(DescriptorHandle<This> bindless)
    {
        return getDescriptorFromHandle(bindless);
    }
}

${{{{
struct DynamicResourceTypeInfo
{
    const char* name;
    const char* kind;
    const char* dynamicKind;
    const char* access;
};

const DynamicResourceTypeInfo kDynamicResourceCastableTypes[] = {
    { "StructuredBuffer<T, L>", "Buffer", "General", "Read" },
    { "RWStructuredBuffer<T, L>", "Buffer", "General", "ReadWrite" },
    { "AppendStructuredBuffer<T, L>", "Buffer", "General", "ReadWrite" },
    { "ConsumeStructuredBuffer<T, L>", "Buffer", "General", "ReadWrite" },
    { "RasterizerOrderedStructuredBuffer<T, L>", "Buffer", "General", "ReadWrite" },
    { "ByteAddressBuffer", "Buffer", "General", "Read" },
    { "RWByteAddressBuffer", "Buffer", "General", "ReadWrite" },
    { "RasterizerOrderedByteAddressBuffer", "Buffer", "General", "ReadWrite" },
    { "SamplerState", "Sampler", "Sampler", "Unknown" },
    { "SamplerComparisonState", "Sampler", "Sampler", "Unknown" },
    { "ConstantBuffer<T, L>", "Buffer", "General", "Read"},
    { "TextureBuffer<T>", "Buffer", "General", "Read"},
    { "RaytracingAccelerationStructure", "AccelerationStructure", "General", "Read"},
};

for (auto type : kDynamicResourceCastableTypes) {
    auto dynamicKind = type.dynamicKind;
    auto kind = type.kind;
    auto typeName = type.name;
    auto access = type.access;
    if (strstr(typeName, "<T, L>"))
        sb << "__generic<T, L : IBufferDataLayout>\n";
    else if (strstr(typeName, "<T>"))
        sb << "__generic<T>\n";
}}}}

extension $(typeName) : __IDynamicResourceCastable<__DynamicResourceKind.$(dynamicKind)>
{
    __intrinsic_op($(kIROp_CastDynamicResource))
    __implicit_conversion($(kConversionCost_GenericParamUpcast))
    __init(__DynamicResource res);

    static const DescriptorKind kind = DescriptorKind.$(kind);
    static const DescriptorAccess descriptorAccess = DescriptorAccess.$(access);

    typealias Handle = DescriptorHandle<$(typeName)>;

    __implicit_conversion($(kConversionCost_ImplicitDereference))
    [ForceInline]
    __init(DescriptorHandle<$(typeName)> bindless)
    {
        return getDescriptorFromHandle(bindless);
    }
}

${{{{
}
}}}}


/// Represents a bindless handle to a descriptor. A descriptor handle is always an ordinary data type and can be
/// declared in any memory location.
/// @remarks Opaque descriptor types such as textures(`Texture2D` etc.), `SamplerState` and buffers (e.g. `StructuredBuffer`)
/// can have undefined size and data representation on many targets. On platforms such as Vulkan and D3D12, descriptors are
/// communicated to the shader code by calling the host side API to write the descriptor into a descriptor set or table, instead
/// of directly writing bytes into an ordinary GPU accessible buffer. As a result, oapque handle types cannot be used in places
/// that refer to a ordinary buffer location, such as as element types of a `StructuredBuffer`.
/// However, a `DescriptorHandle<T>` stores a handle (or address) to the actual descriptor, and is always an ordinary data type
/// that can be manipulated directly in the shader code. This gives the developer the flexibility to embed and pass around descriptor
/// parameters throughout the code, to enable cleaner modular designs.
/// See [User Guide](https://shader-slang.com/slang/user-guide/convenience-features.html#descriptorhandle-for-bindless-descriptor-access)
/// for more information on how to use `DescriptorHandle<T>` in your code.
__magic_type(DescriptorHandleType)
__intrinsic_type($(kIROp_DescriptorHandleType))
struct DescriptorHandle<T:IOpaqueDescriptor> : IComparable
{
    [require(glsl_spirv)]
    [require(hlsl, sm_6_6)]
    [require(wgsl)]
    __intrinsic_op($(kIROp_CastUInt2ToDescriptorHandle))
    __init(uint2 handleValue);

    [ForceInline]
    bool equals(DescriptorHandle<T> other) { return all(__vectorEql((uint2)this, (uint2)other)); }

    [ForceInline]
    bool lessThan(DescriptorHandle<T> other) { let vthis = ((uint2)this); let vother = (uint2)other; return vthis.x < vother.x || (vthis.x == vother.x && vthis.y < vother.y); }

    [ForceInline]
    bool lessThanOrEquals(DescriptorHandle<T> other) { let vthis = ((uint2)this); let vother = (uint2)other; return vthis.x < vother.x || (vthis.x == vother.x && vthis.y <= vother.y); }
}

extension uint2
{
    __intrinsic_op($(kIROp_CastDescriptorHandleToUInt2))
    [require(glsl_spirv)]
    [require(hlsl, sm_6_6)]
    [require(wgsl)]
    __init<T:IOpaqueDescriptor>(DescriptorHandle<T> bindless);
}

__generic<T:IOpaqueDescriptor>
[ForceInline]
__prefix T operator*(DescriptorHandle<T> value)
{
    return getDescriptorFromHandle(value);
}

__intrinsic_op($(kIROp_GetDynamicResourceHeap))
T[] __getDynamicResourceHeap<T:IOpaqueDescriptor>();


// Used by the HLSL backend only, load a sampler state handle from SamplerDescriptorHeap.
__intrinsic_op($(kIROp_LoadSamplerDescriptorFromHeap))
T __loadSamplerDescriptorFromHeap<T>(uint index);

// Used by the HLSL backend only, load a resource handle from ResourceDescriptorHeap.
__intrinsic_op($(kIROp_LoadResourceDescriptorFromHeap))
T __loadResourceDescriptorFromHeap<T>(uint index);

// Used by the HLSL and Metal backends only,
// create a combined texture sampler object from a bindless handle.
__intrinsic_op($(kIROp_MakeCombinedTextureSamplerFromHandle))
T __makeCombinedTextureSamplerFromHandle<T, U>(U handle);

__intrinsic_op($(kIROp_CastDescriptorHandleToResource))
T __castDescriptorHandleToResource<T:IOpaqueDescriptor>(DescriptorHandle<T> ptr);

/// The default implementation of `getDescriptorFromHandle`, which converts from a descriptor handle
/// to a descriptor object.
[ForceInline]
T defaultGetDescriptorFromHandle<T:IOpaqueDescriptor>(DescriptorHandle<T> handleValue)
{
    __target_switch
    {
    case hlsl:
        if (T.kind == DescriptorKind.Sampler)
            return __loadSamplerDescriptorFromHeap<T>(((uint2)handleValue).x);
        else if (T.kind == DescriptorKind.CombinedTextureSampler)
            return __makeCombinedTextureSamplerFromHandle<T>((uint2)handleValue);
        else
            return __loadResourceDescriptorFromHeap<T>(((uint2)handleValue).x);
    case spirv:
    case glsl:
    case wgsl:
        return __getDynamicResourceHeap<T>()[((uint2)handleValue).x];
    default:
        return __castDescriptorHandleToResource<T>(handleValue);
    }
}

/// Declaration of the `getDescriptorFromHandle` that the user code can provide to customize
/// how a descriptor handle is converted into a actual descriptor.
[ForceInline]
extern T getDescriptorFromHandle<T:IOpaqueDescriptor>(DescriptorHandle<T> handleValue)
{
    return defaultGetDescriptorFromHandle(handleValue);
}

__intrinsic_op($(kIROp_NonUniformResourceIndex))
DescriptorHandle<T> nonuniform<T:IOpaqueDescriptor>(DescriptorHandle<T> ptr);

__glsl_version(450)
__glsl_extension(GL_ARB_shader_clock)
[require(glsl_spirv, GL_ARB_shader_clock)]
uint2 clock2x32ARB()
{
    __target_switch
    {
    case glsl: __intrinsic_asm "clock2x32ARB";
    case spirv:
        const uint32_t scopeId_subgroup = 3;
        return spirv_asm {
        OpCapability ShaderClockKHR;
        OpExtension "SPV_KHR_shader_clock";
        result:$$uint2 = OpReadClockKHR $scopeId_subgroup;
        };
    }
}

__glsl_version(450)
__glsl_extension(GL_ARB_shader_clock)
__glsl_extension(GL_ARB_gpu_shader_int64)
[require(glsl_spirv, GL_ARB_shader_clock64)]
uint64_t clockARB()
{
    __target_switch
    {
    case glsl: __intrinsic_asm "clockARB";
    case spirv:
        const uint32_t scopeId_subgroup = 3;
        return spirv_asm {
            OpCapability ShaderClockKHR;
            OpExtension "SPV_KHR_shader_clock";
            result:$$uint64_t = OpReadClockKHR $scopeId_subgroup;
        };
    }
}

extension<T, L : IBufferDataLayout> StructuredBuffer<T, L> : IArray<T>
{
    int getCount() { uint count; uint stride; this.GetDimensions(count, stride); return count; }
}

extension<T, L : IBufferDataLayout> RWStructuredBuffer<T, L> : IRWArray<T>
{
    int getCount() { uint count; uint stride; this.GetDimensions(count, stride); return count; }
}

extension<T, L : IBufferDataLayout> RasterizerOrderedStructuredBuffer<T, L> : IRWArray<T>
{
    int getCount() { uint count; uint stride; this.GetDimensions(count, stride); return count; }
}

//
// Cooperative Matrix type
//

__intrinsic_type($(kIROp_CoopMatrixType))
[require(cooperative_matrix)]
struct CoopMat<T : __BuiltinArithmeticType, let S : CoopMatScope, let M : int, let N : int, let R : CoopMatMatrixUse> : IArray<T>, IArithmetic
{
    //
    // Initialization
    //

    [ForceInline]
    [require(cooperative_matrix)]
    __init()
    {
    }

    [ForceInline]
    [require(cooperative_matrix)]
    __init(T t)
    {
        this.fill(t);
    }

    [ForceInline]
    [require(cooperative_matrix)]
    __init<U : __BuiltinArithmeticType>(CoopMat<U, S, M, N, R> other)
    {
        this.copyFrom(other);
    }

    [ForceInline]
    __init(This x)
    {
        this = x;
    }

    // Required for `IArithmetic`.
    [OverloadRank(-10)]
    [ForceInline]
    __init(int i)
    {
        this = CoopMat<T, S, M, N, R>(T(i));
    }

    //
    // Simple setters
    //

    [require(cooperative_matrix)]
    [mutating]
    [ForceInline]
    void fill(T t)
    {
        this = spirv_asm
        {
            result:$$CoopMat<T, S, M, N, R> = OpConstantComposite $t;
        };
    }

    [require(cooperative_matrix)]
    [mutating]
    [ForceInline]
    void copyFrom<U : __BuiltinArithmeticType>(CoopMat<U, S, M, N, R> other)
    {
        if (__isFloat<T>() && __isInt<U>())
            this = __int_to_float_cast<T>(other);
        else if (__isInt<T>() && __isFloat<U>())
            this = __float_to_int_cast<T>(other);
        else if (__isFloat<T>() && __isFloat<U>())
            this = __real_cast<T>(other);
        else if (__isInt<T>() && __isInt<U>())
            this = __int_cast<T>(other);
    }

    //
    // Subscript
    //

    __intrinsic_op($(kIROp_GetElement))
    [__NoSideEffect]
    T __indexRead(int index);

    __intrinsic_op($(kIROp_GetElementPtr))
    [__ref]
    [__NoSideEffect]
    Ref<T> __indexRef(int index);

    [ForceInline]
    [__NoSideEffect]
    int getCount()
    {
        return getLength();
    }

    [ForceInline]
    [__NoSideEffect]
    int getRowCount()
    {
        return M;
    }

    [ForceInline]
    [__NoSideEffect]
    int getColumnCount()
    {
        return N;
    }

    __subscript(int index) -> T
    {
        [__NoSideEffect]
        [nonmutating]
        get
        {
            return __indexRead(index);
        }

        [mutating]
        set
        {
            __indexRef(index) = newValue;
        }
    }

    /// Returns the number of components owned by each invocation.
    [ForceInline]
    [require(cooperative_matrix)]
    uint getLength()
    {
        return spirv_asm
        {
            result:$$uint = OpCooperativeMatrixLengthKHR $$This;
        };
    }

    //
    // Store
    //

    [ForceInline]
    [require(cooperative_matrix)]
    void store(RWByteAddressBuffer buffer, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
    {
        return store(__getEquivalentStructuredBuffer<T>(buffer), element, stride, matrixLayout);
    }

    [ForceInline]
    [require(cooperative_matrix)]
    void store(RWStructuredBuffer<T> buffer, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
    {
        let zero = 0;
        let alignment = 16U;
        spirv_asm
        {
            %storagePointerType = OpTypePointer StorageBuffer $$T;
            %pointer:%storagePointerType = OpAccessChain $buffer $zero $element;
            OpCooperativeMatrixStoreKHR %pointer $this $matrixLayout $stride Aligned !alignment;
        };
    }

    [__NoSideEffect]
    [ForceInline]
    [require(cooperative_matrix)]
    void store(T* buffer, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
    {
        let alignment = 16U;
        return spirv_asm
        {
            %pointer:$$T* = OpPtrAccessChain $buffer $element;
            OpCooperativeMatrixStoreKHR %pointer $this $matrixLayout $stride Aligned !alignment;
        };
    }

    [require(cooperative_matrix)]
    [ForceInline]
    void store<let U : int>(__ref groupshared T[U] data, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
    {
        let alignment = 16U;
        spirv_asm
        {
            %workgroupPointerType = OpTypePointer Workgroup $$T;
            %pointer:%workgroupPointerType = OpAccessChain &data $element;
            OpCooperativeMatrixStoreKHR %pointer $this $matrixLayout $stride Aligned !alignment;
        };
    }

    [ForceInline]
    [ForceInline]
    [require(cooperative_matrix)]
    void storeAny<U, let V : int>(__ref groupshared U[V] data, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
    {
        let alignment = 16U;
        spirv_asm
        {
            %workgroupPointerType = OpTypePointer Workgroup $$U;
            %pointer:%workgroupPointerType = OpAccessChain &data $element;
            OpCooperativeMatrixStoreKHR %pointer $this $matrixLayout $stride Aligned !alignment;
        };
    }

    [ForceInline]
    [ForceInline]
    [require(cooperative_matrix)]
    void storeAny<U, let V : int, let L : int>(__ref groupshared vector<U, L>[V] data, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
    {
        let alignment = 16U;
        spirv_asm
        {
            %workgroupPointerType = OpTypePointer Workgroup $$vector<U, L>;
            %pointer:%workgroupPointerType = OpAccessChain &data $element;
            OpCooperativeMatrixStoreKHR %pointer $this $matrixLayout $stride Aligned !alignment;
        };
    }

    //
    // Load
    //

    [__NoSideEffect]
    [ForceInline]
    [require(cooperative_matrix)]
    static CoopMat<T, S, M, N, R> load(ByteAddressBuffer buffer, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
    {
        return load(__getEquivalentStructuredBuffer<T>(buffer), element, stride, matrixLayout);
    }

    [__NoSideEffect]
    [ForceInline]
    [require(cooperative_matrix)]
    static CoopMat<T, S, M, N, R> load(RWByteAddressBuffer buffer, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
    {
        return load(__getEquivalentStructuredBuffer<T>(buffer), element, stride, matrixLayout);
    }

    [__NoSideEffect]
    [ForceInline]
    [require(cooperative_matrix)]
    static CoopMat<T, S, M, N, R> load(StructuredBuffer<T> buffer, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
    {
        let zero = 0;
        let alignment = 16U;
        return spirv_asm
        {
            %storagePointerType = OpTypePointer StorageBuffer $$T;
            %pointer:%storagePointerType = OpAccessChain $buffer $zero $element;
            result:$$CoopMat<T, S, M, N, R> = OpCooperativeMatrixLoadKHR %pointer $matrixLayout $stride Aligned !alignment;
        };
    }

    [__NoSideEffect]
    [ForceInline]
    [require(cooperative_matrix)]
    static CoopMat<T, S, M, N, R> load(RWStructuredBuffer<T> buffer, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
    {
        let zero = 0;
        let alignment = 16U;
        return spirv_asm
        {
            %storagePointerType = OpTypePointer StorageBuffer $$T;
            %pointer:%storagePointerType = OpAccessChain $buffer $zero $element;
            result:$$CoopMat<T, S, M, N, R> = OpCooperativeMatrixLoadKHR %pointer $matrixLayout $stride Aligned !alignment;
        };
    }

    [__NoSideEffect]
    [ForceInline]
    [require(cooperative_matrix)]
    static CoopMat<T, S, M, N, R> load(T* buffer, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
    {
        let alignment = 16;
        return spirv_asm
        {
            %pointer:$$T* = OpPtrAccessChain $buffer $element;
            result:$$CoopMat<T, S, M, N, R> = OpCooperativeMatrixLoadKHR %pointer $matrixLayout $stride Aligned !alignment;
        };
    }

    [ForceInline]
    [require(cooperative_matrix)]
    static CoopMat<T, S, M, N, R> load<let U : int>(__constref groupshared T[U] data, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
    {
        let alignment = 16U;
        return spirv_asm
        {
            %workgroupPointerType = OpTypePointer Workgroup $$T;
            %pointer:%workgroupPointerType = OpAccessChain &data $element;
            result:$$CoopMat<T, S, M, N, R> = OpCooperativeMatrixLoadKHR %pointer $matrixLayout $stride Aligned !alignment;
        };
    }

    [ForceInline]
    [require(cooperative_matrix)]
    static CoopMat<T, S, M, N, R> loadAny<U, let V : int>(__constref groupshared U[V] data, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
    {
        let alignment = 16U;
        return spirv_asm
        {
            %workgroupPointerType = OpTypePointer Workgroup $$U;
            %pointer:%workgroupPointerType = OpAccessChain &data $element;
            result:$$CoopMat<T, S, M, N, R> = OpCooperativeMatrixLoadKHR %pointer $matrixLayout $stride Aligned !alignment;
        };
    }

    [ForceInline]
    [require(cooperative_matrix)]
    static CoopMat<T, S, M, N, R> loadAny<U, let V : int, let L : int>(__constref groupshared vector<U, L>[V] data, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
    {
        let alignment = 16U;
        return spirv_asm
        {
            %workgroupPointerType = OpTypePointer Workgroup $$vector<U, L>;
            %pointer:%workgroupPointerType = OpAccessChain &data $element;
            result:$$CoopMat<T, S, M, N, R> = OpCooperativeMatrixLoadKHR %pointer $matrixLayout $stride Aligned !alignment;
        };
    }

    //
    // Arithmetic
    //

    __intrinsic_op($(kIROp_Add))
    This add(This other);

    __intrinsic_op($(kIROp_Sub))
    This sub(This other);

    __intrinsic_op($(kIROp_Mul))
    This mul(This other);

    __intrinsic_op($(kIROp_Div))
    This div(This other);

    __intrinsic_op($(kIROp_Neg))
    This neg();

    This mod(This other)
    {
        This ret;
        for (int i = 0; i < getLength(); ++i)
        {
            ret[i] = this[i] % other[i];
        }
        return ret;
    }

    //
    // Equality and ordering
    //

    bool equals(This other)
    {
        for (int i = 0; i < getLength(); i++)
        {
            if (this[i] != other[i])
            {
                return false;
            }
        }
        return true;
    }

    bool lessThan(This other)
    {
        for (int i = 0; i < getLength(); i++)
        {
            if (this[i] < other[i])
            {
                return true;
            }
            else if (this[i] > other[i])
            {
                return false;
            }
        }
        return false;
    }

    bool lessThanOrEquals(This other)
    {
        for (int i = 0; i < getLength(); i++)
        {
            if (this[i] < other[i])
            {
                return true;
            }
            else if (this[i] > other[i])
            {
                return false;
            }
        }
        return true;
    }
}

//
// Convenience loading functions for cooperative matrices which infer the
// element type for structured buffers, pointers and groupshared arrays.
//

[ForceInline]
[require(cooperative_matrix)]
CoopMat<T, S, M, N, R> coopMatLoad<T : __BuiltinArithmeticType, let S : CoopMatScope, let M : int, let N : int, let R : CoopMatMatrixUse>(ByteAddressBuffer buffer, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
{
    return CoopMat<T, S, M, N, R>.load(buffer, element, stride, matrixLayout);
}

[ForceInline]
[require(cooperative_matrix)]
CoopMat<T, S, M, N, R> coopMatLoad<T : __BuiltinArithmeticType, let S : CoopMatScope, let M : int, let N : int, let R : CoopMatMatrixUse>(RWByteAddressBuffer buffer, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
{
    return CoopMat<T, S, M, N, R>.load(buffer, element, stride, matrixLayout);
}

[ForceInline]
[require(cooperative_matrix)]
CoopMat<T, S, M, N, R> coopMatLoad<T : __BuiltinArithmeticType, let S : CoopMatScope, let M : int, let N : int, let R : CoopMatMatrixUse>(StructuredBuffer<T> buffer, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
{
    return CoopMat<T, S, M, N, R>.load(buffer, element, stride, matrixLayout);
}

[ForceInline]
[require(cooperative_matrix)]
CoopMat<T, S, M, N, R> coopMatLoad<T : __BuiltinArithmeticType, let S : CoopMatScope, let M : int, let N : int, let R : CoopMatMatrixUse>(RWStructuredBuffer<T> buffer, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
{
    return CoopMat<T, S, M, N, R>.load(buffer, element, stride, matrixLayout);
}

[ForceInline]
[require(cooperative_matrix)]
CoopMat<T, S, M, N, R> coopMatLoad<T : __BuiltinArithmeticType, let S : CoopMatScope, let M : int, let N : int, let R : CoopMatMatrixUse>(T* buffer, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
{
    return CoopMat<T, S, M, N, R>.load(buffer, element, stride, matrixLayout);
}

[ForceInline]
[require(cooperative_matrix)]
CoopMat<T, S, M, N, R> coopMatLoad<T : __BuiltinArithmeticType, let S : CoopMatScope, let M : int, let N : int, let R : CoopMatMatrixUse, let U : int>(__constref groupshared T[U] data, uint element, uint stride, constexpr CoopMatMatrixLayout matrixLayout)
{
    return CoopMat<T, S, M, N, R>.load(data, element, stride, matrixLayout);
}

//
// Cooperative Matrix casting
//

__generic<T : __BuiltinArithmeticType, U : __BuiltinArithmeticType, let S : CoopMatScope, let M : int, let N : int, let R : CoopMatMatrixUse>
__intrinsic_op($(kIROp_IntCast))
[require(cooperative_matrix)]
CoopMat<T,S,M,N,R> __int_cast(CoopMat<U,S,M,N,R> val);

__generic<T : __BuiltinArithmeticType, U : __BuiltinArithmeticType, let S : CoopMatScope, let M : int, let N : int, let R : CoopMatMatrixUse>
__intrinsic_op($(kIROp_FloatCast))
[require(cooperative_matrix)]
CoopMat<T,S,M,N,R> __real_cast(CoopMat<U,S,M,N,R> val);

__generic<T : __BuiltinArithmeticType, U : __BuiltinArithmeticType, let S : CoopMatScope, let M : int, let N : int, let R : CoopMatMatrixUse>
__intrinsic_op($(kIROp_CastIntToFloat))
[require(cooperative_matrix)]
CoopMat<T,S,M,N,R> __int_to_float_cast(CoopMat<U,S,M,N,R> val);

__generic<T : __BuiltinArithmeticType, U : __BuiltinArithmeticType, let S : CoopMatScope, let M : int, let N : int, let R : CoopMatMatrixUse>
__intrinsic_op($(kIROp_CastFloatToInt))
[require(cooperative_matrix)]
CoopMat<T,S,M,N,R> __float_to_int_cast(CoopMat<U,S,M,N,R> val);

//
// Cooperative Matrix multiplication with scalar
//

__generic<T : __BuiltinArithmeticType, let S : CoopMatScope, let M : int, let N : int, let R : CoopMatMatrixUse>
[ForceInline]
[require(cooperative_matrix)]
CoopMat<T, S, M, N, R> operator *(CoopMat<T, S, M, N, R> lhs, const T rhs)
{
    return spirv_asm
    {
        result:$$CoopMat<T, S, M, N, R> = OpMatrixTimesScalar $lhs $rhs;
    };
}

__generic<T : __BuiltinArithmeticType, let S : CoopMatScope, let M : int, let N : int, let R : CoopMatMatrixUse>
[ForceInline]
[require(cooperative_matrix)]
CoopMat<T, S, M, N, R> operator *(const T lhs, CoopMat<T, S, M, N, R> rhs)
{
    return rhs * lhs;
}

//
// Cooperative Matrix enums
//

enum CoopMatScope
{
    Device = 1,
    Workgroup = 2,
    Subgroup = 3,
    QueueFamily = 5,
};

enum CoopMatMatrixUse
{
    MatrixA = 0,
    MatrixB = 1,
    MatrixAccumulator = 2,
};

enum CoopMatMatrixLayout
{
    RowMajor = 0,
    ColumnMajor = 1,
};

enum CoopMatMatrixOperands
{
    None = 0x0,
    MatrixASigned = 0x1,
    MatrixBSigned = 0x2,
    MatrixCSigned = 0x4,
    MatrixResultSigned = 0x8,
    SaturatingAccumulation = 0x10,
};

//
// Cooperative Matrix multiply accumulate
//

[require(cooperative_matrix)]
__generic<T : __BuiltinArithmeticType, U : __BuiltinArithmeticType, V : __BuiltinArithmeticType, let S : CoopMatScope, let M : int, let K : int, let N : int, let RA : CoopMatMatrixUse, let RB : CoopMatMatrixUse, let RC : CoopMatMatrixUse>
CoopMat<V, S, M, N, RC> coopMatMulAdd(CoopMat<T, S, M, K, RA> matA, CoopMat<U, S, K, N, RB> matB, CoopMat<V, S, M, N, RC> matC, constexpr CoopMatMatrixOperands operands)
{
    static_assert((RA == CoopMatMatrixUse::MatrixA) && (RB == CoopMatMatrixUse::MatrixB) && (RC == CoopMatMatrixUse::MatrixAccumulator), "matrix uses for `coopMatMulAdd` matrix parameters must be `MatrixA`, `MatrixB` and `MatrixAccumulator`");
    return spirv_asm
    {
        result:$$CoopMat<V, S, M, N, RC> = OpCooperativeMatrixMulAddKHR $matA $matB $matC !operands;
    };
}

//
// Cooperative Vector
//

/// Represents a Cooperative Vector type that is for matrix-vector multiplication that
/// can take an advantage of the hardware acceleration. It can be used for evaluations
/// of neural network in graphics and compute pipeline.
/// @param T The element type of the CoopVec.
/// @param N The vector size.
__intrinsic_type($(kIROp_CoopVectorType))
[require(cooperative_vector)]
struct CoopVec<T : __BuiltinArithmeticType, let N : int> : IArray<T>, IArithmetic
{
    //
    // Initialization 
    //

    [ForceInline]
    [require(cooperative_vector)]
    __init()
    {
        this = CoopVec<T, N>(T(0));
    }

    [ForceInline]
    [require(cooperative_vector)]
    __init(T t)
    {
        this.fill(t);
    }

    [ForceInline]
    [require(cooperative_vector)]
    __init<U : __BuiltinArithmeticType>(CoopVec<U, N> other)
    {
        this.copyFrom(other);
    }

    [ForceInline]
    [require(cooperative_vector)]
    __init<each U : __BuiltinArithmeticType>(expand each U args)
    {
        static_assert(countof(U) == N, "number of arguments to CoopVec constructor must match number of elements");
        this = __makeCoopVec<T, N>(expand (__arithmetic_cast<T>(each args)));
    }

    [OverloadRank(-10)]
    [ForceInline]
    __init(int i)
    {
        this = CoopVec<T, N>(T(i));
    }

    [ForceInline]
    __init(This x)
    {
        this = x;
    }

    //
    // Simple setters
    //

    /// Copy values from another CoopVec instance into this one. The source CoopVec can have a different element type,
    /// in which case appropriate type conversion will be performed.
    /// @param other The source CoopVec to copy from.
    [require(hlsl)] 
    [mutating] 
    [ForceInline]
    void copyFrom<U : __BuiltinArithmeticType>(CoopVec<U,N> other)
    { 
        __target_switch 
        { 
        case hlsl: __intrinsic_asm ".CopyFrom";
        default:
            if (__isFloat<T>() && __isInt<U>())
                this = __int_to_float_cast<T>(other);
            else if (__isInt<T>() && __isFloat<U>())
                this = __float_to_int_cast<T>(other);
            else if (__isFloat<T>() && __isFloat<U>())
                this = __real_cast<T>(other);
            else if (__isInt<T>() && __isInt<U>())
                this = __int_cast<T>(other);
        } 
    }

    /// Fill all elements of this CoopVec with the specified value.
    /// @param t The value to fill all elements with.
    [require(cooperative_vector)] 
    [mutating] 
    [ForceInline]
    void fill(T t)
    { 
        __target_switch
        {
        case spirv:
            this = spirv_asm {
                OpExtension "SPV_EXT_replicated_composites";
                OpCapability ReplicatedCompositesEXT;
                result:$$CoopVec<T, N> = OpCompositeConstructReplicateEXT $t;
            };
        case hlsl:
            case hlsl: __intrinsic_asm ".Fill"; 
        default:
            for(int i = 0; i < N; ++i)
                this[i] = t;
        }
    }

    //
    // Loading and storing
    //

    /// Store all elements of this CoopVec into a buffer at a specified offset.
    /// @param buffer The destination buffer to store the values into.
    /// @param byteOffset16ByteAligned The byte offset from the start of the buffer where the data will be stored. Must be 16-byte aligned.
    [ForceInline]
    [require(cooperative_vector)]
    void store(RWByteAddressBuffer buffer, int32_t byteOffset16ByteAligned = 0)
    {
        __target_switch
        {
        case spirv:
            let ptr = buffer.GetBufferPointer();
            spirv_asm
            {
                // TODO: Should this be a byte offset
                OpCooperativeVectorStoreNV $ptr $byteOffset16ByteAligned $this None;
            };
#ifdef NOT_SUPPORTED_YET
        case hlsl:
            this.__Store(buffer, byteOffset16ByteAligned);
#endif
        default:
            for(int i = 0; i < N; ++i)
                buffer.StoreByteOffset(byteOffset16ByteAligned + __elemToByteOffset<T>(i), this[i]);
        }
    }

    [ForceInline]
    [require(cooperative_vector)]
    void store(RWStructuredBuffer<T> buffer, int32_t byteOffset16ByteAligned = 0)
    {
        __target_switch
        {
        case spirv:
            let ptr = __getStructuredBufferPtr(buffer);
            spirv_asm
            {
                // TODO: Should this be a byte offset
                OpCooperativeVectorStoreNV $ptr $byteOffset16ByteAligned $this None;
            };
#ifdef NOT_SUPPORTED_YET
        case hlsl:
            this.__Store(buffer, byteOffset16ByteAligned);
#endif
        default:
            for(int i = 0; i < N; ++i)
                buffer[i + __byteToElemOffset<T>(byteOffset16ByteAligned)] = this[i];
        }
    }

    [ForceInline]
    [require(cooperative_vector)]
    void store<let M : int>(__ref groupshared T[M] data, int32_t byteOffset16ByteAligned = 0)
    {
        __target_switch
        {
        case spirv:
            spirv_asm{
                OpCooperativeVectorStoreNV &data $byteOffset16ByteAligned $this None;
            };
        case hlsl:
            this.__Store(data, __byteToElemOffset<T>(byteOffset16ByteAligned));
        default:
            for(int i = 0; i < N; ++i)
                data[i + __byteToElemOffset<T>(byteOffset16ByteAligned)] = this[i];
        }
    }

    /// Store the value to a groupshared array of any type. This method is only available when targeting SPIR-V.
    /// @param data The destination array where the data will be stored. The array element type can be different from the CoopVec element type.
    /// @param byteOffset16ByteAligned The byte offset from the start of `data`. Must be a multiple of 16 bytes.
    [ForceInline]
    [require(spirv, cooperative_vector)]
    void storeAny<U, let M : int>(__ref groupshared U[M] data, int32_t byteOffset16ByteAligned = 0)
    {
        __target_switch
        {
        case spirv:
            spirv_asm{
                OpCooperativeVectorStoreNV &data $byteOffset16ByteAligned $this None;
            };
        }
    }

    [ForceInline]
    [require(spirv, cooperative_vector)]
    void storeAny<U, let M : int, let L : int>(__ref groupshared vector<U, L>[M] data, int32_t byteOffset16ByteAligned = 0)
    {
        __target_switch
        {
        case spirv:
            spirv_asm{
                OpCooperativeVectorStoreNV &data $byteOffset16ByteAligned $this None;
            };
        }
    }

    /// Load values from a byte-addressable buffer into a cooperative vector.
    /// @param buffer The source buffer to load data from.
    /// @param byteOffset16ByteAligned The byte offset from the start of the buffer. Must be 16-byte aligned.
    /// @return A new cooperative vector containing the loaded values.
    [ForceInline]
    [__NoSideEffect]
    [require(cooperative_vector)]
    static CoopVec<T, N> load(ByteAddressBuffer buffer, int32_t byteOffset16ByteAligned = 0)
    {
        __target_switch
        {
        case spirv:
            let ptr = buffer.GetBufferPointer();
            return spirv_asm
            {
                result:$$CoopVec<T, N> = OpCooperativeVectorLoadNV $ptr $byteOffset16ByteAligned None;
            };
        case hlsl:
            CoopVec<T, N> ret;
            ret.__Load(buffer, byteOffset16ByteAligned);
            return ret;
        default:
            var vec = CoopVec<T, N>();
            for(int i = 0; i < N; ++i)
                vec[i] = buffer.LoadByteOffset<T>(byteOffset16ByteAligned + __elemToByteOffset<T>(i));
            return vec;
        }
        return CoopVec<T, N>();
    }

    [ForceInline]
    [__NoSideEffect]
    [require(cooperative_vector)]
    static CoopVec<T, N> load(RWByteAddressBuffer buffer, int32_t byteOffset16ByteAligned = 0)
    {
        __target_switch
        {
        case spirv:
            let ptr = buffer.GetBufferPointer();
            return spirv_asm
            {
                result:$$CoopVec<T, N> = OpCooperativeVectorLoadNV $ptr $byteOffset16ByteAligned None;
            };
        case hlsl:
            CoopVec<T, N> ret;
            ret.__Load(buffer, byteOffset16ByteAligned);
            return ret;
        default:
            var vec = CoopVec<T, N>();
            for(int i = 0; i < N; ++i)
                vec[i] = buffer.LoadByteOffset<T>(byteOffset16ByteAligned + __elemToByteOffset<T>(i));
            return vec;
        }
        return CoopVec<T, N>();
    }

    [ForceInline]
    [__NoSideEffect]
    [require(cooperative_vector)]
    static CoopVec<T, N> load(StructuredBuffer<T> buffer, int32_t byteOffset16ByteAligned = 0)
    {
        __target_switch
        {
        case spirv:
            let ptr = __getStructuredBufferPtr(buffer);
            return spirv_asm
            {
                result:$$CoopVec<T, N> = OpCooperativeVectorLoadNV $ptr $byteOffset16ByteAligned None;
            };
#ifdef NOT_SUPPORTED_YET
        case hlsl:
            CoopVec<T, N> ret;
            ret.__Load(buffer, byteOffset16ByteAligned);
            return ret;
#endif
        default:
            var vec = CoopVec<T, N>();
            for(int i = 0; i < N; ++i)
                vec[i] = buffer[__byteToElemOffset<T>(byteOffset16ByteAligned) + i];
            return vec;
        }
        return CoopVec<T, N>();
    }

    [ForceInline]
    [__NoSideEffect]
    [require(spirv, cooperative_vector)]
    static CoopVec<T, N> load(RWStructuredBuffer<T> buffer, int32_t byteOffset16ByteAligned = 0)
    {
        __target_switch
        {
        case spirv:
            let ptr = __getStructuredBufferPtr(buffer);
            return spirv_asm
            {
                result:$$CoopVec<T, N> = OpCooperativeVectorLoadNV $ptr $byteOffset16ByteAligned None;
            };
#ifdef NOT_SUPPORTED_YET
        case hlsl:
            CoopVec<T, N> ret;
            ret.__Load(buffer, byteOffset16ByteAligned);
            return ret;
#endif
        default:
            var vec = CoopVec<T, N>();
            for(int i = 0; i < N; ++i)
                vec[i] = buffer[__byteToElemOffset<T>(byteOffset16ByteAligned) + i];
            return vec;
        }
    }

    // Groupshared
    [ForceInline]
    [__NoSideEffect]
    [require(cooperative_vector)]
    static CoopVec<T, N> load<let M : int>(__constref groupshared const T[M] data, int32_t byteOffset16ByteAligned = 0)
    {
        __target_switch
        {
        case spirv:
            return spirv_asm{
                result:$$CoopVec<T, N> = OpCooperativeVectorLoadNV &data $byteOffset16ByteAligned None
            };
        case hlsl:
            CoopVec<T, N> ret;
            ret.__Load(data, __byteToElemOffset<T>(byteOffset16ByteAligned));
            return ret;
        default:
            CoopVec<T,N> result;
            for(int i = 0; i < N; ++i)
                result[i] = data[i + __byteToElemOffset<T>(byteOffset16ByteAligned)];
            return result;
        }
    }

    /// Load values from a groupshared array into a CoopVec, allowing type conversion between source and destination elements.
    /// This operation is only available when targeting SPIR-V.
    /// @param data The source groupshared array to load from. The element type U can be different from the CoopVec element type T.
    /// @param byteOffset16ByteAligned The byte offset from the start of the array. Must be 16-byte aligned.
    /// @return A new CoopVec containing the loaded and type-converted values.
    [ForceInline]
    [__NoSideEffect]
    [require(spirv, cooperative_vector)]
    static CoopVec<T, N> loadAny<U : __BuiltinArithmeticType, let M : int>(__constref groupshared const U[M] data, int32_t byteOffset16ByteAligned = 0)
    {
        __target_switch
        {
        case spirv:
            return spirv_asm{
                result:$$CoopVec<T, N> = OpCooperativeVectorLoadNV &data $byteOffset16ByteAligned None
            };
        }
    }

    [ForceInline]
    [__NoSideEffect]
    [require(spirv, cooperative_vector)]
    static CoopVec<T, N> loadAny<U : __BuiltinArithmeticType, let M : int, let L : int>(__constref groupshared const vector<U, L>[M] data, int32_t byteOffset16ByteAligned = 0)
    {
        __target_switch
        {
        case spirv:
            return spirv_asm{
                result:$$CoopVec<T, N> = OpCooperativeVectorLoadNV &data $byteOffset16ByteAligned None
            };
        }
    }

    //
    // Subscript
    //

    __intrinsic_op($(kIROp_GetElement))
    [__NoSideEffect]
    T __indexRead(int index);

    __intrinsic_op($(kIROp_GetElementPtr))
    [__ref]
    [__NoSideEffect]
    Ref<T> __indexRef(int index);

    [ForceInline]
    [__NoSideEffect]
    int getCount()
    {
        return N;
    }

    /// Access an individual element in the Cooperative vector by index.
    __subscript(int index) -> T
    {
        [ForceInline]
        [__NoSideEffect]
        [nonmutating]
        get
        {
            __target_switch
            {
            case hlsl: __intrinsic_asm ".ReadFromIndex";
            default: return __indexRead(index);
            }
        }

        [ForceInline]
        [mutating]
        set
        {
            __target_switch
            {
            case hlsl: __intrinsic_asm ".WriteToIndex";
            default: __indexRef(index) = newValue;
            }
        }

        // Unavailable on HLSL
        // The CoopVector HLSL spec says that indexing with a subscript
        // operation can work, but dxc currently crashes with this
        // __intrinsic_op($(kIROp_GetElementPtr))
        // [__ref]
        // ref;
    }

    /// Creates a new cooperative vector with all elements initialized to the specified scalar value.
    /// @param t The scalar value to replicate across all elements.
    /// @return A new cooperative vector where each element equals the input value.
    static CoopVec<T, N> replicate(T t)
    {
        CoopVec<T, N> ret;
        ret.fill(t);
        return ret;
    }

    //
    // IComparable
    //

    /// Checks if this cooperative vector is equal to another cooperative vector by comparing all elements.
    /// @param other The cooperative vector to compare against.
    /// @return True if all corresponding elements are equal, false otherwise.
    bool equals(This other)
    {
        for (int i = 0; i < N; i++)
        {
            if (this[i] != other[i])
            {
                return false;
            }
        }
        return true;
    }

    /// Compares two cooperative vectors lexicographically.
    /// @param other The cooperative vector to compare against.
    /// @return True if this vector is lexicographically less than the other vector.
    /// @remarks This function exists only to conform to IComparable. For cooperative vectors,
    /// lexicographical comparison has limited practical use since the vectors are meant for
    /// parallel computation rather than ordering.
    bool lessThan(This other)
    {
        for (int i = 0; i < N; i++)
        {
            if (this[i] < other[i])
            {
                return true;
            }
            else if (this[i] > other[i])
            {
                return false;
            }
        }
        return false;
    }

    /// Compares two cooperative vectors lexicographically.
    /// @param other The cooperative vector to compare against.
    /// @return True if this vector is lexicographically less than or equal to the other vector.
    /// @remarks This function exists only to conform to IComparable. For cooperative vectors,
    /// lexicographical comparison has limited practical use since the vectors are meant for
    /// parallel computation rather than ordering.
    bool lessThanOrEquals(This other)
    {
        for (int i = 0; i < N; i++)
        {
            if (this[i] < other[i])
            {
                return true;
            }
            else if (this[i] > other[i])
            {
                return false;
            }
        }
        return true;
    }

    //
    // Arithmetic
    //

    __intrinsic_op($(kIROp_Add))
    This __pureAdd(This other);

    [mutating]
    [require(hlsl)]
    void __mutAdd(This other)
    {
        __target_switch 
        { 
        case hlsl: __intrinsic_asm ".Add"; 
        } 
    }

    /// Performs component-wise addition with another cooperative vector.
    /// @param other The cooperative vector to add to this vector.
    /// @return A new cooperative vector containing the sum of the two vectors.
    // TODO: Why is this ForceInline necessary for hlsl, dxc bug?
    [ForceInline]
    This add(This other)
    {
        __target_switch
        {
        case hlsl:
            This ret = this;
            ret.__mutAdd(other);
            return ret;
        default: return __pureAdd(other);
        }
    }

    __intrinsic_op($(kIROp_Sub))
    This __pureSub(This other);

    [mutating]
    [require(hlsl)]
    void __mutSub(This other)
    { __target_switch { case hlsl: __intrinsic_asm ".Subtract"; } }

    /// Performs component-wise subtraction with another cooperative vector.
    /// @param other The cooperative vector to subtract from this vector.
    /// @return A new cooperative vector containing the difference of the two vectors.
    [ForceInline]
    This sub(This other)
    {
        __target_switch
        {
        case hlsl:
            This ret = this;
            ret.__mutSub(other);
            return ret;
        default: return __pureSub(other);
        }
    }

    __intrinsic_op($(kIROp_Mul))
    This __pureMul(This other);

    [mutating]
    [require(hlsl)]
    void __mutMul(This other)
    { __target_switch { case hlsl: __intrinsic_asm ".Multiply"; } }

    /// Performs component-wise multiplication with another cooperative vector.
    /// @param other The cooperative vector to multiply with this vector.
    /// @return A new cooperative vector containing the product of the two vectors.
    [ForceInline]
    This mul(This other)
    {
        __target_switch
        {
        case hlsl:
            This ret = this;
            ret.__mutMul(other);
            return ret;
        default: return __pureMul(other);
        }
    }

    __intrinsic_op($(kIROp_Div))
    This __pureDiv(This other);

    [mutating]
    [require(hlsl)]
    void __mutDiv(This other)
    { __target_switch { case hlsl: __intrinsic_asm ".Divide"; } }

    /// Performs component-wise division with another cooperative vector.
    /// @param other The cooperative vector to divide this vector by.
    /// @return A new cooperative vector containing the quotient of the two vectors.
    [ForceInline]
    This div(This other)
    {
        __target_switch
        {
        case hlsl:
            This ret = this;
            ret.__mutDiv(other);
            return ret;
        default: return __pureDiv(other);
        }
    }

    [mutating]
    [require(hlsl)]
    void __mutMod(This other)
    { __target_switch { case hlsl: __intrinsic_asm ".Mod"; } }

    /// Performs component-wise remainder operation between two cooperative vectors.
    /// @param other The cooperative vector to compute the remainder with.
    /// @return A new cooperative vector containing the remainder of the division between corresponding components.
    [ForceInline]
    This mod(This other)
    {
        __target_switch
        {
        case hlsl:
            This ret = this;
            ret.__mutMod(other);
            return ret;
        default:
            This ret;
            for(int i = 0; i < N; ++i)
                ret[i] = this[i] % other[i];
            return ret;
        }
    }

    __intrinsic_op($(kIROp_Neg))
    This __pureNeg(This other);

    /// Returns a new cooperative vector where each component has its sign negated.
    /// @return A new cooperative vector containing the negated values.
    //[ForceInline]
    This neg()
    {
        __target_switch
        {
        case hlsl:
            This ret = this;
            for(int i = 0; i < N; ++i)
                ret[i] = -this[i];
            return ret;
        default: return __pureNeg(this);
        }
    }

    [mutating]
    [require(hlsl)]
    void __mutScalarMul(T t)
    { __target_switch { case hlsl: __intrinsic_asm ".ScalarMultiply"; } }

    [mutating]
    [require(hlsl)]
    void __mutMin(This other)
    { __target_switch { case hlsl: __intrinsic_asm ".Min"; } }

    [mutating]
    [require(hlsl)]
    void __mutMax(This other)
    { __target_switch { case hlsl: __intrinsic_asm ".Max"; } }

    [mutating]
    [require(hlsl)]
    void __mutClamp(This minVal, This maxVal)
    { __target_switch { case hlsl: __intrinsic_asm ".Clamp"; } }

    //
    // Internal utilities for loading and storing
    //

    [mutating]
    [require(hlsl, byteaddressbuffer)]
    void __Load(const ByteAddressBuffer buffer, uint byteOffset, uint alignment = 0)
    { __target_switch { case hlsl: __intrinsic_asm ".Load"; } }

    [mutating]
    [require(hlsl, byteaddressbuffer_rw)]
    void __Load(const RWByteAddressBuffer buffer, uint byteOffset, uint alignment = 0)
    { __target_switch { case hlsl: __intrinsic_asm ".Load"; } }

    __generic<let M : int>
    [mutating]
    // Careful, this takes the offset in elements
    [require(hlsl)]
    void __Load(__constref groupshared T buffer[M], uint elemOffset)
    { __target_switch { case hlsl: __intrinsic_asm ".Load"; } }

    [require(hlsl, byteaddressbuffer_rw)]
    void __Store(RWByteAddressBuffer buffer, uint byteOffset, uint alignment = 0)
    { __target_switch { case hlsl: __intrinsic_asm ".Store"; } }

    __generic<let M : int>
    [require(hlsl)]
    // Careful, this takes the offset in elements
    void __Store(__ref groupshared T buffer[M], uint elemOffset)
    { __target_switch { case hlsl: __intrinsic_asm ".Store"; } }

${{{{
static const struct {
    bool isRW;
    char const* type;
} kByteAddressBufferCases[] =
{
    {true, "RWByteAddressBuffer"},
    {false, "ByteAddressBuffer"}
};
for(auto buffer : kByteAddressBufferCases) {
}}}}
    [mutating]
    [require(hlsl, byteaddressbuffer_rw)]
    void __mutMatMul<U : __BuiltinArithmeticType, let K : int>(
        CoopVec<U, K> input, uint inputInterpretationHLSL,
        $(buffer.type) matrix, uint matrixOffset, uint matrixInterpretationHLSL,
        uint m, uint k, uint memoryLayoutHLSL, bool transpose, uint matrixStride)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".MatMul";
        }
    }

    [mutating]
    [require(hlsl, byteaddressbuffer_rw)]
    void __mutMatMulAdd<U : __BuiltinArithmeticType, let K : int>(
        CoopVec<U, K> input, uint inputInterpretationHLSL,
        $(buffer.type) matrix, uint matrixOffset, uint matrixInterpretationHLSL,
        $(buffer.type) bias, uint biasOffset, uint biasInterpretationHLSL,
        uint m, uint k, uint memoryLayoutHLSL, bool transpose, uint matrixStride)
    {
        __target_switch
        {
        case hlsl: __intrinsic_asm ".MatMulAdd";
        }
    }

    /// Multiply the given input Cooperative vector with the given matrix and accumulate the result into this vector.
    /// @param input The input Cooperative vector to multiply with the matrix.
    /// @param inputInterpretation Specifies how to interpret the values in the input vector (e.g. as packed values).
    /// @param k The number of columns in the matrix.
    /// @param matrix The matrix buffer to multiply with.
    /// @param matrixOffset Byte offset into the matrix buffer.
    /// @param matrixInterpretation Specifies how to interpret the values in the matrix.
    /// @param memoryLayout Specifies the memory layout of the matrix (row-major or column-major).
    /// @param transpose Whether to transpose the matrix before multiplication.
    /// @param matrixStride The stride between matrix rows/columns in bytes.
    /// @remarks Unlike matMulAccum, this function supports packed input interpretations where multiple values
    /// can be packed into each element of the input vector. The k parameter specifies the actual number of
    /// values to use from the packed input.
    [mutating]
    [ForceInline]
    void matMulAccumPacked<U : __BuiltinArithmeticType, let PackedK : int>(
        CoopVec<U, PackedK> input,
        constexpr CoopVecComponentType inputInterpretation,
        constexpr int k,
        $(buffer.type) matrix,
        int32_t matrixOffset,
        constexpr CoopVecComponentType matrixInterpretation,
        constexpr CoopVecMatrixLayout memoryLayout,
        constexpr bool transpose,
        constexpr uint matrixStride
    )
    {
        static_assert(__isPackedInputInterpretation(inputInterpretation) || k == PackedK
                    , "for non-packed inputInterpretation values k must be equal to the input vector length");
        static_assert(!__isPackedInputInterpretation(inputInterpretation)
                    || k <= __inputInterpretationPackingFactor(inputInterpretation)*PackedK
                    , "for packed inputInterpretation values k must be less than or equal to the input vector length times the packing factor");

        __target_switch
        {
        case hlsl:
            let inputInterpretationHLSL = __getHLSLCoopVecComponentType(inputInterpretation);
            let matrixInterpretationHLSL = __getHLSLCoopVecComponentType(matrixInterpretation);
            let memoryLayoutHLSL = __getHLSLCoopVecMatrixLayout(memoryLayout);
            This temp = this;
            temp.__mutMatMul(
                input,
                inputInterpretationHLSL,
                matrix,
                matrixOffset,
                matrixInterpretationHLSL,
                N,
                k,
                memoryLayoutHLSL,
                transpose,
                matrixStride
            );
            this.__mutAdd(temp);
        default: this = this + coopVecMatMulPacked<T, N, PackedK, U>(
                input,
                inputInterpretation,
                k,
                matrix,
                matrixOffset,
                matrixInterpretation,
                memoryLayout,
                transpose,
                matrixStride
            );
        }
    }

    /// Accumulate the result from a matrix multiplication between an input Cooperative vector and a matrix.
    /// @param input The input Cooperative vector to multiply with the matrix.
    /// @param inputInterpretation Specifies how to interpret the values in the input vector (e.g. as 8-bit integers, 16-bit floats, etc).
    /// @param matrix The matrix to multiply with the input vector.
    /// @param matrixOffset Byte offset into the matrix buffer.
    /// @param matrixInterpretation Specifies how to interpret the values in the matrix (e.g. as 8-bit integers, 16-bit floats, etc).
    /// @param memoryLayout Specifies the memory layout of the matrix (row-major or column-major).
    /// @param transpose Whether to transpose the matrix before multiplication.
    /// @param matrixStride The stride in bytes between rows/columns of the matrix.
    [mutating]
    [ForceInline]
    void matMulAccum<U : __BuiltinArithmeticType, let K : int>(
        CoopVec<U, K> input,
        constexpr CoopVecComponentType inputInterpretation,
        $(buffer.type) matrix,
        int32_t matrixOffset,
        constexpr CoopVecComponentType matrixInterpretation,
        constexpr CoopVecMatrixLayout memoryLayout,
        constexpr bool transpose,
        constexpr uint matrixStride
    )
    {
        static_assert(!__isPackedInputInterpretation(inputInterpretation)
                    , "for packed inputInterpretation values please use coopVecMatMulPacked and specify k manually");
        this.matMulAccumPacked<U, K>(
            input,
            inputInterpretation,
            K,
            matrix,
            matrixOffset,
            matrixInterpretation,
            memoryLayout,
            transpose,
            matrixStride 
        );
    }

    [mutating]
    [ForceInline]
    void matMulAddAccumPacked<U : __BuiltinArithmeticType, let PackedK : int>(
        CoopVec<U, PackedK> input,
        constexpr CoopVecComponentType inputInterpretation,
        constexpr int k,
        $(buffer.type) matrix,
        int32_t matrixOffset,
        constexpr CoopVecComponentType matrixInterpretation,
        $(buffer.type) bias,
        int32_t biasOffset,
        constexpr CoopVecComponentType biasInterpretation,
        constexpr CoopVecMatrixLayout memoryLayout,
        constexpr bool transpose,
        constexpr uint matrixStride
    )
    {
        static_assert(__isPackedInputInterpretation(inputInterpretation) || k == PackedK
                    , "for non-packed inputInterpretation values k must be equal to the input vector length");
        static_assert(!__isPackedInputInterpretation(inputInterpretation)
                    || k <= __inputInterpretationPackingFactor(inputInterpretation)*PackedK
                    , "for packed inputInterpretation values k must be less than or equal to the input vector length times the packing factor");

        __target_switch
        {
        case hlsl:
            let inputInterpretationHLSL = __getHLSLCoopVecComponentType(inputInterpretation);
            let matrixInterpretationHLSL = __getHLSLCoopVecComponentType(matrixInterpretation);
            let biasInterpretationHLSL = __getHLSLCoopVecComponentType(biasInterpretation);
            let memoryLayoutHLSL = __getHLSLCoopVecMatrixLayout(memoryLayout);
            This temp = this;
            temp.__mutMatMulAdd(
                input,
                inputInterpretationHLSL,
                matrix,
                matrixOffset,
                matrixInterpretationHLSL,
                bias,
                biasOffset,
                biasInterpretationHLSL,
                N,
                k,
                memoryLayoutHLSL,
                transpose,
                matrixStride
            );
            this.__mutAdd(temp);
        default: this = this + coopVecMatMulAddPacked<T, N, PackedK, U>(
            input,
            inputInterpretation,
            k,
            matrix,
            matrixOffset,
            matrixInterpretation,
            bias,
            biasOffset,
            biasInterpretation,
            memoryLayout,
            transpose,
            matrixStride 
            );
        }
    }

    /// Performs matrix multiplication and accumulation with bias: this += input * matrix + bias
    /// @param input The input vector to multiply with the matrix
    /// @param inputInterpretation How to interpret the input vector elements (must not be packed)
    /// @param matrix The matrix buffer to multiply with
    /// @param matrixOffset Byte offset into the matrix buffer
    /// @param matrixInterpretation How to interpret the matrix elements
    /// @param bias The bias buffer to add
    /// @param biasOffset Byte offset into the bias buffer
    /// @param biasInterpretation How to interpret the bias elements
    /// @param memoryLayout Memory layout of the matrix (row or column major)
    /// @param transpose Whether to transpose the matrix before multiplication
    /// @param matrixStride Stride between matrix rows/columns in bytes
    /// @remark The key difference from matMulAddAccumPacked is that this method enforces k must equal the input vector length,
    /// while matMulAddAccumPacked allows k to be specified independently for packed interpretations.
    [mutating]
    [ForceInline]
    void matMulAddAccum<U : __BuiltinArithmeticType, let K : int>(
        CoopVec<U, K> input,
        constexpr CoopVecComponentType inputInterpretation,
        $(buffer.type) matrix,
        int32_t matrixOffset,
        constexpr CoopVecComponentType matrixInterpretation,
        $(buffer.type) bias,
        int32_t biasOffset,
        constexpr CoopVecComponentType biasInterpretation,
        constexpr CoopVecMatrixLayout memoryLayout,
        constexpr bool transpose,
        constexpr uint matrixStride
    )
    {
        static_assert(!__isPackedInputInterpretation(inputInterpretation)
                    , "for packed inputInterpretation values please use coopVecMatMulPacked and specify k manually");
        this.matMulAddAccumPacked<U, K>(
            input,
            inputInterpretation,
            K,
            matrix,
            matrixOffset,
            matrixInterpretation,
            bias,
            biasOffset,
            biasInterpretation,
            memoryLayout,
            transpose,
            matrixStride
        );
    }


${{{{
}
}}}}
}

__intrinsic_op($(kIROp_MakeCoopVectorFromValuePack))
CoopVec<T, N> __makeCoopVec<T : __BuiltinArithmeticType, let N : int, each U>(expand each U args);

__generic<T:__BuiltinArithmeticType, U:__BuiltinArithmeticType, let N : int>
__intrinsic_op($(kIROp_IntCast))
[require(cooperative_vector)]
CoopVec<T,N> __int_cast(CoopVec<U,N> val);

__generic<T:__BuiltinArithmeticType, U:__BuiltinArithmeticType, let N : int>
__intrinsic_op($(kIROp_FloatCast))
[require(cooperative_vector)]
CoopVec<T,N> __real_cast(CoopVec<U,N> val);

__generic<T:__BuiltinArithmeticType, U:__BuiltinArithmeticType, let N : int>
__intrinsic_op($(kIROp_CastIntToFloat))
[require(cooperative_vector)]
CoopVec<T,N> __int_to_float_cast(CoopVec<U,N> val);

__generic<T:__BuiltinArithmeticType, U:__BuiltinArithmeticType, let N : int>
__intrinsic_op($(kIROp_CastFloatToInt))
[require(cooperative_vector)]
CoopVec<T,N> __float_to_int_cast(CoopVec<U,N> val);

__generic<T : __BuiltinArithmeticType, let N : int>
[ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> operator *(CoopVec<T, N> lhs, const T rhs)
{
    __target_switch
    {
    case spirv:
        if (__isFloat<T>())
        {
            return spirv_asm
            {
                result:$$CoopVec<T, N> = OpVectorTimesScalar $lhs $rhs;
            };
        }
        else
        {
            for (int i = 0; i < N; ++i)
            {
                lhs[i] *= rhs;
            }
            return lhs;
        }
    case hlsl:
        CoopVec<T, N> ret = lhs;
        ret.__mutScalarMul(rhs);
        return ret;
    default:
        for (int i = 0; i < N; ++i)
        {
            lhs[i] *= rhs;
        }
        return lhs;
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
[ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> operator *(const T lhs, CoopVec<T, N> rhs)
{
    return rhs * lhs;
}

[ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> min<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> x, CoopVec<T, N> y)
{
    __target_switch
    {
    case spirv:
        return spirv_asm
        {
            result:$$CoopVec<T, N> = OpExtInst glsl450 FMin $x $y;
        };
    case hlsl:
        CoopVec<T, N> ret = x;
        ret.__mutMin(y);
        return ret;
    default:
        CoopVec<T, N> ret;
        for(int i = 0; i < N; ++i)
            ret[i] = min(x[i], y[i]);

        return ret;
    }
}

[ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> max<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> x, CoopVec<T, N> y)
{
    __target_switch
    {
    case spirv:
        return spirv_asm
        {
            result:$$CoopVec<T, N> = OpExtInst glsl450 FMax $x $y;
        };
    case hlsl:
        CoopVec<T, N> ret = x;
        ret.__mutMax(y);
        return ret;
    default:
        CoopVec<T, N> ret;
        for(int i = 0; i < N; ++i)
            ret[i] = max(x[i], y[i]);
        return ret;
    }
}

[ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> clamp<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> x, CoopVec<T, N> minVal, CoopVec<T, N> maxVal)
{
    __target_switch
    {
    case spirv:
        return spirv_asm
        {
            result:$$CoopVec<T, N> = OpExtInst glsl450 FClamp $x $minVal $maxVal;
        };
    case hlsl:
        CoopVec<T, N> ret = x;
        ret.__mutClamp(minVal, maxVal);
        return ret;
    default:
        CoopVec<T, N> ret;
        for(int i = 0; i < N; ++i)
            ret[i] = clamp(x[i], minVal[i], maxVal[i]);
        return ret;
    }
}

[ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> min<T : __BuiltinIntegerType, let N : int>(CoopVec<T, N> x, CoopVec<T, N> y)
{
    __target_switch
    {
    case spirv:
        if (__isSignedInt<T>())
        {
            return spirv_asm
            {
                result:$$CoopVec<T, N> = OpExtInst glsl450 SMin $x $y
            };
        }
        else
        {
            return spirv_asm
            {
                result:$$CoopVec<T, N> = OpExtInst glsl450 UMin $x $y
            };
        }
    case hlsl:
        CoopVec<T, N> ret = x;
        ret.__mutMin(y);
        return ret;
    default:
        CoopVec<T, N> ret;
        for(int i = 0; i < N; ++i)
            ret[i] = min(x[i], y[i]);

        return ret;
    }
}

// [ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> max<T : __BuiltinIntegerType, let N : int>(CoopVec<T, N> x, CoopVec<T, N> y)
{
    __target_switch
    {
    case spirv:
        if (__isSignedInt<T>())
        {
            return spirv_asm
            {
                result:$$CoopVec<T, N> = OpExtInst glsl450 SMax $x $y
            };
        }
        else
        {
            return spirv_asm
            {
                result:$$CoopVec<T, N> = OpExtInst glsl450 UMax $x $y
            };
        }
    case hlsl:
        CoopVec<T, N> ret = x;
        ret.__mutMax(y);
        return ret;
    default:
        CoopVec<T, N> ret;
        for(int i = 0; i < N; ++i)
            ret[i] = max(x[i], y[i]);
        return ret;
    }
}

// [ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> clamp<T : __BuiltinIntegerType, let N : int>(CoopVec<T, N> x, CoopVec<T, N> minVal, CoopVec<T, N> maxVal)
{
    __target_switch
    {
    case spirv:
        if (__isSignedInt<T>())
        {
            return spirv_asm
            {
                result:$$CoopVec<T, N> = OpExtInst glsl450 SClamp $x $minVal $maxVal
            };
        }
        else
        {
            return spirv_asm
            {
                result:$$CoopVec<T, N> = OpExtInst glsl450 UClamp $x $minVal $maxVal
            };
        }
    case hlsl:
        CoopVec<T, N> ret = x;
        ret.__mutClamp(minVal, maxVal);
        return ret;
    default:
        CoopVec<T, N> ret;
        for(int i = 0; i < N; ++i)
            ret[i] = clamp(x[i], minVal[i], maxVal[i]);
        return ret;
    }
}

// [ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> step<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> edge, CoopVec<T, N> x)
{
    __target_switch
    {
    case spirv:
        return spirv_asm
        {
            result:$$CoopVec<T, N> = OpExtInst glsl450 Step $edge $x;
        };
    default:
        CoopVec<T, N> ret;
        for(int i = 0; i < N; ++i)
            ret[i] = step(edge[i], x[i]);
        return ret;
    }
}

// [ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> exp<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> x)
{
    __target_switch
    {
    case spirv:
        return spirv_asm
        {
            result:$$CoopVec<T, N> = OpExtInst glsl450 Exp $x;
        };
    default:
        CoopVec<T, N> ret;
        for(int i = 0; i < N; ++i)
            ret[i] = exp(x[i]);
        return ret;
    }
}

// [ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> log<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> x)
{
    __target_switch
    {
    case spirv:
        return spirv_asm
        {
            result:$$CoopVec<T, N> = OpExtInst glsl450 Log $x;
        };
    default:
        CoopVec<T, N> ret;
        for(int i = 0; i < N; ++i)
            ret[i] = log(x[i]);
        return ret;
    }
}

// [ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> tanh<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> x)
{
    __target_switch
    {
    case spirv:
        return spirv_asm
        {
            result:$$CoopVec<T, N> = OpExtInst glsl450 Tanh $x;
        };
    default:
        CoopVec<T, N> ret;
        for(int i = 0; i < N; ++i)
            ret[i] = tanh(x[i]);
        return ret;
    }
}

// TODO: Why does this fail when inlined on HLSL,
// We generate some really weird code...
// [ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> atan<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> yOverX)
{
    __target_switch
    {
    case spirv:
        return spirv_asm
        {
            result:$$CoopVec<T, N> = OpExtInst glsl450 Atan $yOverX;
        };
    default:
        CoopVec<T, N> ret;
        for(int i = 0; i < N; ++i)
            ret[i] = atan(yOverX[i]);
        return ret;
    }
}

// [ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> fma<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> a, CoopVec<T, N> b, CoopVec<T, N> c)
{
    // TODO: Investigate, why does this fail if it's not inlined
    // replacing fma with mad below also fixes things...
    // dxc generated substantially different code
    __target_switch
    {
    case spirv:
        return spirv_asm
        {
            result:$$CoopVec<T, N> = OpExtInst glsl450 Fma $a $b $c;
        };
    default:
        CoopVec<T, N> ret;
        for(int i = 0; i < N; ++i)
            ret[i] = mad(a[i], b[i], c[i]);
        return ret;
    }
}

// Buffers from which values of arbitrary type can be loaded from byte offsets
interface IPhysicalBuffer
{
    [__unsafeForceInlineEarly]
    T LoadByteOffset<T>(int offset);

    [__unsafeForceInlineEarly]
    Ptr<uint32_t[]> GetBufferPointer();
}

// Buffers to which values of arbitrary type can be stored at byte offsets
interface IRWPhysicalBuffer : IPhysicalBuffer
{
    [__unsafeForceInlineEarly]
    void StoreByteOffset<T>(int offset, T element);
}

extension ByteAddressBuffer : IPhysicalBuffer
{
    [__unsafeForceInlineEarly]
    Ptr<uint32_t[]> GetBufferPointer()
    {
        return __getStructuredBufferPtr(__getEquivalentStructuredBuffer<uint32_t>(this));
    }

    [__unsafeForceInlineEarly]
    T LoadByteOffset<T>(int offset)
    {
        return this.Load<T>(offset);
    }
}

extension RWByteAddressBuffer : IPhysicalBuffer
{
    [__unsafeForceInlineEarly]
    Ptr<uint32_t[]> GetBufferPointer()
    {
        return __getStructuredBufferPtr(__getEquivalentStructuredBuffer<uint32_t>(this));
    }

    [__unsafeForceInlineEarly]
    T LoadByteOffset<T>(int offset)
    {
        return this.Load<T>(offset);
    }
}

extension RWByteAddressBuffer : IRWPhysicalBuffer
{
    [__unsafeForceInlineEarly]
    void StoreByteOffset<T>(int offset, T element)
    {
        return this.Store<T>(offset, element);
    }
}


//
// Convenience loading functions for cooperative vectors which infer the
// element type for structured buffers and groupshared arrays (and ByteAddressBuffers for consistency
//

/// Load values from a byte-addressable buffer into a cooperative vector.
/// @param buffer The source buffer to load data from.
/// @param byteOffset16ByteAligned The byte offset from the start of the buffer. Must be 16-byte aligned.
/// @return A new cooperative vector containing the loaded values.
[ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> coopVecLoad<let N : int, T : __BuiltinArithmeticType>(ByteAddressBuffer buffer, int32_t byteOffset16ByteAligned = 0)
{
    return CoopVec<T, N>.load(buffer, byteOffset16ByteAligned);
}

[ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> coopVecLoad<let N : int, T : __BuiltinArithmeticType>(RWByteAddressBuffer buffer, int32_t byteOffset16ByteAligned = 0)
{
    return CoopVec<T, N>.load(buffer, byteOffset16ByteAligned);
}

[ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> coopVecLoad<let N : int, T : __BuiltinArithmeticType>(StructuredBuffer<T> buffer, int32_t byteOffset16ByteAligned = 0)
{
    return CoopVec<T, N>.load(buffer, byteOffset16ByteAligned);
}

[ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> coopVecLoad<let N : int, T : __BuiltinArithmeticType>(RWStructuredBuffer<T> buffer, int32_t byteOffset16ByteAligned = 0)
{
    return CoopVec<T, N>.load(buffer, byteOffset16ByteAligned);
}

// Groupshared
[ForceInline]
[require(cooperative_vector)]
CoopVec<T, N> coopVecLoadGroupshared<let N : int, T : __BuiltinArithmeticType, let M : int>(__constref groupshared const T[M] data, int32_t byteOffset16ByteAligned = 0)
{
    return CoopVec<T, N>.load(data, byteOffset16ByteAligned);
}

//
// Coop Vector matrix multiplication
//


/// Specifies the memory layout for matrices used in cooperative vector operations.
/// @remarks This enum defines different matrix layout options that affect how matrix data is stored and accessed,
/// including standard row-major and column-major layouts as well as specialized layouts optimized for specific operations.
enum CoopVecMatrixLayout
{
    RowMajor,
    ColumnMajor,
    InferencingOptimal,
    TrainingOptimal
};

/// Specifies how to interpret the values in a cooperative vector or matrix.
/// @remarks This enum defines the various data types that can be used for elements in cooperative vectors and matrices,
/// including packed formats where multiple values can be stored in a single element.
enum CoopVecComponentType
{
    FloatE4M3,
    FloatE5M2,
    Float16,
    Float32,
    Float64,
    SignedInt8,
    SignedInt16,
    SignedInt32,
    SignedInt64,
    SignedInt8Packed,
    UnsignedInt8,
    UnsignedInt16,
    UnsignedInt32,
    UnsignedInt64,
    UnsignedInt8Packed
};

[ForceInline]
int __inputInterpretationPackingFactor(CoopVecComponentType componentType)
{
    switch (componentType)
    {
    case CoopVecComponentType::SignedInt8Packed:
    case CoopVecComponentType::UnsignedInt8Packed:
        return 4;
    }
    return 1;
}

[ForceInline]
bool __isPackedInputInterpretation(CoopVecComponentType componentType)
{
    return __inputInterpretationPackingFactor(componentType) != 1;
}

// TODO: We might consider some way of specifying these from our lookup tables
[ForceInline]
uint32_t __getSpvCoopVecMatrixLayout(CoopVecMatrixLayout layout)
{
    switch (layout)
    {
    case CoopVecMatrixLayout::RowMajor:
        return 0;
    case CoopVecMatrixLayout::ColumnMajor:
        return 1;
    case CoopVecMatrixLayout::InferencingOptimal:
        return 2;
    case CoopVecMatrixLayout::TrainingOptimal:
        return 3;
    default:
        static_assert(false, "unsupported layout value");
    }
    return 0xffffffff;
}

[ForceInline]
uint32_t __getHLSLCoopVecMatrixLayout(CoopVecMatrixLayout layout)
{
    switch (layout)
    {
    // TODO: Check these are the same
    case CoopVecMatrixLayout::RowMajor:
        return 0;
    case CoopVecMatrixLayout::ColumnMajor:
        return 1;
    case CoopVecMatrixLayout::InferencingOptimal:
        return 2;
    case CoopVecMatrixLayout::TrainingOptimal:
        return 3;
    default:
        static_assert(false, "unsupported layout value");
    }
    return 0xffffffff;
}

[ForceInline]
uint32_t __getSpvCoopVecComponentType(CoopVecComponentType componentType)
{
    switch (componentType)
    {
    case CoopVecComponentType::Float16:
        return 0;
    case CoopVecComponentType::Float32:
        return 1;
    case CoopVecComponentType::Float64:
        return 2;
    case CoopVecComponentType::SignedInt8:
        return 3;
    case CoopVecComponentType::SignedInt16:
        return 4;
    case CoopVecComponentType::SignedInt32:
        return 5;
    case CoopVecComponentType::SignedInt8Packed:
        return 1000491000;
    case CoopVecComponentType::SignedInt64:
        return 6;
    case CoopVecComponentType::UnsignedInt8:
        return 7;
    case CoopVecComponentType::UnsignedInt16:
        return 8;
    case CoopVecComponentType::UnsignedInt32:
        return 9;
    case CoopVecComponentType::UnsignedInt8Packed:
        return 1000491001;
    case CoopVecComponentType::UnsignedInt64:
        return 10;
    case CoopVecComponentType::FloatE4M3:
        return 1000491002;
    case CoopVecComponentType::FloatE5M2:
        return 1000491003;
    default:
        static_assert(false, "unsupported componentType value");
    }
    return 0xffffffff;
}

[ForceInline]
uint32_t __getHLSLCoopVecComponentType(CoopVecComponentType componentType)
{
    switch (componentType)
    {
    case CoopVecComponentType::Float16:
        return 0;
    case CoopVecComponentType::Float32:
        return 1;
    case CoopVecComponentType::UnsignedInt8:
        return 2;
    case CoopVecComponentType::UnsignedInt16:
        return 3;
    case CoopVecComponentType::UnsignedInt32:
        return 4;
    case CoopVecComponentType::SignedInt8:
        return 5;
    case CoopVecComponentType::SignedInt16:
        return 6;
    case CoopVecComponentType::SignedInt32:
        return 7;
    case CoopVecComponentType::SignedInt8Packed:
        return 8;
    case CoopVecComponentType::UnsignedInt8Packed:
        return 9;
    case CoopVecComponentType::FloatE4M3:
        return 10;
    case CoopVecComponentType::FloatE5M2:
        return 11;
    default:
        static_assert(false, "unsupported componentType value");
    }
    return 32;
}

[ForceInline]
uint32_t __coopVecComponentTypeStride(CoopVecComponentType componentType)
{
    switch (componentType)
    {
    case CoopVecComponentType::Float16:
        return 2;
    case CoopVecComponentType::Float32:
        return 4;
    case CoopVecComponentType::Float64:
        return 8;
    case CoopVecComponentType::SignedInt8:
        return 1;
    case CoopVecComponentType::SignedInt16:
        return 2;
    case CoopVecComponentType::SignedInt32:
        return 4;
    case CoopVecComponentType::SignedInt8Packed:
        return 4;
    case CoopVecComponentType::SignedInt64:
        return 8;
    case CoopVecComponentType::UnsignedInt8:
        return 1;
    case CoopVecComponentType::UnsignedInt16:
        return 2;
    case CoopVecComponentType::UnsignedInt32:
        return 4;
    case CoopVecComponentType::UnsignedInt8Packed:
        return 4;
    case CoopVecComponentType::UnsignedInt64:
        return 8;
    default:
        static_assert(false, "unsupported componentType value");
    }
    return 0xffffffff;
}

${{{{
static const struct {
    bool isRW;
    char const* type;
} kByteAddressBufferCases_[] =
{
    {true, "RWByteAddressBuffer"},
    {false, "ByteAddressBuffer"},
};
for(auto buffer : kByteAddressBufferCases_) {
}}}}

/// Multiply a cooperative vector with a matrix and return the result.
/// @param input The input cooperative vector to multiply with the matrix.
/// @param inputInterpretation Specifies how to interpret the values in the input vector (e.g. as packed values).
/// @param k The number of columns in the matrix.
/// @param matrix The matrix buffer to multiply with.
/// @param matrixOffset Byte offset into the matrix buffer.
/// @param matrixInterpretation Specifies how to interpret the values in the matrix.
/// @param memoryLayout Specifies the memory layout of the matrix (row-major or column-major).
/// @param transpose Whether to transpose the matrix before multiplication.
/// @param matrixStride The stride between matrix rows/columns in bytes.
/// @return A new cooperative vector containing the result of the matrix multiplication.
/// @remarks Unlike coopVecMatMul, this function supports packed input interpretations where multiple values
/// can be packed into each element of the input vector. The k parameter specifies the actual number of
/// values to use from the packed input.
// TODO: Can we ForceInline for just hlsl? the other platforms don't really
// need it
[ForceInline]
[require(cooperative_vector)]
__generic<T : __BuiltinArithmeticType, let M : int, let PackedK : int, U : __BuiltinArithmeticType>
CoopVec<T, M> coopVecMatMulPacked(
    CoopVec<U, PackedK> input,
    constexpr CoopVecComponentType inputInterpretation,
    constexpr int k,
    $(buffer.type) matrix,
    int32_t matrixOffset,
    constexpr CoopVecComponentType matrixInterpretation,
    constexpr CoopVecMatrixLayout memoryLayout,
    constexpr bool transpose,
    constexpr uint matrixStride
)
{
    static_assert(__isPackedInputInterpretation(inputInterpretation) || k == PackedK
                 , "for non-packed inputInterpretation values k must be equal to the input vector length");
    static_assert(!__isPackedInputInterpretation(inputInterpretation)
                || k <= __inputInterpretationPackingFactor(inputInterpretation)*PackedK
                 , "for packed inputInterpretation values k must be less than or equal to the input vector length times the packing factor");

    __target_switch
    {
    case spirv:
        let m : int32_t = M;
        let matrixInterpretationSpirv : int32_t = __getSpvCoopVecComponentType(matrixInterpretation);
        let inputInterpretationSpirv : int32_t = __getSpvCoopVecComponentType(inputInterpretation);
        let memoryLayoutSpirv : int32_t = __getSpvCoopVecMatrixLayout(memoryLayout);
        let matrixPtr = matrix.GetBufferPointer();
        return spirv_asm
        {
            result:$$CoopVec<T, M> = OpCooperativeVectorMatrixMulNV $input $inputInterpretationSpirv $matrixPtr $matrixOffset $matrixInterpretationSpirv $m $k $memoryLayoutSpirv $transpose $matrixStride;
        };

    case hlsl:
        var ret = CoopVec<T, M>(0);
        let inputInterpretationHLSL = __getHLSLCoopVecComponentType(inputInterpretation);
        let matrixInterpretationHLSL = __getHLSLCoopVecComponentType(matrixInterpretation);
        let memoryLayoutHLSL = __getHLSLCoopVecMatrixLayout(memoryLayout);
        ret.__mutMatMul(
             input,
             inputInterpretationHLSL,
             matrix,
             matrixOffset,
             matrixInterpretationHLSL,
             M,
             k,
             memoryLayoutHLSL,
             transpose,
             matrixStride
        );
        return ret;

    default:
        var result = CoopVec<T, M>(0);
        var v = CoopVec<T, PackedK*4>();
        // TODO: Insert language from the spec to describe this madness
        if(k == PackedK)
        {
            for(int i = 0; i < k; ++i)
                v[i] = __arithmetic_cast<T>(input[i]);
        }
        else
        {
            static_assert(k == PackedK*4, "K must be 4 * PackedK for the non-spirv coopVecMatMulPacked backend");
            static_assert(inputInterpretation == CoopVecComponentType::SignedInt8Packed ||
                          inputInterpretation == CoopVecComponentType::UnsignedInt8Packed,
                          "Packing is only supported for 4*int8 or 4*uint8 vectors");
            for(int i = 0; i < k; ++i)
            {
                let n = __arithmetic_cast<int32_t>(input[i/4]);
                let s = int8_t(n >> ((i % 4) * 8) & 0xff);
                v[i] = T(s);
            }
        }

        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                int row = (transpose ^ memoryLayout == CoopVecMatrixLayout::ColumnMajor) ? j : i;
                int col = (transpose ^ memoryLayout == CoopVecMatrixLayout::ColumnMajor) ? i : j;
                int offset = matrixOffset + (row * matrixStride + col * __coopVecComponentTypeStride(matrixInterpretation));

                switch (matrixInterpretation)
                {
                    case CoopVecComponentType::Float16:
                        result[i] += __arithmetic_cast<T>(matrix.LoadByteOffset<half>(offset)) * v[j];
                        break;
                    case CoopVecComponentType::Float32:
                        result[i] += __arithmetic_cast<T>(matrix.LoadByteOffset<float>(offset)) * v[j];
                        break;
                    case CoopVecComponentType::Float64:
                        result[i] += __arithmetic_cast<T>(matrix.LoadByteOffset<double>(offset)) * v[j];
                        break;
                    case CoopVecComponentType::SignedInt8:
                        result[i] += __arithmetic_cast<T>(matrix.LoadByteOffset<int8_t>(offset)) * v[j];
                        break;
                    case CoopVecComponentType::SignedInt16:
                        result[i] += __arithmetic_cast<T>(matrix.LoadByteOffset<int16_t>(offset)) * v[j];
                        break;
                    case CoopVecComponentType::SignedInt32:
                        result[i] += __arithmetic_cast<T>(matrix.LoadByteOffset<int32_t>(offset)) * v[j];
                        break;
                    case CoopVecComponentType::SignedInt64:
                        result[i] += __arithmetic_cast<T>(matrix.LoadByteOffset<int64_t>(offset)) * v[j];
                        break;
                    case CoopVecComponentType::SignedInt8Packed:
                        result[i] += __arithmetic_cast<T>(matrix.LoadByteOffset<int8_t>(offset)) * v[j];
                        break;
                    case CoopVecComponentType::UnsignedInt8:
                        result[i] += __arithmetic_cast<T>(matrix.LoadByteOffset<uint8_t>(offset)) * v[j];
                        break;
                    case CoopVecComponentType::UnsignedInt16:
                        result[i] += __arithmetic_cast<T>(matrix.LoadByteOffset<uint16_t>(offset)) * v[j];
                        break;
                    case CoopVecComponentType::UnsignedInt32:
                        result[i] += __arithmetic_cast<T>(matrix.LoadByteOffset<uint32_t>(offset)) * v[j];
                        break;
                    case CoopVecComponentType::UnsignedInt64:
                        result[i] += __arithmetic_cast<T>(matrix.LoadByteOffset<uint64_t>(offset)) * v[j];
                        break;
                    case CoopVecComponentType::UnsignedInt8Packed:
                        result[i] += __arithmetic_cast<T>(matrix.LoadByteOffset<uint8_t>(offset)) * v[j];
                        break;
                }
            }
        }

        return result;
    }
}

/// Multiply a cooperative vector with a matrix.
/// @param input The input cooperative vector to multiply with the matrix.
/// @param inputInterpretation Specifies how to interpret the values in the input vector (e.g. as 8-bit integers, 16-bit floats, etc).
/// @param matrix The matrix to multiply with the input vector.
/// @param matrixOffset Byte offset into the matrix buffer.
/// @param matrixInterpretation Specifies how to interpret the values in the matrix (e.g. as 8-bit integers, 16-bit floats, etc).
/// @param memoryLayout Specifies the memory layout of the matrix (row-major or column-major).
/// @param transpose Whether to transpose the matrix before multiplication.
/// @param matrixStride The stride in bytes between rows/columns of the matrix.
/// @return A new cooperative vector containing the result of the matrix multiplication.
[ForceInline]
[require(cooperative_vector)]
__generic<T : __BuiltinArithmeticType, let M : int, let K : int, U : __BuiltinArithmeticType>
CoopVec<T, M> coopVecMatMul(
    CoopVec<U, K> input,
    constexpr CoopVecComponentType inputInterpretation,
    $(buffer.type) matrix,
    int32_t matrixOffset,
    constexpr CoopVecComponentType matrixInterpretation,
    constexpr CoopVecMatrixLayout memoryLayout,
    constexpr bool transpose,
    constexpr uint matrixStride
)
{
    static_assert(!__isPackedInputInterpretation(inputInterpretation)
                 , "for packed inputInterpretation values please use coopVecMatMulPacked and specify k manually");
    return coopVecMatMulPacked<
        T, M, K, U>(
        input,
        inputInterpretation,
        K,
        matrix,
        matrixOffset,
        matrixInterpretation,
        memoryLayout,
        transpose,
        matrixStride);
}

/// Multiply a cooperative vector with a matrix and add a bias vector.
/// @param input The input cooperative vector to multiply with the matrix.
/// @param inputInterpretation Specifies how to interpret the values in the input vector (e.g. as packed values).
/// @param k The number of columns in the matrix.
/// @param matrix The matrix buffer to multiply with.
/// @param matrixOffset Byte offset into the matrix buffer.
/// @param matrixInterpretation Specifies how to interpret the values in the matrix.
/// @param bias The bias buffer to add after multiplication.
/// @param biasOffset Byte offset into the bias buffer.
/// @param biasInterpretation Specifies how to interpret the values in the bias vector.
/// @param memoryLayout Specifies the memory layout of the matrix (row-major or column-major).
/// @param transpose Whether to transpose the matrix before multiplication.
/// @param matrixStride The stride between matrix rows/columns in bytes.
/// @return A new cooperative vector containing the result of the matrix multiplication with added bias.
/// @remarks Unlike coopVecMatMulAdd, this function supports packed input interpretations where multiple values
/// can be packed into each element of the input vector. The k parameter specifies the actual number of
/// values to use from the packed input.
[ForceInline]
[require(cooperative_vector)]
CoopVec<T, M> coopVecMatMulAddPacked<T : __BuiltinArithmeticType, let M : int, let PackedK : int, U : __BuiltinArithmeticType>(
    CoopVec<U, PackedK> input,
    constexpr CoopVecComponentType inputInterpretation,
    constexpr int k,
    $(buffer.type) matrix,
    int32_t matrixOffset,
    constexpr CoopVecComponentType matrixInterpretation,
    $(buffer.type) bias,
    int32_t biasOffset,
    constexpr CoopVecComponentType biasInterpretation,
    constexpr CoopVecMatrixLayout memoryLayout,
    constexpr bool transpose,
    constexpr uint matrixStride
)
{
    static_assert(__isPackedInputInterpretation(inputInterpretation) || k == PackedK
                 , "for non-packed inputInterpretation values k must be equal to the input vector length");
    static_assert(!__isPackedInputInterpretation(inputInterpretation)
                || k <= __inputInterpretationPackingFactor(inputInterpretation)*PackedK
                 , "for packed inputInterpretation values k must be less than or equal to the input vector length times the packing factor");

    __target_switch
    {
    case spirv:
        let m : int32_t = M;
        let matrixInterpretationSpirv : int32_t = __getSpvCoopVecComponentType(matrixInterpretation);
        let biasInterpretationSpirv : int32_t = __getSpvCoopVecComponentType(biasInterpretation);
        let inputInterpretationSpirv : int32_t = __getSpvCoopVecComponentType(inputInterpretation);
        let memoryLayoutSpirv : int32_t = __getSpvCoopVecMatrixLayout(memoryLayout);
        let matrixPtr = matrix.GetBufferPointer();
        let biasPtr = bias.GetBufferPointer();
        return spirv_asm
        {
            result:$$CoopVec<T, M> = OpCooperativeVectorMatrixMulAddNV $input $inputInterpretationSpirv $matrixPtr $matrixOffset $matrixInterpretationSpirv $biasPtr $biasOffset $biasInterpretationSpirv $m $k $memoryLayoutSpirv $transpose $matrixStride;
        };

    case hlsl:
        var ret = CoopVec<T, M>(0);
        let inputInterpretationHLSL = __getHLSLCoopVecComponentType(inputInterpretation);
        let matrixInterpretationHLSL = __getHLSLCoopVecComponentType(matrixInterpretation);
        let biasInterpretationHLSL = __getHLSLCoopVecComponentType(biasInterpretation);
        let memoryLayoutHLSL = __getHLSLCoopVecMatrixLayout(memoryLayout);
        ret.__mutMatMulAdd(
             input,
             inputInterpretationHLSL,
             matrix,
             matrixOffset,
             matrixInterpretationHLSL,
             bias,
             biasOffset,
             biasInterpretationHLSL,
             M,
             k,
             memoryLayoutHLSL,
             transpose,
             matrixStride
        );
        return ret;

    default:
        var result = coopVecMatMulPacked<T, M, PackedK, U>(
            input,
            inputInterpretation,
            k,
            matrix,
            matrixOffset,
            matrixInterpretation,
            memoryLayout,
            transpose,
            matrixStride);

        for (int i = 0; i < M; ++i)
        {
            int b = biasOffset + i * __coopVecComponentTypeStride(biasInterpretation);
            switch (biasInterpretation)
            {
                case CoopVecComponentType::Float16:
                    result[i] += __arithmetic_cast<T>(bias.LoadByteOffset<half>(b));
                    break;
                case CoopVecComponentType::Float32:
                    result[i] += __arithmetic_cast<T>(bias.LoadByteOffset<float>(b));
                    break;
                case CoopVecComponentType::Float64:
                    result[i] += __arithmetic_cast<T>(bias.LoadByteOffset<double>(b));
                    break;
                case CoopVecComponentType::SignedInt8:
                    result[i] += __arithmetic_cast<T>(bias.LoadByteOffset<int8_t>(b));
                    break;
                case CoopVecComponentType::SignedInt16:
                    result[i] += __arithmetic_cast<T>(bias.LoadByteOffset<int16_t>(b));
                    break;
                case CoopVecComponentType::SignedInt32:
                    result[i] += __arithmetic_cast<T>(bias.LoadByteOffset<int32_t>(b));
                    break;
                case CoopVecComponentType::SignedInt64:
                    result[i] += __arithmetic_cast<T>(bias.LoadByteOffset<int64_t>(b));
                    break;
                case CoopVecComponentType::SignedInt8Packed:
                    result[i] += __arithmetic_cast<T>(bias.LoadByteOffset<int8_t>(b));
                    break;
                case CoopVecComponentType::UnsignedInt8:
                    result[i] += __arithmetic_cast<T>(bias.LoadByteOffset<uint8_t>(b));
                    break;
                case CoopVecComponentType::UnsignedInt16:
                    result[i] += __arithmetic_cast<T>(bias.LoadByteOffset<uint16_t>(b));
                    break;
                case CoopVecComponentType::UnsignedInt32:
                    result[i] += __arithmetic_cast<T>(bias.LoadByteOffset<uint32_t>(b));
                    break;
                case CoopVecComponentType::UnsignedInt64:
                    result[i] += __arithmetic_cast<T>(bias.LoadByteOffset<uint64_t>(b));
                    break;
                case CoopVecComponentType::UnsignedInt8Packed:
                    result[i] += __arithmetic_cast<T>(bias.LoadByteOffset<uint8_t>(b));
                    break;
            }
        }

        return result;
    }
}

/// Multiply a cooperative vector with a matrix and add a bias vector.
/// @param input The input cooperative vector to multiply with the matrix.
/// @param inputInterpretation Specifies how to interpret the values in the input vector.
/// @param matrix The matrix buffer to multiply with.
/// @param matrixOffset Byte offset into the matrix buffer.
/// @param matrixInterpretation Specifies how to interpret the values in the matrix.
/// @param bias The bias buffer to add after multiplication.
/// @param biasOffset Byte offset into the bias buffer.
/// @param biasInterpretation Specifies how to interpret the values in the bias vector.
/// @param memoryLayout Specifies the memory layout of the matrix (row-major or column-major).
/// @param transpose Whether to transpose the matrix before multiplication.
/// @param matrixStride The stride between matrix rows/columns in bytes.
/// @return A new cooperative vector containing the result of the matrix multiplication plus bias.
[ForceInline]
[require(cooperative_vector)]
__generic<T : __BuiltinArithmeticType, let M : int, let K : int, U : __BuiltinArithmeticType>
CoopVec<T, M> coopVecMatMulAdd(
    CoopVec<U, K> input,
    constexpr CoopVecComponentType inputInterpretation,
    $(buffer.type) matrix,
    int32_t matrixOffset,
    constexpr CoopVecComponentType matrixInterpretation,
    $(buffer.type) bias,
    int32_t biasOffset,
    constexpr CoopVecComponentType biasInterpretation,
    constexpr CoopVecMatrixLayout memoryLayout,
    constexpr bool transpose,
    constexpr uint matrixStride
)
{
    static_assert(!__isPackedInputInterpretation(inputInterpretation)
                 , "for packed inputInterpretation values please use coopVecMatMulPacked and specify k manually");
    return coopVecMatMulAddPacked<
        T, M, K, U>(
        input,
        inputInterpretation,
        K,
        matrix,
        matrixOffset,
        matrixInterpretation,
        bias,
        biasOffset,
        biasInterpretation,
        memoryLayout,
        transpose,
        matrixStride);
}

//
// Coop Vector accumulation
//

${{{{
if(buffer.isRW)
{
}}}}
/// Accumulate the outer product of two cooperative vectors into a matrix.
/// @param a The first cooperative vector.
/// @param b The second cooperative vector.
/// @param matrix The matrix buffer to accumulate the result into.
/// @param matrixOffset Byte offset into the matrix buffer.
/// @param matrixStride The stride between matrix rows/columns in bytes.
/// @param memoryLayout Specifies the memory layout of the matrix (row-major or column-major).
/// @param matrixInterpretation Specifies how to interpret the values in the matrix.
[require(cooperative_vector)]
void coopVecOuterProductAccumulate<T : __BuiltinArithmeticType, let M : int, let N : int>(
    CoopVec<T, M> a,
    CoopVec<T, N> b,
    $(buffer.type) matrix,
    int32_t matrixOffset,
    constexpr uint matrixStride,
    constexpr CoopVecMatrixLayout memoryLayout,
    constexpr CoopVecComponentType matrixInterpretation,
)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "$0.OuterProductAccumulate($1, $2, $3, $4, $5, $6)";
    case spirv:
        let matrixInterpretationSpirv : int = __getSpvCoopVecComponentType(matrixInterpretation);
        let memoryLayoutSpirv : int = __getSpvCoopVecMatrixLayout(memoryLayout);
        let matrixPtr = matrix.GetBufferPointer();
        spirv_asm
        {
            OpCapability CooperativeVectorTrainingNV;
            OpCooperativeVectorOuterProductAccumulateNV $matrixPtr $matrixOffset $a $b $memoryLayoutSpirv $matrixInterpretationSpirv $matrixStride;
        };
    default:
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                T product = a[i] * b[j];
                int row = (memoryLayout == CoopVecMatrixLayout::ColumnMajor) ? j : i;
                int col = (memoryLayout == CoopVecMatrixLayout::ColumnMajor) ? i : j;
                int offset = matrixOffset + (row * matrixStride + col * __coopVecComponentTypeStride(matrixInterpretation));

                switch (matrixInterpretation)
                {
                    case CoopVecComponentType::Float16:
                        matrix.StoreByteOffset<half>(offset, matrix.LoadByteOffset<half>(offset) + __arithmetic_cast<half>(product));
                        break;
                    case CoopVecComponentType::Float32:
                        matrix.StoreByteOffset<float>(offset, matrix.LoadByteOffset<float>(offset) + __arithmetic_cast<float>(product));
                        break;
                    case CoopVecComponentType::Float64:
                        matrix.StoreByteOffset<double>(offset, matrix.LoadByteOffset<double>(offset) + __arithmetic_cast<double>(product));
                        break;
                    case CoopVecComponentType::SignedInt8:
                        matrix.StoreByteOffset<int8_t>(offset, matrix.LoadByteOffset<int8_t>(offset) + __arithmetic_cast<int8_t>(product));
                        break;
                    case CoopVecComponentType::SignedInt16:
                        matrix.StoreByteOffset<int16_t>(offset, matrix.LoadByteOffset<int16_t>(offset) + __arithmetic_cast<int16_t>(product));
                        break;
                    case CoopVecComponentType::SignedInt32:
                        matrix.StoreByteOffset<int32_t>(offset, matrix.LoadByteOffset<int32_t>(offset) + __arithmetic_cast<int32_t>(product));
                        break;
                    case CoopVecComponentType::SignedInt64:
                        matrix.StoreByteOffset<int64_t>(offset, matrix.LoadByteOffset<int64_t>(offset) + __arithmetic_cast<int64_t>(product));
                        break;
                    case CoopVecComponentType::SignedInt8Packed:
                        matrix.StoreByteOffset<int8_t>(offset, matrix.LoadByteOffset<int8_t>(offset) + __arithmetic_cast<int8_t>(product));
                        break;
                    case CoopVecComponentType::UnsignedInt8:
                        matrix.StoreByteOffset<uint8_t>(offset, matrix.LoadByteOffset<uint8_t>(offset) + __arithmetic_cast<uint8_t>(product));
                        break;
                    case CoopVecComponentType::UnsignedInt16:
                        matrix.StoreByteOffset<uint16_t>(offset, matrix.LoadByteOffset<uint16_t>(offset) + __arithmetic_cast<uint16_t>(product));
                        break;
                    case CoopVecComponentType::UnsignedInt32:
                        matrix.StoreByteOffset<uint32_t>(offset, matrix.LoadByteOffset<uint32_t>(offset) + __arithmetic_cast<uint32_t>(product));
                        break;
                    case CoopVecComponentType::UnsignedInt64:
                        matrix.StoreByteOffset<uint64_t>(offset, matrix.LoadByteOffset<uint64_t>(offset) + __arithmetic_cast<uint64_t>(product));
                        break;
                    case CoopVecComponentType::UnsignedInt8Packed:
                        matrix.StoreByteOffset<uint8_t>(offset, matrix.LoadByteOffset<uint8_t>(offset) + __arithmetic_cast<uint8_t>(product));
                        break;
                }
            }
        }
    }
}

/// Accumulate the sum of a cooperative vector into a buffer at the specified offset.
/// @param v The cooperative vector to sum.
/// @param buffer The buffer to accumulate the sum into.
/// @param offset Byte offset into the buffer.
[require(cooperative_vector)]
void coopVecReduceSumAccumulate<T : __BuiltinArithmeticType, let N : int>(
    CoopVec<T, N> v,
    $(buffer.type) buffer,
    int32_t offset
)
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "$0.ReduceSumAccumulate($1, $2)";
    case spirv:
        let bufferPtr = buffer.GetBufferPointer();
        spirv_asm
        {
            OpCapability CooperativeVectorTrainingNV;
            OpCooperativeVectorReduceSumAccumulateNV $bufferPtr $offset $v;
        };
    default:
        for (int i = 0; i < N; ++i)
        {
            int byteOffset = offset + i * __naturalStrideOf<T>();
            T currentValue = buffer.LoadByteOffset<T>(byteOffset);
            T newValue = currentValue + __arithmetic_cast<T>(v[i]);
            buffer.StoreByteOffset(byteOffset, newValue);
        }
    }
}

${{{{
} // if rw
} // buffer type loop
}}}}

${{{{
static const struct {
    bool isRW;
    char const* type;
} kStructuredBufferCases_[] =
{
    {true, "RWStructuredBuffer<IgnoredBufferElementType>"},
    {false, "StructuredBuffer<IgnoredBufferElementType>"},
};
for(auto buffer : kStructuredBufferCases_) {
}}}}

/// Multiply a cooperative vector with a matrix and return the result.
/// @param input The input cooperative vector to multiply with the matrix.
/// @param inputInterpretation Specifies how to interpret the values in the input vector (e.g. as packed values).
/// @param k The number of columns in the matrix.
/// @param matrix The matrix buffer to multiply with.
/// @param matrixOffset Byte offset into the matrix buffer.
/// @param matrixInterpretation Specifies how to interpret the values in the matrix.
/// @param memoryLayout Specifies the memory layout of the matrix (row-major or column-major).
/// @param transpose Whether to transpose the matrix before multiplication.
/// @param matrixStride The stride between matrix rows/columns in bytes.
/// @return A new cooperative vector containing the result of the matrix multiplication.
/// @remarks Unlike coopVecMatMul, this function supports packed input interpretations where multiple values
/// can be packed into each element of the input vector. The k parameter specifies the actual number of
/// values to use from the packed input.
[require(spirv, cooperative_vector)]
__generic<T : __BuiltinArithmeticType, let M : int, let PackedK : int, U : __BuiltinArithmeticType,IgnoredBufferElementType>
CoopVec<T, M> coopVecMatMulPacked(
    CoopVec<U, PackedK> input,
    constexpr CoopVecComponentType inputInterpretation,
    constexpr int k,
    $(buffer.type) matrix,
    int32_t matrixOffset,
    constexpr CoopVecComponentType matrixInterpretation,
    constexpr CoopVecMatrixLayout memoryLayout,
    constexpr bool transpose,
    constexpr uint matrixStride
)
{
    static_assert(__isPackedInputInterpretation(inputInterpretation) || k == PackedK
                 , "for non-packed inputInterpretation values k must be equal to the input vector length");
    static_assert(!__isPackedInputInterpretation(inputInterpretation)
                || k <= __inputInterpretationPackingFactor(inputInterpretation)*PackedK
                 , "for packed inputInterpretation values k must be less than or equal to the input vector length times the packing factor");
    __target_switch
    {
    case spirv:
        let m : int32_t = M;
        let matrixInterpretationSpirv : int32_t = __getSpvCoopVecComponentType(matrixInterpretation);
        let inputInterpretationSpirv : int32_t = __getSpvCoopVecComponentType(inputInterpretation);
        let memoryLayoutSpirv : int32_t = __getSpvCoopVecMatrixLayout(memoryLayout);
        let matrixPtr = __getStructuredBufferPtr(matrix);
        return spirv_asm
        {
            result:$$CoopVec<T, M> = OpCooperativeVectorMatrixMulNV $input $inputInterpretationSpirv $matrixPtr $matrixOffset $matrixInterpretationSpirv $m $k $memoryLayoutSpirv $transpose $matrixStride;
        };
    }
}

// specialized coopVecMatMul for non-packed inputs
[require(spirv, cooperative_vector)]
__generic<T : __BuiltinArithmeticType, let M : int, let K : int, U : __BuiltinArithmeticType,IgnoredBufferElementType>
CoopVec<T, M> coopVecMatMul(
    CoopVec<U, K> input,
    constexpr CoopVecComponentType inputInterpretation,
    $(buffer.type) matrix,
    int32_t matrixOffset,
    constexpr CoopVecComponentType matrixInterpretation,
    constexpr CoopVecMatrixLayout memoryLayout,
    constexpr bool transpose,
    constexpr uint matrixStride
)
{
    static_assert(!__isPackedInputInterpretation(inputInterpretation)
                 , "for packed inputInterpretation values please use coopVecMatMulPacked and specify k manually");
    return coopVecMatMulPacked<
        T, M, K, U, IgnoredBufferElementType>(
        input,
        inputInterpretation,
        K,
        matrix,
        matrixOffset,
        matrixInterpretation,
        memoryLayout,
        transpose,
        matrixStride);
}

[require(spirv, cooperative_vector)]
CoopVec<T, M> coopVecMatMulAddPacked<T : __BuiltinArithmeticType, let M : int, let PackedK : int, U : __BuiltinArithmeticType, IgnoredBufferElementType>(
    CoopVec<U, PackedK> input,
    constexpr CoopVecComponentType inputInterpretation,
    constexpr int k,
    $(buffer.type) matrix,
    int32_t matrixOffset,
    constexpr CoopVecComponentType matrixInterpretation,
    $(buffer.type) bias,
    int32_t biasOffset,
    constexpr CoopVecComponentType biasInterpretation,
    constexpr CoopVecMatrixLayout memoryLayout,
    constexpr bool transpose,
    constexpr uint matrixStride
)
{
    static_assert(__isPackedInputInterpretation(inputInterpretation) || k == PackedK
                 , "for non-packed inputInterpretation values k must be equal to the input vector length");
    static_assert(!__isPackedInputInterpretation(inputInterpretation)
                || k <= __inputInterpretationPackingFactor(inputInterpretation)*PackedK
                 , "for packed inputInterpretation values k must be less than or equal to the input vector length times the packing factor");

    __target_switch
    {
    case spirv:
        let m : int32_t = M;
        let matrixInterpretationSpirv : int32_t = __getSpvCoopVecComponentType(matrixInterpretation);
        let biasInterpretationSpirv : int32_t = __getSpvCoopVecComponentType(biasInterpretation);
        let inputInterpretationSpirv : int32_t = __getSpvCoopVecComponentType(inputInterpretation);
        let memoryLayoutSpirv : int32_t = __getSpvCoopVecMatrixLayout(memoryLayout);
        let matrixPtr = __getStructuredBufferPtr(matrix);
        let biasPtr = __getStructuredBufferPtr(bias);
        return spirv_asm
        {
            result:$$CoopVec<T, M> = OpCooperativeVectorMatrixMulAddNV $input $inputInterpretationSpirv $matrixPtr $matrixOffset $matrixInterpretationSpirv $biasPtr $biasOffset $biasInterpretationSpirv $m $k $memoryLayoutSpirv $transpose $matrixStride;
        };
    }
}

[require(spirv, cooperative_vector)]
CoopVec<T, M> coopVecMatMulAdd<T : __BuiltinArithmeticType, let M : int, let K : int, U : __BuiltinArithmeticType, IgnoredBufferElementType>(
    CoopVec<U, K> input,
    constexpr CoopVecComponentType inputInterpretation,
    $(buffer.type) matrix,
    int32_t matrixOffset,
    constexpr CoopVecComponentType matrixInterpretation,
    $(buffer.type) bias,
    int32_t biasOffset,
    constexpr CoopVecComponentType biasInterpretation,
    constexpr CoopVecMatrixLayout memoryLayout,
    constexpr bool transpose,
    constexpr uint matrixStride
)
{
    static_assert(!__isPackedInputInterpretation(inputInterpretation)
                 , "for packed inputInterpretation values please use coopVecMatMulPacked and specify k manually");
    return coopVecMatMulAddPacked<
        T, M, K, U, IgnoredBufferElementType>(
        input,
        inputInterpretation,
        K,
        matrix,
        matrixOffset,
        matrixInterpretation,
        bias,
        biasOffset,
        biasInterpretation,
        memoryLayout,
        transpose,
        matrixStride);
}

//
// Coop Vector accumulation
//

${{{{
if(buffer.isRW)
{
}}}}
[require(spirv, cooperative_vector_training)]
void coopVecOuterProductAccumulate<T : __BuiltinArithmeticType, let M : int, let N : int, IgnoredBufferElementType>(
    CoopVec<T, M> a,
    CoopVec<T, N> b,
    $(buffer.type) matrix,
    int32_t matrixOffset,
    constexpr uint matrixStride,
    constexpr CoopVecMatrixLayout memoryLayout,
    constexpr CoopVecComponentType matrixInterpretation,
)
{
    __target_switch
    {
    case spirv:
        let matrixInterpretationSpirv : int = __getSpvCoopVecComponentType(matrixInterpretation);
        let memoryLayoutSpirv : int = __getSpvCoopVecMatrixLayout(memoryLayout);
        let matrixPtr = __getStructuredBufferPtr(matrix);
        spirv_asm
        {
            OpCapability CooperativeVectorTrainingNV;
            OpCooperativeVectorOuterProductAccumulateNV $matrixPtr $matrixOffset $a $b $memoryLayoutSpirv $matrixInterpretationSpirv $matrixStride;
        };
    }
}

[require(spirv, cooperative_vector_training)]
void coopVecReduceSumAccumulate<T : __BuiltinArithmeticType, let N : int, IgnoredBufferElementType>(
    CoopVec<T, N> v,
    $(buffer.type) buffer,
    int32_t offset
)
{
    __target_switch
    {
    case spirv:
        let bufferPtr = __getStructuredBufferPtr(buffer);
        spirv_asm
        {
            OpCapability CooperativeVectorTrainingNV;
            OpCooperativeVectorReduceSumAccumulateNV $bufferPtr $offset $v;
        };
    }
}

${{{{
} // if rw
} // buffer type loop
}}}}


[require(spirv, cooperative_vector_training)]
void coopVecOuterProductAccumulate<T : __BuiltinArithmeticType, let M : int, let N : int, U : __BuiltinArithmeticType, let IgnoredBufferSize : int>(
    CoopVec<T, M> a,
    CoopVec<T, N> b,
    __ref groupshared U[IgnoredBufferSize] matrix,
    int32_t matrixOffset,
    constexpr uint matrixStride,
    constexpr CoopVecMatrixLayout memoryLayout,
    constexpr CoopVecComponentType matrixInterpretation,
)
{
    __target_switch
    {
    case spirv:
        let matrixInterpretationSpirv : int = __getSpvCoopVecComponentType(matrixInterpretation);
        let memoryLayoutSpirv : int = __getSpvCoopVecMatrixLayout(memoryLayout);
        spirv_asm
        {
            OpCapability CooperativeVectorTrainingNV;
            OpCooperativeVectorOuterProductAccumulateNV &matrix $matrixOffset $a $b $memoryLayoutSpirv $matrixInterpretationSpirv $matrixStride;
        };
    }
}

[require(spirv, cooperative_vector_training)]
void coopVecReduceSumAccumulate<T : __BuiltinArithmeticType, let N : int, U, let IgnoredBufferSize : int>(
    CoopVec<T, N> v,
    __ref groupshared U[IgnoredBufferSize] buffer,
    int32_t offset
)
{
    __target_switch
    {
    case spirv:
        spirv_asm
        {
            OpCapability CooperativeVectorTrainingNV;
            OpCooperativeVectorReduceSumAccumulateNV &buffer $offset $v;
        };
    }
}

//@public:

/// Mark a variable as being workgroup uniform.
/// @param v The variable to mark as workgroup uniform.
/// @return The value of `v`.
/// @remarks This intrinsic maps to `workgroupUniformLoad` when targeting WGSL and is a no-op on other targets.
/// WGSL is strict on uniformity, and this intrinsic is needed to mark a variable as workgroup uniform in order
// to silence uniformity errors in certain cases.
T workgroupUniformLoad<T>(__ref T v)
{
    __target_switch
    {
    case wgsl:
        __intrinsic_asm "workgroupUniformLoad(&($0))";
    default:
        return v;
    }
}

//
// HLSL Pack/Unpack Math Intrinsics
//
// These were introduced in SM 6.6 but requirements are dropped to SM 5.0 here
// to expose these intrinsics on targets that do not have SM 6.6 features.
//

//@public:

typealias uint8_t4_packed = uint;
typealias int8_t4_packed = uint;

/// Unpack 4 signed 8-bit values into a vector of 16 bit integers.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
int16_t4 unpack_s8s16(int8_t4_packed packed)
{
    return unpackInt4x8ToInt16(packed);
}

/// Unpack 4 unsigned 8-bit values into a vector of 16 bit integers.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint16_t4 unpack_u8u16(uint8_t4_packed packed)
{
    return unpackUint4x8ToUint16(packed);
}

/// Unpack 4 signed 8-bit values into a vector of 32 bit integers.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
int32_t4 unpack_s8s32(int8_t4_packed packed)
{
    return unpackInt4x8ToInt32(packed);
}

/// Unpack 4 unsigned 8-bit values into a vector of 32 bit integers.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint32_t4 unpack_u8u32(uint8_t4_packed packed)
{
    return unpackUint4x8ToUint32(packed);
}

/// Pack a vector of 4 unsigned 32/16 bit integers into a packed value of 4 8-bit integers, dropping unused bits.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint8_t4_packed pack_u8(uint32_t4 unpackedValue)
{
    return packUint4x8(unpackedValue);
}

/// Pack a vector of 4 signed 32/16 bit integers into a packed value of 4 8-bit integers, dropping unused bits.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
int8_t4_packed pack_s8(int32_t4 unpackedValue)
{
    return packInt4x8(unpackedValue);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint8_t4_packed pack_u8(uint16_t4 unpackedValue)
{
    return packUint4x8(unpackedValue);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
int8_t4_packed pack_s8(int16_t4 unpackedValue)
{
    return packInt4x8(unpackedValue);
}

/// Pack a vector of 4 unsigned 32/16 bit integers into a packed value of 4 8-bit integers,
/// clamping each value to the range [0, 255] to ensure it fits within 8 bits.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint8_t4_packed pack_clamp_u8(int32_t4 unpackedValue)
{
    return packUint4x8Clamp(unpackedValue);
}

/// Pack a vector of 4 signed 32/16 bit integers into a packed value of 4 8-bit integers,
/// clamping each value to the range [-128, 127] to ensure it fits within 8 bits.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
int8_t4_packed pack_clamp_s8(int32_t4 unpackedValue)
{
    return packInt4x8Clamp(unpackedValue);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint8_t4_packed pack_clamp_u8(int16_t4 unpackedValue)
{
    return packUint4x8Clamp(unpackedValue);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
int8_t4_packed pack_clamp_s8(int16_t4 unpackedValue)
{
    return packInt4x8Clamp(unpackedValue);
}

// Work-graphs

//@public:
/// read-only input to Broadcasting launch node.
__generic<T>
//TODO: DispatchNodeInputRecord should be available only for broadcasting node shader.
//[require(broadcasting_node)]
[require(spirv)]
struct DispatchNodeInputRecord
{
    /// Provide an access to a record object that only holds a single record.
    NodePayloadPtr<T> Get()
    {
        int index = 0;
        __target_switch
        {
        case spirv:
            return spirv_asm
            {
                %in_payload_t = OpTypeNodePayloadArrayAMDX $$T;
                %in_payload_ptr_t = OpTypePointer NodePayloadAMDX %in_payload_t;
                %var = OpVariable %in_payload_ptr_t NodePayloadAMDX;
                result : $$NodePayloadPtr<T> = OpAccessChain %var $index;
            };
        }
    }
};

//
// Pack/Unpack Intrinsics
//

//@hidden:

[__readNone]
[ForceInline]
uint16_t __lsbAsUint16(uint32_t val)
{
    return uint16_t(val & 0xFFU);
}

[__readNone]
[ForceInline]
uint32_t __lsbAsUint32(uint32_t val)
{
    return (val & 0xFFU);
}

[__readNone]
[ForceInline]
int8_t __lsbAsInt8(uint32_t val)
{
    return int8_t(val);
}

[__readNone]
[ForceInline]
int16_t __lsbAsInt16(uint32_t val)
{
    return int16_t(__lsbAsInt8(val));
}

[__readNone]
[ForceInline]
int32_t __lsbAsInt32(uint32_t val)
{
    return int32_t(__lsbAsInt8(val));
}

[__readNone]
[ForceInline]
uint32_t2 __unpackUint2x16ToUint32(uint packedValue)
{
    return uint32_t2(packedValue & 0xFFFFU, packedValue >> 16U);
}

[__readNone]
[ForceInline]
int32_t2 __unpackInt2x16ToInt32(uint packedValue)
{
    int signedValue = int(packedValue);
    return int32_t2(signedValue << 16U, signedValue) >> 16U;
}

[__readNone]
[ForceInline]
uint __packUint2x16(uint32_t2 unpackedValue)
{
    return (unpackedValue.x & 0xFFFF) | (unpackedValue.y << 16U);
}

[__readNone]
[ForceInline]
uint __packInt2x16(int32_t2 unpackedValue)
{
    return uint((unpackedValue.x & 0xFFFF) | (unpackedValue.y << 16U));
}

//@public:

/// Unpack 4 unsigned 8-bit values into a vector of 32 bit integers.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint32_t4 unpackUint4x8ToUint32(uint packedValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "unpack_u8u32";
    case wgsl: __intrinsic_asm "unpack4xU8";
    case spirv:
        return spirv_asm
        {
            %u8Vec = OpBitcast $$vector<uint8_t, 4> $packedValue;
            result:$$vector<uint32_t, 4> = OpUConvert %u8Vec
        };
    default:
        return uint32_t4
        (
            __lsbAsUint32(packedValue),
            __lsbAsUint32(packedValue >> 8U),
            __lsbAsUint32(packedValue >> 16U),
            uint32_t(packedValue >> 24U),
        );
    }
}

/// Unpack 4 unsigned 8-bit values into a vector of 16 bit integers.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint16_t4 unpackUint4x8ToUint16(uint packedValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "unpack_u8u16";
    case spirv:
        return spirv_asm
        {
            %u8Vec = OpBitcast $$vector<uint8_t, 4> $packedValue;
            result:$$vector<uint16_t, 4> = OpUConvert %u8Vec
        };
    default:
        return uint16_t4
        (
            __lsbAsUint16(packedValue),
            __lsbAsUint16(packedValue >> 8U),
            __lsbAsUint16(packedValue >> 16U),
            uint16_t(packedValue >> 24U),
        );
    }
}

/// Unpack 4 signed 8-bit values into a vector of 32 bit integers.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
int32_t4 unpackInt4x8ToInt32(uint packedValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "unpack_s8s32";
    case wgsl: __intrinsic_asm "unpack4xI8";
    case spirv:
        return spirv_asm
        {
            %s8Vec = OpBitcast $$vector<int8_t, 4> $packedValue;
            result:$$vector<int32_t, 4> = OpSConvert %s8Vec
        };
    default:
        return int32_t4
        (
            __lsbAsInt32(packedValue),
            __lsbAsInt32(packedValue >> 8U),
            __lsbAsInt32(packedValue >> 16U),
            int32_t(int8_t(packedValue >> 24U)),
        );
    }
}

/// Unpack 4 signed 8-bit values into a vector of 16 bit integers.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
int16_t4 unpackInt4x8ToInt16(uint packedValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "unpack_s8s16";
    case spirv:
        return spirv_asm
        {
            %s8Vec = OpBitcast $$vector<int8_t, 4> $packedValue;
            result:$$vector<int16_t, 4> = OpSConvert %s8Vec
        };
    default:
        return int16_t4
        (
            __lsbAsInt16(packedValue),
            __lsbAsInt16(packedValue >> 8U),
            __lsbAsInt16(packedValue >> 16U),
            int16_t(int8_t(packedValue >> 24U)),
        );
    }
}

/// Pack a vector of 4 unsigned 32/16 bit integers into a packed value of 4 8-bit integers, dropping unused bits.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packUint4x8(uint32_t4 unpackedValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "pack_u8";
    case wgsl: __intrinsic_asm "pack4xU8";
    default:
        return __lsbAsUint32(unpackedValue.x)
            | (__lsbAsUint32(unpackedValue.y) << 8U)
            | (__lsbAsUint32(unpackedValue.z) << 16U)
            | (unpackedValue.w << 24U);
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packUint4x8(uint16_t4 unpackedValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "pack_u8";
    default:
        return packUint4x8(uint32_t4(unpackedValue));
    }
}

/// Pack a vector of 4 signed 32/16 bit integers into a packed value of 4 8-bit integers, dropping unused bits.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packInt4x8(int32_t4 unpackedValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "pack_s8";
    case wgsl: __intrinsic_asm "pack4xI8";
    default:
        return packUint4x8(uint32_t4(unpackedValue));
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packInt4x8(int16_t4 unpackedValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "pack_s8";
    default:
        return packUint4x8(uint32_t4(unpackedValue));
    }
}

/// Pack a vector of 4 signed 32/16 bit integers into a packed value of 4 8-bit integers,
/// clamping each value to the range [0, 255] to ensure it fits within 8 bits.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packUint4x8Clamp(int32_t4 unpackedValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "pack_clamp_u8";
    case wgsl: __intrinsic_asm "pack4xU8Clamp(vec4<u32>($0))";
    default:
        return packInt4x8(clamp(unpackedValue, 0, 255));
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packUint4x8Clamp(int16_t4 unpackedValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "pack_clamp_u8";
    default:
        return packInt4x8(clamp(unpackedValue, 0, 255));
    }
}

/// Pack a vector of 4 signed 32/16 bit integers into a packed value of 4 8-bit integers,
/// clamping each value to the range [-128, 127] to ensure it fits within 8 bits.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packInt4x8Clamp(int32_t4 unpackedValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "pack_clamp_s8";
    case wgsl: __intrinsic_asm "pack4xI8Clamp";
    default:
        return packInt4x8(clamp(unpackedValue, -128, 127));
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packInt4x8Clamp(int16_t4 unpackedValue)
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "pack_clamp_s8";
    default:
        return packInt4x8(clamp(unpackedValue, -128, 127));
    }
}

//
// Floating-point Pack/Unpack Intrinsics
//

// @public:

/// Unpack a 32-bit unsigned integer into four 8-bit unsigned integers.
/// Then, each 8-bit value is converted to a normalized single-precision
/// floating-point value to generate a 4-component vector.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
float4 unpackUnorm4x8ToFloat(uint packedValue)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "unpackUnorm4x8";
    case metal: __intrinsic_asm "unpack_unorm4x8_to_float";
    case spirv:
        return spirv_asm
        {
            result:$$float4 = OpExtInst glsl450 UnpackUnorm4x8 $packedValue;
        };
    case wgsl: __intrinsic_asm "unpack4x8unorm";
    default:
        uint4 unpackedIntegers = unpackUint4x8ToUint32(packedValue);
        return float4(unpackedIntegers) / 255.0;
    }
}

/// Unpack a 32-bit unsigned integer into four 8-bit unsigned integers.
/// Then, each 8-bit value is converted to a normalized half-precision
/// floating-point value to generate a 4-component vector.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
half4 unpackUnorm4x8ToHalf(uint packedValue)
{
    __target_switch
    {
    case metal: __intrinsic_asm "unpack_unorm4x8_to_half";
    default:
        return half4(unpackUnorm4x8ToFloat(packedValue));
    }
}

/// Unpack a 32-bit unsigned integer into four 8-bit signed integers.
/// Then, each 8-bit value is converted to a normalized single-precision
/// floating-point value to generate a 4-component vector.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
float4 unpackSnorm4x8ToFloat(uint packedValue)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "unpackSnorm4x8";
    case metal: __intrinsic_asm "unpack_snorm4x8_to_float";
    case spirv:
        return spirv_asm
        {
            result:$$float4 = OpExtInst glsl450 UnpackSnorm4x8 $packedValue;
        };
    case wgsl: __intrinsic_asm "unpack4x8snorm";
    default:
        int4 unpackedIntegers = unpackInt4x8ToInt32(packedValue);
        return clamp(float4(unpackedIntegers) / 127.0, -1.0, 1.0);
    }
}

/// Unpack a 32-bit unsigned integer into four 8-bit signed integers.
/// Then, each 8-bit value is converted to a normalized half-precision
/// floating-point value to generate a 4-component vector.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
half4 unpackSnorm4x8ToHalf(uint packedValue)
{
    __target_switch
    {
    case metal: __intrinsic_asm "unpack_snorm4x8_to_half";
    default:
        return half4(unpackSnorm4x8ToFloat(packedValue));
    }
}

/// Unpack a 32-bit unsigned integer into two 16-bit usigned integers.
/// Then, each 16-bit value is converted to a normalized single-precision
/// floating-point value to generate a 2-component vector.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
float2 unpackUnorm2x16ToFloat(uint packedValue)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "unpackUnorm2x16";
    case metal: __intrinsic_asm "unpack_unorm2x16_to_float";
    case spirv:
        return spirv_asm
        {
            result:$$float2 = OpExtInst glsl450 UnpackUnorm2x16 $packedValue;
        };
    case wgsl: __intrinsic_asm "unpack2x16unorm";
    default:
        uint2 unpackedIntegers = __unpackUint2x16ToUint32(packedValue);
        return float2(unpackedIntegers) / 65535.0;
    }
}

/// Unpack a 32-bit unsigned integer into two 16-bit usigned integers.
/// Then, each 16-bit value is converted to a normalized half-precision
/// floating-point value to generate a 2-component vector.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
half2 unpackUnorm2x16ToHalf(uint packedValue)
{
    __target_switch
    {
    case metal: __intrinsic_asm "unpack_unorm2x16_to_half";
    default:
        return half2(unpackUnorm2x16ToFloat(packedValue));
    }
}

/// Unpack a 32-bit unsigned integer into two 16-bit signed integers.
/// Then, each 16-bit value is converted to a normalized single-precision
/// floating-point value to generate a 2-component vector.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
float2 unpackSnorm2x16ToFloat(uint packedValue)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "unpackSnorm2x16";
    case metal: __intrinsic_asm "unpack_snorm2x16_to_float";
    case spirv:
        return spirv_asm
        {
            result:$$float2 = OpExtInst glsl450 UnpackSnorm2x16 $packedValue;
        };
    case wgsl: __intrinsic_asm "unpack2x16snorm";
    default:
        int2 unpackedIntegers = __unpackInt2x16ToInt32(packedValue);
        return clamp(float2(unpackedIntegers) / 32767.0, -1.0, 1.0);
    }
}

/// Unpack a 32-bit unsigned integer into two 16-bit signed integers.
/// Then, each 16-bit value is converted to a normalized half-precision
/// floating-point value to generate a 2-component vector.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
half2 unpackSnorm2x16ToHalf(uint packedValue)
{
    __target_switch
    {
    case metal: __intrinsic_asm "unpack_snorm2x16_to_half";
    default:
        return half2(unpackSnorm2x16ToFloat(packedValue));
    }
}

/// Unpack a 32-bit unsigned integer into two 16-bit signed integers.
/// Then, each 16-bit value is converted to an IEEE-754 binary16 single-precision
/// floating-point value to generate a 2-component vector.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
float2 unpackHalf2x16ToFloat(uint packedValue)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "unpackHalf2x16";
    case spirv:
        return spirv_asm
        {
            result:$$float2 = OpExtInst glsl450 UnpackHalf2x16 $packedValue;
        };
    case wgsl: __intrinsic_asm "unpack2x16float";
    default:
        uint2 unpackedIntegers = __unpackUint2x16ToUint32(packedValue);
        return f16tof32(unpackedIntegers);
    }
}

/// Unpack a 32-bit unsigned integer into two 16-bit signed integers.
/// Then, each 16-bit value is converted to an IEEE-754 binary16 half-precision
/// floating-point value to generate a 2-component vector.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
half2 unpackHalf2x16ToHalf(uint packedValue)
{
    return half2(unpackHalf2x16ToFloat(packedValue));
}

/// Convert a 4-component vector of normalized unsigned single/half-precision floating-point
/// values to four 8-bit integer values, then pack these 8-bit values into a
/// 32-bit unsigned integer.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packUnorm4x8(float4 unpackedValue)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "packUnorm4x8";
    case metal: __intrinsic_asm "pack_float_to_unorm4x8";
    case spirv:
        return spirv_asm
        {
            result:$$uint = OpExtInst glsl450 PackUnorm4x8 $unpackedValue
        };
    case wgsl: __intrinsic_asm "pack4x8unorm";
    default:
        uint4 values = uint4(round(saturate(unpackedValue) * 255.0));
        return packUint4x8(values);
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packUnorm4x8(half4 unpackedValue)
{
    __target_switch
    {
    case metal: __intrinsic_asm "pack_half_to_unorm4x8";
    default:
        return packUnorm4x8(float4(unpackedValue));
    }
}

/// Convert a 4-component vector of normalized signed single/half-precision floating-point
/// values to four 8-bit integer values, then pack these 8-bit values into a
/// 32-bit unsigned integer.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packSnorm4x8(float4 unpackedValue)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "packSnorm4x8";
    case metal: __intrinsic_asm "pack_float_to_snorm4x8";
    case spirv:
        return spirv_asm
        {
            result:$$uint = OpExtInst glsl450 PackSnorm4x8 $unpackedValue
        };
    case wgsl: __intrinsic_asm "pack4x8snorm";
    default:
        int4 values = int4(round(clamp(unpackedValue, -1.0, 1.0) * 127.0)) & 0xFF;
        return packInt4x8(values);
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packSnorm4x8(half4 unpackedValue)
{
    __target_switch
    {
    case metal: __intrinsic_asm "pack_half_to_snorm4x8";
    default:
        return packSnorm4x8(float4(unpackedValue));
    }
}

/// Convert a 2-component vector of normalized unsigned single/half-precision floating-point
/// values to two 16-bit integer values, then pack these 16-bit values into a
/// 32-bit unsigned integer.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packUnorm2x16(float2 unpackedValue)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "packUnorm2x16";
    case metal: __intrinsic_asm "pack_float_to_unorm2x16";
    case spirv:
        return spirv_asm
        {
            result:$$uint = OpExtInst glsl450 PackUnorm2x16 $unpackedValue;
        };
    case wgsl: __intrinsic_asm "pack2x16unorm";
    default:
        uint2 values = uint2(round(saturate(unpackedValue) * 65535.0));
        return __packUint2x16(values);
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packUnorm2x16(half2 unpackedValue)
{
    __target_switch
    {
    case metal: __intrinsic_asm "pack_half_to_unorm2x16";
    default:
        return packUnorm2x16(float2(unpackedValue));
    }
}

/// Convert a 2-component vector of normalized signed single/half-precision floating-point
/// values to two 16-bit integer values, then pack these 16-bit values into a
/// 32-bit unsigned integer.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packSnorm2x16(float2 unpackedValue)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "packSnorm2x16";
    case metal: __intrinsic_asm "pack_float_to_snorm2x16";
    case spirv:
        return spirv_asm
        {
            result:$$uint = OpExtInst glsl450 PackSnorm2x16 $unpackedValue;
        };
    case wgsl: __intrinsic_asm "pack2x16snorm";
    default:
        int2 values = int2(round(clamp(unpackedValue, -1.0, 1.0) * 32767.0)) & 0xFFFF;
        return __packInt2x16(values);
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packSnorm2x16(half2 unpackedValue)
{
    __target_switch
    {
    case metal: __intrinsic_asm "pack_half_to_snorm2x16";
    default:
        return packSnorm2x16(float2(unpackedValue));
    }
}

/// Convert a 2-component vector of IEEE-754 binary16 single/half-precision floating-point
/// values to two 16-bit integer values, then pack these 16-bit values into a
/// 32-bit unsigned integer.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packHalf2x16(float2 unpackedValue)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "packHalf2x16";
    case spirv:
        return spirv_asm
        {
            result:$$uint = OpExtInst glsl450 PackHalf2x16 $unpackedValue;
        };
    case wgsl: __intrinsic_asm "pack2x16float";
    default:
        uint2 values = f32tof16(unpackedValue);
        return __packUint2x16(values);
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
uint packHalf2x16(half2 unpackedValue)
{
    return packHalf2x16(float2(unpackedValue));
}
