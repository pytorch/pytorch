// TODO: These keywords are not recognized but they should be.
#define highp
#define mediump
#define lowp

#define VECTOR_MAP_UNARY(TYPE, COUNT, FUNC, VALUE) \
    vector<TYPE,COUNT> result; [ForceUnroll] for(int i = 0; i < COUNT; ++i) { result[i] = FUNC(VALUE[i]); } return result

#define VECTOR_MAP_TRINARY(TYPE, COUNT, FUNC, A, B, C) \
    vector<TYPE,COUNT> result; [ForceUnroll] for(int i = 0; i < COUNT; ++i) { result[i] = FUNC(A[i], B[i], C[i]); } return result

//
// OpenGL 4.60 spec
//

//
// Section 4.1. 'asic Types'
//

public typealias f16vec2 = half2;
public typealias f16vec3 = half3;
public typealias f16vec4 = half4;

public typealias vec2 = vector<float, 2>;
public typealias vec3 = vector<float, 3>;
public typealias vec4 = vector<float, 4>;

public typealias dvec2 = vector<double, 2>;
public typealias dvec3 = vector<double, 3>;
public typealias dvec4 = vector<double, 4>;

public typealias bvec2 = vector<bool, 2>;
public typealias bvec3 = vector<bool, 3>;
public typealias bvec4 = vector<bool, 4>;

public typealias ivec2 = vector<int, 2>;
public typealias ivec3 = vector<int, 3>;
public typealias ivec4 = vector<int, 4>;

public typealias uvec2 = vector<uint, 2>;
public typealias uvec3 = vector<uint, 3>;
public typealias uvec4 = vector<uint, 4>;

public typealias i8vec2 = vector<int8_t, 2>;
public typealias i8vec3 = vector<int8_t, 3>;
public typealias i8vec4 = vector<int8_t, 4>;

public typealias u8vec2 = vector<uint8_t, 2>;
public typealias u8vec3 = vector<uint8_t, 3>;
public typealias u8vec4 = vector<uint8_t, 4>;

public typealias i16vec2 = vector<int16_t, 2>;
public typealias i16vec3 = vector<int16_t, 3>;
public typealias i16vec4 = vector<int16_t, 4>;

public typealias u16vec2 = vector<uint16_t, 2>;
public typealias u16vec3 = vector<uint16_t, 3>;
public typealias u16vec4 = vector<uint16_t, 4>;

public typealias i64vec2 = vector<int64_t, 2>;
public typealias i64vec3 = vector<int64_t, 3>;
public typealias i64vec4 = vector<int64_t, 4>;

public typealias u64vec2 = vector<uint64_t, 2>;
public typealias u64vec3 = vector<uint64_t, 3>;
public typealias u64vec4 = vector<uint64_t, 4>;

public typealias mat2 = matrix<float, 2, 2>;
public typealias mat3 = matrix<float, 3, 3>;
public typealias mat4 = matrix<float, 4, 4>;

public typealias mat2x2 = matrix<float, 2, 2>;
public typealias mat2x3 = matrix<float, 2, 3>;
public typealias mat2x4 = matrix<float, 2, 4>;

public typealias mat3x2 = matrix<float, 3, 2>;
public typealias mat3x3 = matrix<float, 3, 3>;
public typealias mat3x4 = matrix<float, 3, 4>;

public typealias mat4x2 = matrix<float, 4, 2>;
public typealias mat4x3 = matrix<float, 4, 3>;
public typealias mat4x4 = matrix<float, 4, 4>;

public typealias dmat2 = matrix<double, 2, 2>;
public typealias dmat3 = matrix<double, 3, 3>;
public typealias dmat4 = matrix<double, 4, 4>;

public typealias dmat2x2 = matrix<double, 2, 2>;
public typealias dmat2x3 = matrix<double, 2, 3>;
public typealias dmat2x4 = matrix<double, 2, 4>;

public typealias dmat3x2 = matrix<double, 3, 2>;
public typealias dmat3x3 = matrix<double, 3, 3>;
public typealias dmat3x4 = matrix<double, 3, 4>;

public typealias dmat4x2 = matrix<double, 4, 2>;
public typealias dmat4x3 = matrix<double, 4, 3>;
public typealias dmat4x4 = matrix<double, 4, 4>;

public typealias f16mat2x2 = matrix<half, 2, 2>;
public typealias f16mat2x3 = matrix<half, 2, 3>;
public typealias f16mat2x4 = matrix<half, 2, 4>;


public out float4 gl_Position : SV_Position;
public out float gl_PointSize : SV_PointSize;
public in float2 gl_PointCoord : SV_PointCoord;

public in vec4 gl_FragCoord : SV_Position;
public out float gl_FragDepth : SV_Depth;
public out int gl_FragStencilRef : SV_StencilRef;

public in uvec3 gl_GlobalInvocationID : SV_DispatchThreadID;
public in uvec3 gl_WorkGroupID : SV_GroupID;
public in uint gl_LocalInvocationIndex : SV_GroupIndex;
public in uvec3 gl_LocalInvocationID : SV_GroupThreadID;

public property uint3 gl_NumWorkGroups {

    [require(glsl_spirv, GLSL_430_SPIRV_1_0_compute)]
    get 
    {
        __target_switch
        {
        case glsl:
            __intrinsic_asm "(gl_NumWorkGroups)";
        case spirv:
            return spirv_asm {
                    result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3);
                };
        }
    }
}

[require(compute)]
public property uint3 gl_WorkGroupSize
{
    [__unsafeForceInlineEarly]
    [require(compute)]
    get
    {
        return WorkgroupSize();
    }
}

// TODO: define overload for tessellation control stage.
public in int gl_InvocationID : SV_GSInstanceID;

internal in int __sv_InstanceIndex : SV_InstanceID;

// SPIRV InstanceIndex builtin for vertex shader
public property int gl_InstanceIndex
{
    [require(vertex)]
    get
    {
        __target_switch
        {
        default:
            return __sv_InstanceIndex;
        case glsl:
            __intrinsic_asm "gl_InstanceIndex";
        case spirv:
            return spirv_asm {
                result:$$int = OpLoad builtin(InstanceIndex:int);
            };
        }
    }
}
public in bool gl_FrontFacing : SV_IsFrontFace;

// TODO: define overload for geometry stage.
public in int gl_Layer : SV_RenderTargetArrayIndex;

public in int gl_SampleID : SV_SampleIndex;
public in int gl_VertexIndex : SV_VertexID;
public in int gl_ViewIndex : SV_ViewID;
public in int gl_ViewportIndex : SV_ViewportArrayIndex;
public in int gl_BaseVertex : SV_StartVertexLocation;
public in int gl_BaseInstance : SV_StartInstanceLocation;


// Override operator* behavior to compute algebric product of matrices and vectors.

[OverloadRank(15)]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public matrix<float, N, N> operator*<let N:int>(matrix<float, N, N> m1, matrix<float, N, N> m2)
{
    return mul(m2, m1);
}

[OverloadRank(15)]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public matrix<half, N, N> operator*<let N:int>(matrix<half, N, N> m1, matrix<half, N, N> m2)
{
    return mul(m2, m1);
}

[OverloadRank(15)]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public matrix<double, N, N> operator*<let N:int>(matrix<double, N, N> m1, matrix<double, N, N> m2)
{
    return mul(m2, m1);
}

[ForceInline]
[OverloadRank(15)]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public matrix<T, R, L> operator*<T:__BuiltinFloatingPointType, let L : int, let C : int, let R : int>(matrix<T, C, L> m1, matrix<T, R, C> m2)
{
    return mul(m2, m1);
}

[ForceInline]
[OverloadRank(15)]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public vector<T, R> operator*<T:__BuiltinFloatingPointType, let C : int, let R : int>(vector<T, C> v, matrix<T, R, C> m)
{
    return mul(m, v);
}

[ForceInline]
[OverloadRank(15)]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public vector<T, C> operator*<T:__BuiltinFloatingPointType, let C : int, let R : int>(matrix<T, R, C> m, vector<T, R> v)
{
    return mul(v, m);
}

__intrinsic_op(mul)
public matrix<T, N, M> matrixCompMult<T:__BuiltinFloatingPointType, let N:int, let M : int>(matrix<T,N,M> left, matrix<T,N,M> right);

__intrinsic_op(cmpLE)
public vector<bool, N> lessThanEqual<T, let N:int>(vector<T, N> x, vector<T, N> y);
__intrinsic_op(cmpLT)
public vector<bool, N> lessThan<T, let N:int>(vector<T, N> x, vector<T, N> y);
__intrinsic_op(cmpGT)
public vector<bool, N> greaterThan<T, let N:int>(vector<T, N> x, vector<T, N> y);
__intrinsic_op(cmpGE)
public vector<bool, N> greaterThanEqual<T, let N:int>(vector<T, N> x, vector<T, N> y);
__intrinsic_op(cmpEQ)
public vector<bool, N> equal<T, let N:int>(vector<T, N> x, vector<T, N> y);
__intrinsic_op(cmpNE)
public vector<bool, N> notEqual<T, let N:int>(vector<T, N> x, vector<T, N> y);

__generic<T>
public extension vector<T, 2>
{
    [ForceInline] public __init(vector<T, 3> bigger) { this = bigger.xy; }
    [ForceInline] public __init(vector<T, 4> bigger) { this = bigger.xy; }
}

__generic<T>
public extension vector<T, 3>
{
    [ForceInline] public __init(vector<T, 4> bigger) { this = bigger.xyz; }
}

[ForceInline]
[OverloadRank(15)]
[require(cpp_cuda_glsl_hlsl_spirv)]
public bool operator==<T:__BuiltinArithmeticType, let N:int>(vector<T, N> left, vector<T, N> right)
{
    return all(equal(left, right));
}

[ForceInline]
[OverloadRank(15)]
[require(cpp_cuda_glsl_hlsl_spirv)]
public bool operator!=<T:__BuiltinArithmeticType, let N:int>(vector<T, N> left, vector<T, N> right)
{
    return any(notEqual(left, right));
}

[ForceInline]
[OverloadRank(14)]
[require(cpp_cuda_glsl_hlsl_spirv)]
public bool operator==<T:__BuiltinFloatingPointType, let N:int>(vector<T, N> left, vector<T, N> right)
{
    return all(equal(left, right));
}

[ForceInline]
[OverloadRank(14)]
[require(cpp_cuda_glsl_hlsl_spirv)]
public bool operator!=<T:__BuiltinFloatingPointType, let N:int>(vector<T, N> left, vector<T, N> right)
{
    return any(notEqual(left, right));
}

[ForceInline]
[OverloadRank(14)]
[require(cpp_cuda_glsl_hlsl_spirv)]
public bool operator==<T:__BuiltinLogicalType, let N:int>(vector<T, N> left, vector<T, N> right)
{
    return all(equal(left, right));
}

[ForceInline]
[OverloadRank(14)]
[require(cpp_cuda_glsl_hlsl_spirv)]
public bool operator!=<T:__BuiltinLogicalType, let N:int>(vector<T, N> left, vector<T, N> right)
{
    return any(notEqual(left, right));
}

${{{{
for (auto type : kBaseTypes) {
    char const* typeName = type.name;
    if (!type.flags) continue;
}}}}
[ForceInline]
[OverloadRank(15)]
[require(cpp_cuda_glsl_hlsl_spirv)]
public bool operator==<let N:int>(vector<$(typeName), N> left, vector<$(typeName), N> right)
{
    return all(equal(left, right));
}

[ForceInline]
[OverloadRank(15)]
[require(cpp_cuda_glsl_hlsl_spirv)]
public bool operator!=<let N:int>(vector<$(typeName), N> left, vector<$(typeName), N> right)
{
    return any(notEqual(left, right));
}
${{{{
}
}}}}

/// Array length
__generic<T, let N : int>
public extension Array<T, N>
{
    [ForceInline]
    public int length()
    {
        return this.getCount();
    }
}

//
// Section 8.1. Angle and Trigonometry Functions
//

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public T atan(T y, T x)
{
    return atan2(y, x);
}

__generic<T : __BuiltinFloatingPointType, let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public vector<T,N> atan(vector<T,N> y, vector<T,N> x)
{
    return atan2(y, x);
}

//
// Section 8.2. Exponential Functions
//

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public T inversesqrt(T x)
{
    return rsqrt(x);
}

__generic<T : __BuiltinFloatingPointType, let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public vector<T, N> inversesqrt(vector<T, N> x)
{
    return rsqrt(x);
}

//
// Section 8.3. Common Functions
//

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
public T roundEven(T x)
{
    return rint(x);
}

__generic<T : __BuiltinFloatingPointType, let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv, sm_4_0_version)]
public vector<T,N> roundEven(vector<T,N> x)
{
    return rint(x);
}

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public T mod(T x, T y)
{
    // SPIR-V doesn't have "modulus".
    // All of Op?Mod and OpFRem are "remainder".

    __target_switch
    {
    case glsl: __intrinsic_asm "mod";
    default:
        return x - y * floor(x / y);
    }
}

__generic<T : __BuiltinFloatingPointType, let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public vector<T, N> mod(vector<T, N> x, T y)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "mod";
    default:
        return x - y * floor(x / y);
    }
}

__generic<T : __BuiltinFloatingPointType, let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public vector<T, N> mod(vector<T, N> x, vector<T, N> y)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "mod";
    default:
        return x - y * floor(x / y);
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public vector<T,N> min(vector<T,N> x, T y)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "min";
    default:
        return min(x, vector<T,N>(y));
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public vector<T,N> max(vector<T,N> x, T y)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "max";
    default:
        return max(x, vector<T,N>(y));
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public vector<T,N> clamp(vector<T,N> x, T minBound, T maxBound)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "clamp";
    default:
        return clamp(x, vector<T,N>(minBound), vector<T,N>(maxBound));
    }
}

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public T mix(T x, T y, T a)
{
    return lerp(x, y, a);
}

__generic<T : __BuiltinFloatingPointType, let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public vector<T, N> mix(vector<T, N> x, vector<T, N> y, T a)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "mix";
    default:
        return mix(x, y, vector<T, N>(a));
    }
}

__generic<T : __BuiltinFloatingPointType, let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public vector<T, N> mix(vector<T, N> x, vector<T, N> y, vector<T, N> a)
{
    return lerp(x, y, a);
}

__generic<T>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public T mix(T x, T y, bool a)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "mix";
    case spirv: return spirv_asm {
        result:$$T = OpSelect $a $y $x
    };
    default:
        return (a ? y : x);
    }
}

__generic<T, let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, sm_4_0_version)]
public vector<T, N> mix(vector<T, N> x, vector<T, N> y, vector<bool, N> a)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "mix";
    case spirv: return spirv_asm {
        result:$$vector<T,N> = OpSelect $a $y $x
    };
    default:
        vector<T, N> result;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
        {
            result[i] = (a[i] ? y[i] : x[i]);
        }
        return result;
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
public int floatBitsToInt(highp float x)
{
    return asint(x);
}

__generic<let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
public vector<int, N> floatBitsToInt(highp vector<float, N> x)
{
    return asint(x);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
public uint floatBitsToUint(highp float x)
{
    return asuint(x);
}

__generic<let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
public vector<uint, N> floatBitsToUint(highp vector<float, N> x)
{
    return asuint(x);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
public float intBitsToFloat(highp int x)
{
    return asfloat(x);
}

__generic<let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
public vector<float, N> intBitsToFloat(highp vector<int, N> x)
{
    return asfloat(x);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
public float uintBitsToFloat(highp uint x)
{
    return asfloat(x);
}

__generic<let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
public vector<float, N> uintBitsToFloat(highp vector<uint, N> x)
{
    return asfloat(x);
}

//
// Section 8.4. Floating-Point Pack and Unpack Functions
//

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
public vec2 unpackUnorm2x16(uint p)
{
    return unpackUnorm2x16ToFloat(p);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
public vec2 unpackSnorm2x16(uint p)
{
    return unpackSnorm2x16ToFloat(p);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
public vec4 unpackUnorm4x8(highp uint p)
{
    return unpackUnorm4x8ToFloat(p);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
public vec4 unpackSnorm4x8(highp uint p)
{
    return unpackSnorm4x8ToFloat(p);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
public float half2float(uint h)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "half2float";
    default:
        uint s = ((h & uint(0x8000)) << uint(16));
        uint e = 0;
        uint m = ((h & uint(0x03ff)) << uint(13));
        if (m != 0)
        {
            e = (((h & uint(0x7c00)) + uint(0x1c000)) << uint(13));
        }
        return uintBitsToFloat(s | e | m); 
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, pack_vector)]
public vec2 unpackHalf2x16(uint p)
{
    return unpackHalf2x16ToFloat(p);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
public double packDouble2x32(uvec2 v)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "packDouble2x32";
    case spirv: return spirv_asm {
        result:$$double = OpExtInst glsl450 PackDouble2x32 $v
    };
    default:
        // TODO: there is no "asdouble()"
        //return asdouble(uint64_t(v.x) | (uint64_t(v.y) << 32));
        return 0.0;
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_4_0)]
public uvec2 unpackDouble2x32(double v)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "unpackDouble2x32";
    case spirv: return spirv_asm {
        result:$$uvec2 = OpExtInst glsl450 UnpackDouble2x32 $v
    };
    default:
        // TODO: there is no "asuint64()"
        uint64_t u = 0; // asuint64(v);
        return uvec2(uint(u & 0xFFFFFFFF), uint(u >> 32));
    }
}

/// Unpack a 32-bit unsigned integer into 4 8-bit unsigned integers.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, pack_vector)]
__intrinsic_op($(kIROp_BitCast))
public u8vec4 unpack8(uint32_t u);

/// Unpack a 32-bit signed integer into 4 8-bit signed integers.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, pack_vector)]
__intrinsic_op($(kIROp_BitCast))
public i8vec4 unpack8(int32_t u);

/// Unpack a 16-bit unsigned integer into 2 8-bit unsigned integers.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, pack_vector)]
__intrinsic_op($(kIROp_BitCast))
public u8vec2 unpack8(uint16_t u);

/// Unpack a 16-bit signed integer into 2 8-bit signed integers.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, pack_vector)]
__intrinsic_op($(kIROp_BitCast))
public i8vec2 unpack8(int16_t u);

/// Unpack a 32-bit unsigned integer into 2 16-bit unsigned integers.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, pack_vector)]
__intrinsic_op($(kIROp_BitCast))
public u16vec2 unpack16(uint32_t u);

/// Unpack a 32-bit signed integer into 2 16-bit signed integers.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, pack_vector)]
__intrinsic_op($(kIROp_BitCast))
public i16vec2 unpack16(int32_t u);

/// Pack 2 16-bit unsigned integers into a 32-bit unsigned integer.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, pack_vector)]
__intrinsic_op($(kIROp_BitCast))
public uint32_t pack32(u16vec2 v);

/// Pack 2 16-bit signed integers into a 32-bit signed integer.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, pack_vector)]
__intrinsic_op($(kIROp_BitCast))
public int32_t pack32(i16vec2 v);

/// Pack 4 8-bit unsigned integers into a 32-bit unsigned integer.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, pack_vector)]
__intrinsic_op($(kIROp_BitCast))
public uint32_t pack32(u8vec4 v);

/// Pack 4 8-bit signed integers into a 32-bit signed integer.
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, pack_vector)]
__intrinsic_op($(kIROp_BitCast))
public int32_t pack32(i8vec4 v);

//
// Section 8.5. Geometric Functions
//

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, GLSL_400)]
public T faceforward(T n, T i, T ng)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "faceforward";
    case spirv: return spirv_asm {
        OpExtInst $$T result glsl450 FaceForward $n $i $ng
    };
    default:
        return dot(ng, i) < T(0.0f) ? n : -n;
    }
}

//
// Section 8.6. Matrix Functions
//

__generic<T : __BuiltinFloatingPointType, let C : int, let R : int>
[__readNone]
[ForceInline]
[OverloadRank(15)]
[require(cpp_cuda_glsl_hlsl_spirv, GLSL_400)]
public matrix<T, R, C> outerProduct(vector<T, C> c, vector<T, R> r)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "outerProduct";

    // Note: SPIR-V takes the input arguments in an opposite order
    // compared to GLSL. SPIR-V spec document says,
    // "Its (second argument) number of components must equal the
    //  number of columns in Result Type."
    //
    case spirv: return spirv_asm {
        result:$$matrix<T,R,C> = OpOuterProduct $c $r
    };
    default:
        matrix<T, R, C> result;
        for (int j = 0; j < R; ++j)
        {
            for (int i = 0; i < C; ++i)
            {
                result[j][i] = c[i] * r[j];
            }
        }
        return result;
    }
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[require(glsl_spirv, GLSL_400)]
public matrix<T,N,N> inverse(matrix<T,N,N> m)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "inverse";
    case spirv: return spirv_asm {
        OpExtInst $$matrix<T,N,N> result glsl450 MatrixInverse $m
    };
    }
}

//
// Section 8.8. Integer Functions
//

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, GLSL_400)]
public uint uaddCarry(highp uint x, highp uint y, out lowp uint carry)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "uaddCarry";
    case spirv: return spirv_asm {
        %ResType = OpTypeStruct $$uint $$uint;
        %temp:%ResType = OpIAddCarry $x $y;
        %carry:$$uint = OpCompositeExtract %temp 1;
        OpStore &carry %carry;
        result:$$uint = OpCompositeExtract %temp 0
    };
    default:
        let result = x * y;
        carry = ((result < x || result < y) ? 1 : 0);
        return result;
    }
}

__generic<let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, GLSL_400)]
public vector<uint,N> uaddCarry(highp vector<uint,N> x, highp vector<uint,N> y, out lowp vector<uint,N> carry)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "uaddCarry";
    case spirv: return spirv_asm {
        %ResType = OpTypeStruct $$vector<uint,N> $$vector<uint,N>;
        %temp:%ResType = OpIAddCarry $x $y;
        %carry:$$vector<uint,N> = OpCompositeExtract %temp 1;
        OpStore &carry %carry;
        result:$$vector<uint,N> = OpCompositeExtract %temp 0
    };
    default:
        VECTOR_MAP_TRINARY(uint, N, uaddCarry, x, y, carry);
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, GLSL_400)]
public uint usubBorrow(highp uint x, highp uint y, out lowp uint borrow)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "usubBorrow";
    case spirv: return spirv_asm {
        %ResType = OpTypeStruct $$uint $$uint;
        %temp:%ResType = OpISubBorrow $x $y;
        %borrow:$$uint = OpCompositeExtract %temp 1;
        OpStore &borrow %borrow;
        result:$$uint = OpCompositeExtract %temp 0
    };
    default:
        borrow = (y > x) ? 1 : 0;
        return x - y;
    }
}

__generic<let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, GLSL_400)]
public vector<uint,N> usubBorrow(highp vector<uint,N> x, highp vector<uint,N> y, out lowp vector<uint,N> borrow)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "usubBorrow";
    case spirv: return spirv_asm {
        %ResType = OpTypeStruct $$vector<uint,N> $$vector<uint,N>;
        %temp:%ResType = OpISubBorrow $x $y;
        %borrow:$$vector<uint,N> = OpCompositeExtract %temp 1;
        OpStore &borrow %borrow;
        result:$$vector<uint,N> = OpCompositeExtract %temp 0
    };
    default:
        VECTOR_MAP_TRINARY(uint, N, usubBorrow, x, y, borrow);
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, GLSL_400)]
public void umulExtended(highp uint x, highp uint y, out highp uint msb, out highp uint lsb)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "umulExtended";
    case spirv: spirv_asm {
        %ResType = OpTypeStruct $$uint $$uint;
        %temp:%ResType = OpUMulExtended $x $y;
        %lsb:$$uint = OpCompositeExtract %temp 0;
        %msb:$$uint = OpCompositeExtract %temp 1;
        OpStore &lsb %lsb;
        OpStore &msb %msb;
    };
    default:
        uint64_t result = x * y;
        msb = uint(result >> 32);
        lsb = uint(result);
    }
}

__generic<let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, GLSL_400)]
public void umulExtended(highp vector<uint,N> x, highp vector<uint,N> y, out highp vector<uint,N> msb, out highp vector<uint,N> lsb)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "umulExtended";
    case spirv: spirv_asm {
        %ResType = OpTypeStruct $$vector<uint,N> $$vector<uint,N>;
        %temp:%ResType = OpUMulExtended $x $y;
        %lsb:$$vector<uint,N> = OpCompositeExtract %temp 0;
        %msb:$$vector<uint,N> = OpCompositeExtract %temp 1;
        OpStore &lsb %lsb;
        OpStore &msb %msb;
    };
    default:
        [ForceUnroll]
        for(int i = 0; i < N; ++i)
        {
            umulExtended(x[i], y[i], msb[i], lsb[i]);
        }
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, GLSL_400)]
public void imulExtended(highp int x, highp int y, out highp int msb, out highp int lsb)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "imulExtended";
    case spirv: spirv_asm {
        %ResType = OpTypeStruct $$int $$int;
        %temp:%ResType = OpSMulExtended $x $y;
        %lsb:$$int = OpCompositeExtract %temp 0;
        %msb:$$int = OpCompositeExtract %temp 1;
        OpStore &lsb %lsb;
        OpStore &msb %msb;
    };
    default:
        int64_t result = x * y;
        msb = int(result >> 32);
        lsb = int(result);
    }
}

__generic<let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, GLSL_400)]
public void imulExtended(highp vector<int,N> x, highp vector<int,N> y, out highp vector<int,N> msb, out highp vector<int,N> lsb)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "imulExtended";
    case spirv: spirv_asm {
        %ResType = OpTypeStruct $$vector<int,N> $$vector<int,N>;
        %temp:%ResType = OpSMulExtended $x $y;
        %lsb:$$vector<int,N> = OpCompositeExtract %temp 0;
        %msb:$$vector<int,N> = OpCompositeExtract %temp 1;
        OpStore &lsb %lsb;
        OpStore &msb %msb;
    };
    default:
        [ForceUnroll]
        for(int i = 0; i < N; ++i)
        {
           imulExtended(x[i], y[i], msb[i], lsb[i]);
        }
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, GLSL_400)]
public int bitfieldReverse(highp int value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "bitfieldReverse";
    case spirv: return spirv_asm {
        result:$$int = OpBitReverse $value
    };
    default:
        value = ((value & 0xAAAAAAAA) >> 1) | ((value & 0x55555555) << 1);
        value = ((value & 0xCCCCCCCC) >> 2) | ((value & 0x33333333) << 2);
        value = ((value & 0xF0F0F0F0) >> 4) | ((value & 0x0F0F0F0F) << 4);
        value = ((value & 0xFF00FF00) >> 8) | ((value & 0x00FF00FF) << 8);
        value = ((value & 0xFFFF0000) >> 16) | ((value & 0x0000FFFF) << 16);
        return value;
    }
}

__generic<let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, GLSL_400)]
public vector<int,N> bitfieldReverse(highp vector<int,N> value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "bitfieldReverse";
    case spirv: return spirv_asm {
        result:$$vector<int,N> = OpBitReverse $value
    };
    default:
        VECTOR_MAP_UNARY(int, N, bitfieldReverse, value);
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, GLSL_400)]
public uint bitfieldReverse(highp uint value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "bitfieldReverse";
    case spirv: return spirv_asm {
        result:$$uint = OpBitReverse $value
    };
    default:
        value = ((value & 0xAAAAAAAA) >> 1) | ((value & 0x55555555) << 1);
        value = ((value & 0xCCCCCCCC) >> 2) | ((value & 0x33333333) << 2);
        value = ((value & 0xF0F0F0F0) >> 4) | ((value & 0x0F0F0F0F) << 4);
        value = ((value & 0xFF00FF00) >> 8) | ((value & 0x00FF00FF) << 8);
        value = ((value & 0xFFFF0000) >> 16) | ((value & 0x0000FFFF) << 16);
        return value;
    }
}

__generic<let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, GLSL_400)]
public vector<uint,N> bitfieldReverse(highp vector<uint,N> value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "bitfieldReverse";
    case spirv: return spirv_asm {
        result:$$vector<uint,N> = OpBitReverse $value
    };
    default:
        VECTOR_MAP_UNARY(int, N, bitfieldReverse, value);
    }
}

[__readNone] 
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
public uint bitCount(uint value)
{
    return countbits(value);
}

__generic<let N:int>
[__readNone] 
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
public vector<uint,N> bitCount(vector<uint,N> value)
{
    return countbits(value);
}

[__readNone] 
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
public int bitCount(int value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "bitCount";
    case spirv: return spirv_asm {
        result:$$int = OpBitCount $value
    };
    default:
        return countbits(uint(value));
    }
}
    
__generic<let N:int>
[__readNone] 
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
public vector<int,N> bitCount(vector<int,N> value)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "bitCount";
    case spirv: return spirv_asm {
        result:$$vector<int,N> = OpBitCount $value
    };
    default:
        VECTOR_MAP_UNARY(int, N, countbits, value);
    }
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
public int findLSB(int v)
{
    return firstbitlow(v);
}

__generic<let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
public vector<int,N> findLSB(vector<int,N> value)
{
    return firstbitlow(value);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
public uint findLSB(uint v)
{
    return firstbitlow(v);
}

__generic<let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
public vector<uint,N> findLSB(vector<uint,N> value)
{
    return firstbitlow(value);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
public int findMSB(int value)
{
    return firstbithigh(value);
}

__generic<let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
public vector<int,N> findMSB(vector<int,N> value)
{
    return firstbithigh(value);
}

[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
public uint findMSB(uint value)
{
    return firstbithigh(value);
}

__generic<let N:int>
[__readNone]
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, shader5_sm_5_0)]
public vector<uint,N> findMSB(vector<uint,N> value)
{
    return firstbithigh(value);
}

__generic<let N:int>
[__readNone]
[ForceInline]
public vector<bool,N> not(vector<bool,N> x)
{
    return !x;
}

__intrinsic_op(vectorReshape)
vector<T,N> __vectorReshape2<T, let N : int, U>(U vin);

//
// Section 8.9.1. Texture Query Functions
//

public typealias usampler1D = Sampler1D<uint4>;
public typealias isampler1D = Sampler1D<int4>;
public typealias sampler1D = Sampler1D<float4>;

public typealias usampler2D = Sampler2D<uint4>;
public typealias isampler2D = Sampler2D<int4>;
public typealias sampler2D = Sampler2D<float4>;

public typealias usampler3D = Sampler3D<uint4>;
public typealias isampler3D = Sampler3D<int4>;
public typealias sampler3D = Sampler3D<float4>;

public typealias usamplerCube = SamplerCube<uint4>;
public typealias isamplerCube = SamplerCube<int4>;
public typealias samplerCube = SamplerCube<float4>;

__generic<let sampleCount:int=0, let format:int=0>
public typealias sampler1DShadow = _Texture<
    float,
    __Shape1D,
    0, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    1, // isCombined
    format
>;

__generic<let sampleCount:int=0, let format:int=0>
public typealias sampler2DShadow = _Texture<
    float,
    __Shape2D,
    0, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    1, // isCombined
    format
>;

__generic<let sampleCount:int=0, let format:int=0>
public typealias samplerCubeShadow = _Texture<
    float,
    __ShapeCube,
    0, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    1, // isCombined
    format
>;

public typealias usampler1DArray = Sampler1DArray<uint4>;
public typealias isampler1DArray = Sampler1DArray<int4>;
public typealias sampler1DArray = Sampler1DArray<float4>;

public typealias usampler2DArray = Sampler2DArray<uint4>;
public typealias isampler2DArray = Sampler2DArray<int4>;
public typealias sampler2DArray = Sampler2DArray<float4>;

public typealias usamplerCubeArray = SamplerCubeArray<uint4>;
public typealias isamplerCubeArray = SamplerCubeArray<int4>;
public typealias samplerCubeArray = SamplerCubeArray<float4>;

__generic<let sampleCount:int=0, let format:int=0>
public typealias sampler1DArrayShadow = _Texture<
    float,
    __Shape1D,
    1, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    1, // isCombined
    format
>;

__generic<let sampleCount:int=0, let format:int=0>
public typealias sampler2DArrayShadow = _Texture<
    float,
    __Shape2D,
    1, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    1, // isCombined
    format
>;

__generic<let sampleCount:int=0, let format:int=0>
public typealias samplerCubeArrayShadow = _Texture<
    float,
    __ShapeCube,
    1, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    1, // isCombined
    format
>;

public typealias sampler2DMS = Sampler2DMS<float4>;
public typealias isampler2DMS = Sampler2DMS<int4>;
public typealias usampler2DMS = Sampler2DMS<uint4>;

public typealias sampler2DMSArray = Sampler2DMSArray<float4>;
public typealias isampler2DMSArray = Sampler2DMSArray<int4>;
public typealias usampler2DMSArray = Sampler2DMSArray<uint4>;

__generic<T:ITexelElement=float4, let sampleCount:int=0, let format:int=0>
public typealias Sampler2DRect = _Texture<T, __Shape2D, 0, 0, sampleCount, 0, 0, 1, format>;
public typealias sampler2DRect = Sampler2DRect<float4>;
public typealias isampler2DRect = Sampler2DRect<int4>;
public typealias usampler2DRect = Sampler2DRect<uint4>;

__generic<let sampleCount:int=0, let format:int=0>
public typealias sampler2DRectShadow = _Texture<
    float,
    __Shape2D,
    0, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    1, // isCombined
    format
>;

__generic<T:ITexelElement, let format:int=0>
public typealias SamplerBuffer = _Texture<
    T,
    __ShapeBuffer,
    0, // isArray
    0, // isMS
    0, // sampleCount
    1, // RW
    0, // isShadow
    0, // isCombined
    format
>;
public typealias samplerBuffer = SamplerBuffer<vec4>;
public typealias isamplerBuffer = SamplerBuffer<int4>;
public typealias usamplerBuffer = SamplerBuffer<uint4>;


// -------------------
// textureSize
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_size)]
public int textureSize(Sampler1D<T> sampler, int lod)
{
    int result;
    int numberOfLevels;
    sampler.GetDimensions(lod, result, numberOfLevels);
    return result;
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_size)]
public ivec2 textureSize(Sampler2D<T> sampler, int lod)
{
    vector<int,2> result;
    int numberOfLevels;
    sampler.GetDimensions(lod, result.x, result.y, numberOfLevels);
    return result;
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_size)]
public ivec3 textureSize(Sampler3D<T> sampler, int lod)
{
    vector<int,3> result;
    int numberOfLevels;
    sampler.GetDimensions(lod, result.x, result.y, result.z, numberOfLevels);
    return result;
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_size)]
public ivec2 textureSize(SamplerCube<T> sampler, int lod)
{
    vector<int,2> result;
    int numberOfLevels;
    sampler.GetDimensions(lod, result.x, result.y, numberOfLevels);
    return result;
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_size)]
public int textureSize(sampler1DShadow sampler, int lod)
{
    int result;
    int numberOfLevels;
    sampler.GetDimensions(lod, result, numberOfLevels);
    return result;
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_size)]
public ivec2 textureSize(sampler2DShadow sampler, int lod)
{
    vector<int,2> result;
    int numberOfLevels;
    sampler.GetDimensions(lod, result.x, result.y, numberOfLevels);
    return result;
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_size)]
public ivec2 textureSize(samplerCubeShadow sampler, int lod)
{
    vector<int,2> result;
    int numberOfLevels;
    sampler.GetDimensions(lod, result.x, result.y, numberOfLevels);
    return result;
}

[require(glsl_hlsl_spirv, texture_size)]
__generic<T:ITexelElement>
[ForceInline]
public ivec3 textureSize(SamplerCubeArray<T> sampler, int lod)
{
    vector<int,3> result;
    int numberOfLevels;
    sampler.GetDimensions(lod, result.x, result.y, result.z, numberOfLevels);
    return result;
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_size)]
public ivec3 textureSize(samplerCubeArrayShadow sampler, int lod)
{
    vector<int,3> result;
    int numberOfLevels;
    sampler.GetDimensions(lod, result.x, result.y, result.z, numberOfLevels);
    return result;
}

[require(glsl_hlsl_spirv, texture_size)]
__generic<T:ITexelElement>
[ForceInline]
public ivec2 textureSize(Sampler2DRect<T> sampler)
{
    vector<int,2> result;
    int numberOfLevels;
    sampler.GetDimensions(0, result.x, result.y, numberOfLevels);
    return result;
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_size)]
public ivec2 textureSize(sampler2DRectShadow sampler)
{
    vector<int,2> result;
    int numberOfLevels;
    sampler.GetDimensions(result.x, result.y);
    return result;
}

[require(glsl_hlsl_spirv, texture_size)]
__generic<T:ITexelElement>
[ForceInline]
public ivec2 textureSize(Sampler1DArray<T> sampler, int lod)
{
    vector<int,2> result;
    int numberOfLevels;
    sampler.GetDimensions(lod, result.x, result.y, numberOfLevels);
    return result;
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_size)]
public ivec2 textureSize(sampler1DArrayShadow sampler, int lod)
{
    vector<int,2> result;
    int numberOfLevels;
    sampler.GetDimensions(lod, result.x, result.y, numberOfLevels);
    return result;
}

[require(glsl_hlsl_spirv, texture_size)]
__generic<T:ITexelElement>
[ForceInline]
public ivec3 textureSize(Sampler2DArray<T> sampler, int lod)
{
    vector<int,3> result;
    int numberOfLevels;
    sampler.GetDimensions(lod, result.x, result.y, result.z, numberOfLevels);
    return result;
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_size)]
public ivec3 textureSize(sampler2DArrayShadow sampler, int lod)
{
    vector<int,3> result;
    int numberOfLevels;
    sampler.GetDimensions(lod, result.x, result.y, result.z, numberOfLevels);
    return result;
}

[require(glsl_hlsl_spirv, texture_size)]
__generic<T:ITexelElement, let format:int>
[ForceInline]
public int textureSize(SamplerBuffer<T,format> sampler)
{
    uint result;
    sampler.GetDimensions(result);
    return int(result);
}

[require(glsl_hlsl_spirv, texture_size)]
__generic<T:ITexelElement, let sampleCount:int>
[ForceInline]
public ivec2 textureSize(Sampler2DMS<T,sampleCount> sampler)
{
    vector<int,2> result;
    int sampleCount;
    int numberOfLevels;
    sampler.GetDimensions(result.x, result.y, sampleCount);
    return result;
}

[require(glsl_hlsl_spirv, texture_size)]
__generic<T:ITexelElement, let sampleCount:int>
[ForceInline]
public ivec3 textureSize(Sampler2DMSArray<T,sampleCount> sampler)
{
    vector<int,3> result;
    int sampleCount;
    int numberOfLevels;
    sampler.GetDimensions(result.x, result.y, result.z, sampleCount);
    return result;
}

// -------------------
// textureQueryLod
// -------------------

__generic<T:ITexelElement, let isArray:int, let sampleCount:int, let isShadow:int, let format:int>
[ForceInline]
[require(glsl_hlsl_metal_spirv, texture_querylod)]
public vec2 textureQueryLod(_Texture<
        T,
        __Shape1D,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        isShadow,
        1, // isCombined
        format
    > sampler, float p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "textureQueryLod";
    case spirv:
        return spirv_asm {
            OpCapability ImageQuery;
            result:$$float2 = OpImageQueryLod $sampler $p
        };
    default:
        return vec2(
            sampler.CalculateLevelOfDetail(p),
            sampler.CalculateLevelOfDetailUnclamped(p)
            );
    }
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let isShadow:int, let format:int>
[ForceInline]
[require(glsl_hlsl_metal_spirv, texture_querylod)]
public vec2 textureQueryLod(_Texture<
        T,
        Shape,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        isShadow,
        1, // isCombined
        format
    > sampler, vector<float,Shape.dimensions> p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "textureQueryLod";
    case spirv:
        return spirv_asm {
            OpCapability ImageQuery;
            result:$$float2 = OpImageQueryLod $sampler $p
        };
    default:
        return vec2(
            sampler.CalculateLevelOfDetail(p),
            sampler.CalculateLevelOfDetailUnclamped(p)
            );
    }
}

// -------------------
// textureQueryLevels
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_metal_spirv, texture_querylevels)]
public int textureQueryLevels(Sampler1D<T> sampler)
{
    int width;
    int numberOfLevels;
    sampler.GetDimensions(0, width, numberOfLevels);
    return numberOfLevels;
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_metal_spirv, texture_querylevels)]
public int textureQueryLevels(Sampler2D<T> sampler)
{
    vector<int,2> dim;
    int numberOfLevels;
    sampler.GetDimensions(0, dim.x, dim.y, numberOfLevels);
    return numberOfLevels;
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_metal_spirv, texture_querylevels)]
public int textureQueryLevels(Sampler3D<T> sampler)
{
    vector<int,3> dim;
    int numberOfLevels;
    sampler.GetDimensions(0, dim.x, dim.y, dim.z, numberOfLevels);
    return numberOfLevels;
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_metal_spirv, texture_querylevels)]
public int textureQueryLevels(SamplerCube<T> sampler)
{
    vector<int,2> dim;
    int numberOfLevels;
    sampler.GetDimensions(0, dim.x, dim.y, numberOfLevels);
    return numberOfLevels;
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_metal_spirv, texture_querylevels)]
public int textureQueryLevels(Sampler1DArray<T> sampler)
{
    vector<int,2> dim;
    int numberOfLevels;
    sampler.GetDimensions(0, dim.x, dim.y, numberOfLevels);
    return numberOfLevels;
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_metal_spirv, texture_querylevels)]
public int textureQueryLevels(Sampler2DArray<T> sampler)
{
    vector<int,3> dim;
    int numberOfLevels;
    sampler.GetDimensions(0, dim.x, dim.y, dim.z, numberOfLevels);
    return numberOfLevels;
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_metal_spirv, texture_querylevels)]
public int textureQueryLevels(SamplerCubeArray<T> sampler)
{
    vector<int,3> dim;
    int numberOfLevels;
    sampler.GetDimensions(0, dim.x, dim.y, dim.z, numberOfLevels);
    return numberOfLevels;
}

[ForceInline]
[require(cpp_glsl_hlsl_metal_spirv, texture_querylevels)]
public int textureQueryLevels(sampler1DShadow sampler)
{
    int dim;
    int numberOfLevels;
    sampler.GetDimensions(0, dim, numberOfLevels);
    return numberOfLevels;
}

[ForceInline]
[require(cpp_glsl_hlsl_metal_spirv, texture_querylevels)]
public int textureQueryLevels(sampler2DShadow sampler)
{
    vector<int,2> dim;
    int numberOfLevels;
    sampler.GetDimensions(0, dim.x, dim.y, numberOfLevels);
    return numberOfLevels;
}

[ForceInline]
[require(cpp_glsl_hlsl_metal_spirv, texture_querylevels)]
public int textureQueryLevels(samplerCubeShadow sampler)
{
    vector<int,2> dim;
    int numberOfLevels;
    sampler.GetDimensions(0, dim.x, dim.y, numberOfLevels);
    return numberOfLevels;
}

[ForceInline]
[require(cpp_glsl_hlsl_metal_spirv, texture_querylevels)]
public int textureQueryLevels(sampler1DArrayShadow sampler)
{
    vector<int,2> dim;
    int numberOfLevels;
    sampler.GetDimensions(0, dim.x, dim.y, numberOfLevels);
    return numberOfLevels;
}

[ForceInline]
[require(cpp_glsl_hlsl_metal_spirv, texture_querylevels)]
public int textureQueryLevels(sampler2DArrayShadow sampler)
{
    vector<int,3> dim;
    int numberOfLevels;
    sampler.GetDimensions(0, dim.x, dim.y, dim.z, numberOfLevels);
    return numberOfLevels;
}

[ForceInline]
[require(cpp_glsl_hlsl_metal_spirv, texture_querylevels)]
public int textureQueryLevels(samplerCubeArrayShadow sampler)
{
    vector<int,3> dim;
    int numberOfLevels;
    sampler.GetDimensions(0, dim.x, dim.y, dim.z, numberOfLevels);
    return numberOfLevels;
}

// -------------------
// textureSamples
// -------------------

__generic<T:ITexelElement, let sampleCount:int>
[ForceInline]
[require(glsl_hlsl_spirv, image_samples)]
public int textureSamples(Sampler2DMS<T,sampleCount> sampler)
{
    vector<int,2> dim;
    int sampleCount;
    int numberOfLevels;
    sampler.GetDimensions( dim.x, dim.y, sampleCount);
    return sampleCount;
}

__generic<T:ITexelElement, let sampleCount:int>
[ForceInline]
[require(glsl_hlsl_spirv, image_samples)]
public int textureSamples(Sampler2DMSArray<T,sampleCount> sampler)
{
    vector<int,3> dim;
    int sampleCount;
    int numberOfLevels;
    sampler.GetDimensions(dim.x, dim.y, dim.z, sampleCount);
    return sampleCount;
}

//
// Section 8.9.2. Texel Lookup Functions
//

// -------------------
// texture
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> texture(Sampler1D<T> sampler, float p)
{
    return __vectorReshape2<T.Element, 4>(sampler.Sample(p));
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> texture(Sampler1D<T> sampler, float p, constexpr float bias)
{
    return __vectorReshape2<T.Element, 4>(sampler.SampleBias(p, bias));
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let format:int>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> texture(_Texture<
        T,
        Shape,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        0, // isShadow
        1, // isCombined
        format
    > sampler, vector<float,Shape.dimensions+isArray> p)
{
    return __vectorReshape2<T.Element, 4>(sampler.Sample(p));
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let format:int>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> texture(_Texture<
        T,
        Shape,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        0, // isShadow
        1, // isCombined
        format
    > sampler, vector<float,Shape.dimensions+isArray> p, constexpr float bias)
{
    return __vectorReshape2<T.Element,4>(sampler.SampleBias(p, bias));
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float texture(sampler1DShadow sampler, vec3 p)
{
    return sampler.SampleCmp(p.x, p.z);
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float texture(sampler1DShadow sampler, vec3 p, float bias)
{
    float location = p.x;
    float compareValue = p.z;

    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "texture";
    case spirv:
        return spirv_asm
        {
            result:$$float = OpImageSampleDrefImplicitLod $sampler $location $compareValue Bias $bias;
        };
    default:
        // TODO: Need to apply bias
        return sampler.SampleCmp(location, compareValue);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float texture(sampler2DShadow sampler, vec3 p)
{
    return sampler.SampleCmp(p.xy, p.z);
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float texture(sampler2DShadow sampler, vec3 p, float bias)
{
    vec2 location = p.xy;
    float compareValue = p.z;

    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "texture";
    case spirv:
        return spirv_asm
        {
            result:$$float = OpImageSampleDrefImplicitLod $sampler $location $compareValue Bias $bias;
        };
    default: 
        // TODO: Need to apply bias
        return sampler.SampleCmp(location, compareValue);
    }
}

[require(glsl_hlsl_spirv, texture_shadowlod)]
[ForceInline]
public float texture(samplerCubeShadow sampler, vec4 p)
{
    return sampler.SampleCmp(p.xyz, p.w);
}

[require(glsl_hlsl_spirv, texture_shadowlod)]
[ForceInline]
public float texture(samplerCubeShadow sampler, vec4 p, float bias)
{
    vec3 location = p.xyz;
    float compareValue = p.w;

    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "texture";
    case spirv:
        return spirv_asm
        {
            result:$$float = OpImageSampleDrefImplicitLod $sampler $location $compareValue Bias $bias;
        };
    default: 
        // TODO: Need to apply bias
        return sampler.SampleCmp(location, compareValue);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float texture(sampler1DArrayShadow sampler, vec3 p)
{
    return sampler.SampleCmp(p.xy, p.z);
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float texture(sampler1DArrayShadow sampler, vec3 p, float bias)
{
    vec2 location = p.xy;
    float compareValue = p.z;

    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "texture";
    case spirv:
        return spirv_asm
        {
            result:$$float = OpImageSampleDrefImplicitLod $sampler $location $compareValue Bias $bias;
        };
    default: 
        // TODO: Need to apply bias
        return sampler.SampleCmp(location, compareValue);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float texture(sampler2DArrayShadow sampler, vec4 p)
{
    return sampler.SampleCmp(p.xyz, p.w);
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float texture(samplerCubeArrayShadow sampler, vec4 p, float compare)
{
    return sampler.SampleCmp(p, compare);
}

// -------------------
// textureProj
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProj(Sampler1D<T> sampler, vec2 p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProj";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p
    };
    default:
        return texture(sampler, p.x / p.y);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProj(Sampler1D<T> sampler, vec2 p, float bias)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProj";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p Bias $bias
    };
    default:
        return texture(sampler, p.x / p.y, bias);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProj(Sampler1D<T> sampler, vec4 p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProj";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p
    };
    default:
        return texture(sampler, p.x / p.w);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProj(Sampler1D<T> sampler, vec4 p, float bias)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProj";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p Bias $bias
    };
    default:
        return texture(sampler, p.x / p.w, bias);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProj(Sampler2D<T> sampler, vec3 p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProj";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p
    };
    default:
        return texture(sampler, p.xy / p.z);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProj(Sampler2D<T> sampler, vec3 p, float bias)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProj";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p Bias $bias
    };
    default:
        return texture(sampler, p.xy / p.z, bias);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProj(Sampler2D<T> sampler, vec4 p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProj";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p
    };
    default:
        return texture(sampler, p.xy / p.w);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProj(Sampler2D<T> sampler, vec4 p, float bias)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProj";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p Bias $bias
    };
    default:
        return texture(sampler, p.xy / p.w, bias);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProj(Sampler3D<T> sampler, vec4 p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProj";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p
    };
    default:
        return texture(sampler, p.xyz / p.w);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProj(Sampler3D<T> sampler, vec4 p, float bias)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProj";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p Bias $bias
    };
    default:
        return texture(sampler, p.xyz / p.w, bias);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureProj(sampler1DShadow sampler, vec4 p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProj";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xw__ = p.xwww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefImplicitLod $sampler $xw__ $compareValue
        };
    }
    default:
        return texture(sampler, p.xyz / p.w);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureProj(sampler1DShadow sampler, vec4 p, float bias)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProj";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xw__ = p.xwww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefImplicitLod $sampler $xw__ $compareValue Bias $bias
        };
    }
    default:
        return texture(sampler, p.xyz / p.w, bias);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureProj(sampler2DShadow sampler, vec4 p)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProj";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xyw_ = p.xyww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefImplicitLod $sampler $xyw_ $compareValue
        };
    }
    default:
        return texture(sampler, p.xyz / p.w);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureProj(sampler2DShadow sampler, vec4 p, float bias)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProj";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xyw_ = p.xyww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefImplicitLod $sampler $xyw_ $compareValue Bias $bias
        };
    }
    default:
        return texture(sampler, p.xyz / p.w, bias);
    }
}

// -------------------
// textureLod
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureLod(Sampler1D<T> sampler, float p, float lod)
{
    return __vectorReshape2<T.Element,4>(sampler.SampleLevel(p, lod));
}

__generic<T:ITexelElement, Shape: __ITextureShape, let isArray:int, let sampleCount:int, let format:int>
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureLod(_Texture<
        T,
        Shape,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        0, // isShadow
        1, // isCombined
        format
    > sampler, vector<float,Shape.dimensions+isArray> p, float lod)
{
    return __vectorReshape2<T.Element,4>(sampler.SampleLevel(p, lod));
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureLod(sampler2DShadow sampler, vec3 p, float lod)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLod";
    case spirv:
    {
        float compareValue = p.z;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Lod $lod
        };
    }
    default:
        return sampler.SampleCmp(p.xy, p.z);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureLod(sampler2DArrayShadow sampler, vec4 p, float lod)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLod";
    case spirv:
    {
        float compareValue = p.w;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Lod $lod
        };
    }
    default:
        return sampler.SampleCmp(p.xyz, p.w);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureLod(sampler1DShadow sampler, vec3 p, float lod)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLod";
    case spirv:
    {
        float compareValue = p.z;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Lod $lod
        };
    }
    default:
        return sampler.SampleCmp(p.x, p.z);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureLod(sampler1DArrayShadow sampler, vec3 p, float lod)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLod";
    case spirv:
    {
        float compareValue = p.z;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Lod $lod
        };
    }
    default:
        return sampler.SampleCmp(p.xy, p.z);
    }
}

// -------------------
// textureOffset
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureOffset(Sampler1D<T> sampler, float p, constexpr int offset, float bias = 0.0)
{
    return __vectorReshape2<T.Element,4>(sampler.SampleBias(p, bias, offset));
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureOffset(Sampler2D<T> sampler, vec2 p, constexpr ivec2 offset, float bias = 0.0)
{
    return __vectorReshape2<T.Element,4>(sampler.SampleBias(p, bias, offset));
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureOffset(Sampler3D<T> sampler, vec3 p, constexpr ivec3 offset, float bias = 0.0)
{
    return __vectorReshape2<T.Element,4>(sampler.SampleBias(p, bias, offset));
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureOffset(sampler2DShadow sampler, vec3 p, constexpr ivec2 offset, float bias = 0.0)
{
    vec2 location = p.xy;
    float compareValue = p.z;

    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureOffset";
    case spirv:
        return spirv_asm
        {
            result:$$float = OpImageSampleDrefImplicitLod $sampler $location $compareValue Bias|ConstOffset $bias $offset;
        };
    default:
        // TODO: Need to apply bias
        return sampler.SampleCmp(location, compareValue, offset);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureOffset(sampler1DShadow sampler, vec3 p, constexpr int offset)
{
    float location = p.x;
    float compareValue = p.z;

    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureOffset";
    case spirv:
        return spirv_asm
        {
            result:$$float = OpImageSampleDrefImplicitLod $sampler $location $compareValue ConstOffset $offset;
        };
    default:
        // TODO: Need to apply bias
        return sampler.SampleCmp(location, compareValue, offset);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureOffset(sampler1DShadow sampler, vec3 p, constexpr int offset, float bias)
{
    float location = p.x;
    float compareValue = p.z;

    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureOffset";
    case spirv:
        return spirv_asm
        {
            result:$$float = OpImageSampleDrefImplicitLod $sampler $location $compareValue Bias|ConstOffset $bias $offset;
        };
    default:
        // TODO: Need to apply bias
        return sampler.SampleCmp(location, compareValue, offset);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureOffset(Sampler1DArray<T> sampler, vec2 p, constexpr int offset, float bias = 0.0)
{
    return __vectorReshape2<T.Element,4>(sampler.SampleBias(p, bias, offset));
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureOffset(Sampler2DArray<T> sampler, vec3 p, constexpr ivec2 offset, float bias = 0.0)
{
    return __vectorReshape2<T.Element,4>(sampler.SampleBias(p, bias, offset));
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureOffset(sampler1DArrayShadow sampler, vec3 p, constexpr int offset)
{
    vec2 location = p.xy;
    float compareValue = p.z;

    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureOffset";
    case spirv:
        return spirv_asm
        {
            result:$$float = OpImageSampleDrefImplicitLod $sampler $location $compareValue ConstOffset $offset;
        };
    default:
        // TODO: Need to apply bias
        return sampler.SampleCmp(location, compareValue, vector<int,1>(offset));
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureOffset(sampler1DArrayShadow sampler, vec3 p, constexpr int offset, float bias)
{
    vec2 location = p.xy;
    float compareValue = p.z;

    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureOffset";
    case spirv:
        return spirv_asm
        {
            result:$$float = OpImageSampleDrefImplicitLod $sampler $location $compareValue Bias|ConstOffset $bias $offset;
        };
    default:
        // TODO: Need to apply bias
        return sampler.SampleCmp(location, compareValue, vector<int,1>(offset));
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureOffset(sampler2DArrayShadow sampler, vec4 p, constexpr ivec2 offset)
{
    return sampler.SampleCmp(p.xyz, p.w, offset);
}

// -------------------
// texelFetch
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1_samplerless)]
public vector<T.Element,4> texelFetch(Sampler1D<T> sampler, int p, int lod)
{
    return __vectorReshape2<T.Element,4>(sampler.Load(int2(p, lod)));
}

__generic<T:ITexelElement, Shape:__ITextureShape, let isArray:int, let sampleCount:int, let format:int>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1_samplerless)]
public vector<T.Element,4> texelFetch(_Texture<
        T,
        Shape,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        0, // isShadow
        1, // isCombined
        format
    > sampler, vector<int,Shape.dimensions+isArray> p, int lod)
{
    return __vectorReshape2<T.Element,4>(sampler.Load(__makeVector(p,lod)));
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1_samplerless)]
public vector<T.Element,4> texelFetch(Sampler2DRect<T> sampler, ivec2 p)
{
    return __vectorReshape2<T.Element,4>(sampler.Load(int3(p.xy,0)));
}

__generic<T:ITexelElement, let format:int>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_1_samplerless)]
public vector<T.Element,4> texelFetch(SamplerBuffer<T,format> sampler, int p)
{
    return __vectorReshape2<T.Element,4>(sampler.Load(p));
}

__generic<T:ITexelElement, let isArray:int, let sampleCount:int, let format:int>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1_samplerless)]
public vector<T.Element,4> texelFetch(_Texture<
        T,
        __Shape2D,
        isArray,
        1, // isMS
        sampleCount,
        0, // access
        0, // isShadow
        1, // isCombined
        format
    > sampler, vector<int,2+isArray> p, int lod)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "texelFetch";
    default:
        // TODO: Need to apply lod
        return __vectorReshape2<T.Element,4>(sampler.Load(__makeVector(p, 0)));
    }
}

// -------------------
// texelFetchOffset
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1_samplerless)]
public vector<T.Element,4> texelFetchOffset(Sampler1D<T> sampler, int p, int lod, constexpr int offset)
{
    return __vectorReshape2<T.Element,4>(sampler.Load(int2(p, lod), offset));
}

__generic<T:ITexelElement, Shape:__ITextureShape, let isArray:int, let sampleCount:int, let format:int>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1_samplerless)]
public vector<T.Element,4> texelFetchOffset(_Texture<
        T,
        Shape,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        0, // isShadow
        1, // isCombined
        format
    > sampler, vector<int,Shape.dimensions+isArray> p, int lod, constexpr vector<int,Shape.planeDimensions> offset)
{
    return __vectorReshape2<T.Element,4>(sampler.Load(__makeVector(p,lod), offset));
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1_samplerless)]
public vector<T.Element,4> texelFetchOffset(Sampler2DRect<T> sampler, ivec2 p, constexpr ivec2 offset)
{
    return __vectorReshape2<T.Element,4>(sampler.Load(__makeVector(p, 0), offset));
}

// -------------------
// textureProjOffset
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjOffset(Sampler1D<T> sampler, vec2 p, constexpr int offset)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjOffset";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p ConstOffset $offset
    };
    default:
        return textureOffset(sampler, p.x / p.y, offset);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjOffset(Sampler1D<T> sampler, vec2 p, constexpr int offset, float bias)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjOffset";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p Bias|ConstOffset $bias $offset
    };
    default:
        return textureOffset(sampler, p.x / p.y, offset, bias);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjOffset(Sampler1D<T> sampler, vec4 p, constexpr int offset)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjOffset";
    case spirv:
    {
        vec4 xw__ = p.xwww;
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $xw__ ConstOffset $offset
        };
    }
    default:
        return textureOffset(sampler, p.x / p.w, offset);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjOffset(Sampler1D<T> sampler, vec4 p, constexpr int offset, float bias)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjOffset";
    case spirv:
    {
        vec4 xw__ = p.xwww;
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $xw__ Bias|ConstOffset $bias $offset
        };
    }
    default:
        return textureOffset(sampler, p.x / p.w, offset, bias);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjOffset(Sampler2D<T> sampler, vec3 p, constexpr ivec2 offset)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjOffset";
    case spirv: return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p ConstOffset $offset
    };
    default:
        return textureOffset(sampler, p.xy / p.z, offset);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjOffset(Sampler2D<T> sampler, vec3 p, constexpr ivec2 offset, float bias)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjOffset";
    case spirv: return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p Bias|ConstOffset $bias $offset
    };
    default:
        return textureOffset(sampler, p.xy / p.z, offset, bias);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjOffset(Sampler2D<T> sampler, vec4 p, constexpr ivec2 offset)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjOffset";
    case spirv:
    {
        vec4 xyw__ = p.xyww;
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $xyw__ ConstOffset $offset
        };
    }
    default:
        return textureOffset(sampler, p.xy / p.w, offset);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjOffset(Sampler2D<T> sampler, vec4 p, constexpr ivec2 offset, float bias)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjOffset";
    case spirv:
    {
        vec4 xyw_ = p.xyww;
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $xyw_ Bias|ConstOffset $bias $offset
        };
    }
    default:
        return textureOffset(sampler, p.xy / p.w, offset, bias);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjOffset(Sampler3D<T> sampler, vec4 p, constexpr ivec3 offset)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjOffset";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p ConstOffset $offset
    };
    default:
        return textureOffset(sampler, p.xyz / p.w, offset);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjOffset(Sampler3D<T> sampler, vec4 p, constexpr ivec3 offset, float bias)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjOffset";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjImplicitLod $sampler $p Bias|ConstOffset $bias $offset
    };
    default:
        return textureOffset(sampler, p.xyz / p.w, offset, bias);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureProjOffset(sampler1DShadow sampler, vec4 p, constexpr int offset)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjOffset";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xw__ = p.xwww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefImplicitLod $sampler $xw__ $compareValue ConstOffset $offset
        };
    }
    default:
        return textureOffset(sampler, p.xyz / p.w, offset);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureProjOffset(sampler1DShadow sampler, vec4 p, constexpr int offset, float bias)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjOffset";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xw__ = p.xwww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefImplicitLod $sampler $xw__ $compareValue Bias|ConstOffset $bias $offset
        };
    }
    default:
        return textureOffset(sampler, p.xyz / p.w, offset, bias);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureProjOffset(sampler2DShadow sampler, vec4 p, constexpr ivec2 offset)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjOffset";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xyw_ = p.xyww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefImplicitLod $sampler $xyw_ $compareValue ConstOffset $offset
        };
    }
    default:
        return textureOffset(sampler, p.xyz / p.w, offset);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureProjOffset(sampler2DShadow sampler, vec4 p, constexpr ivec2 offset, float bias)
{
    __requireComputeDerivative();
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjOffset";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xyw_ = p.xyww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefImplicitLod $sampler $xyw_ $compareValue Bias|ConstOffset $bias $offset
        };
    }
    default:
        return textureOffset(sampler, p.xyz / p.w, offset, bias);
    }
}

// -------------------
// textureLodOffset
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0)]
public vector<T.Element,4> textureLodOffset(Sampler1D<T> sampler, float p, float lod, constexpr int offset)
{
    return __vectorReshape2<T.Element,4>(sampler.SampleLevel(p, lod, offset));
}

__generic<T:ITexelElement, Shape:__ITextureShape, let isArray:int, let sampleCount:int, let format:int>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_0)]
public vector<T.Element,4> textureLodOffset(_Texture<
        T,
        Shape,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        0, // isShadow
        1, // isCombined
        format
    > sampler, vector<float,Shape.dimensions+isArray> p, float lod, constexpr vector<int,Shape.planeDimensions> offset)
{
    return __vectorReshape2<T.Element,4>(sampler.SampleLevel(p, lod, offset));
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureLodOffset(sampler1DShadow sampler, vec3 p, float lod, constexpr int offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLodOffset";
    case spirv:
    {
        float compareValue = p.z;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Lod|ConstOffset $lod $offset
        };
    }
    default:
        // TODO: Need to apply lod
        return sampler.SampleCmpLevelZero(p.x, p.z, offset);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureLodOffset(sampler2DShadow sampler, vec3 p, float lod, constexpr ivec2 offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLodOffset";
    case spirv:
    {
        float compareValue = p.z;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Lod|ConstOffset $lod $offset
        };
    }
    default:
        // TODO: Need to apply lod
        return sampler.SampleCmpLevelZero(p.xy, p.z, offset);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureLodOffset(sampler1DArrayShadow sampler, vec3 p, float lod, constexpr int offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureLodOffset";
    case spirv:
    {
        float compareValue = p.z;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Lod|ConstOffset $lod $offset
        };
    }
    default:
        // TODO: Need to apply lod
        return sampler.SampleCmpLevelZero(p.xy, p.z, offset);
    }
}

// -------------------
// textureProjLod
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjLod(Sampler1D<T> sampler, vec2 p, float lod)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjLod";
    case spirv: return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $p Lod $lod
    };
    default:
        return textureLod(sampler, p.x / p.y, lod);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjLod(Sampler1D<T> sampler, vec4 p, float lod)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjLod";
    case spirv:
    {
        vec4 xw__ = p.xwww;
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $xw__ Lod $lod
        };
    }
    default:
        return textureLod(sampler, p.x / p.w, lod);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjLod(Sampler2D<T> sampler, vec3 p, float lod)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjLod";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $p Lod $lod
    };
    default:
        return textureLod(sampler, p.xy / p.z, lod);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjLod(Sampler2D<T> sampler, vec4 p, float lod)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjLod";
    case spirv:
    {
        vec4 xyw_ = p.xyww;
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $xyw_ Lod $lod
        };
    }
    default:
        return textureLod(sampler, p.xy / p.w, lod);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_cuda_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjLod(Sampler3D<T> sampler, vec4 p, float lod)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjLod";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $p Lod $lod
    };
    default:
        return textureLod(sampler, p.xyz / p.w, lod);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureProjLod(sampler1DShadow sampler, vec4 p, float lod)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjLod";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xw__ = p.xwww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefExplicitLod $sampler $xw__ $compareValue Lod $lod
        };
    }
    default:
        return textureLod(sampler, p.xyz / p.w, lod);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureProjLod(sampler2DShadow sampler, vec4 p, float lod)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjLod";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xyw_ = p.xyww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefExplicitLod $sampler $xyw_ $compareValue Lod $lod
        };
    }
    default:
        return textureLod(sampler, p.xyz / p.w, lod);
    }
}

// -------------------
// textureProjLodOffset
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjLodOffset(Sampler1D<T> sampler, vec2 p, float lod, constexpr int offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjLodOffset";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $p Lod|ConstOffset $lod $offset
    };
    default:
        return textureLodOffset(sampler, p.x / p.y, lod, offset);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjLodOffset(Sampler1D<T> sampler, vec4 p, float lod, constexpr int offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjLodOffset";
    case spirv:
    {
        vec4 xw__ = p.xwww;
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $xw__ Lod|ConstOffset $lod $offset
        };
    }
    default:
        return textureLodOffset(sampler, p.x / p.w, lod, offset);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjLodOffset(Sampler2D<T> sampler, vec3 p, float lod, constexpr ivec2 offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjLodOffset";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $p Lod|ConstOffset $lod $offset
    };
    default:
        return textureLodOffset(sampler, p.xy / p.z, lod, offset);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjLodOffset(Sampler2D<T> sampler, vec4 p, float lod, constexpr ivec2 offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjLodOffset";
    case spirv:
    {
        vec4 xyw_ = p.xyww;
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $xyw_ Lod|ConstOffset $lod $offset
        };
    }
    default:
        return textureLodOffset(sampler, p.xy / p.w, lod, offset);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vector<T.Element,4> textureProjLodOffset(Sampler3D<T> sampler, vec4 p, float lod, constexpr ivec3 offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjLodOffset";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $p Lod|ConstOffset $lod $offset
    };
    default:
        return textureLodOffset(sampler, p.xyz / p.w, lod, offset);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureProjLodOffset(sampler1DShadow sampler, vec4 p, float lod, constexpr int offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjLodOffset";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xw__ = p.xwww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefExplicitLod $sampler $xw__ $compareValue Lod|ConstOffset $lod $offset
        };
    }
    default:
        return textureLodOffset(sampler, p.xyz / p.w, lod, offset);
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, texture_shadowlod)]
public float textureProjLodOffset(sampler2DShadow sampler, vec4 p, float lod, constexpr ivec2 offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjLodOffset";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xyw_ = p.xyww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefExplicitLod $sampler $xyw_ $compareValue Lod|ConstOffset $lod $offset
        };
    }
    default:
        return textureLodOffset(sampler, p.xyz / p.w, lod, offset);
    }
}

// -------------------
// textureGrad
// -------------------


__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
public vector<T.Element,4> textureGrad(Sampler1D<T> sampler, float p, float dPdx, float dPdy)
{
    return __vectorReshape2<T.Element,4>(sampler.SampleGrad(p, dPdx, dPdy));
}

__generic<T:ITexelElement, Shape:__ITextureShape, let isArray:int, let sampleCount:int, let format:int>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
public vector<T.Element,4> textureGrad(_Texture<
        T,
        Shape,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        0, // isShadow
        1, // isCombined
        format
    > sampler, vector<float,Shape.dimensions+isArray> p, vector<float,Shape.dimensions> dPdx, vector<float,Shape.dimensions> dPdy)
{
    return __vectorReshape2<T.Element,4>(sampler.SampleGrad(p, dPdx, dPdy));
}

[ForceInline]
[require(glsl_spirv, texture_shadowlod)]
public float textureGrad(sampler1DShadow sampler, vec3 p, float dPdx, float dPdy)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureGrad";
    case spirv:
    {
        float compareValue = p.z;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Grad $dPdx $dPdy
        };
    }
    }
}

[ForceInline]
[require(glsl_spirv, texture_shadowlod)]
public float textureGrad(sampler1DArrayShadow sampler, vec3 p, float dPdx, float dPdy)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureGrad";
    case spirv:
    {
        float compareValue = p.z;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Grad $dPdx $dPdy
        };
    }
    }
}

[ForceInline]
[require(glsl_spirv, texture_shadowlod)]
public float textureGrad(sampler2DShadow sampler, vec3 p, vec2 dPdx, vec2 dPdy)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureGrad";
    case spirv:
    {
        float compareValue = p.z;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Grad $dPdx $dPdy
        };
    }
    }
}

[ForceInline]
[require(glsl_spirv, texture_shadowlod)]
public float textureGrad(samplerCubeShadow sampler, vec4 p, vec3 dPdx, vec3 dPdy)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureGrad";
    case spirv:
    {
        float compareValue = p.w;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Grad $dPdx $dPdy
        };
    }
    }
}

[ForceInline]
[require(glsl_spirv, texture_shadowlod)]
public float textureGrad(sampler2DArrayShadow sampler, vec4 p, vec2 dPdx, vec2 dPdy)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureGrad";
    case spirv:
    {
        float compareValue = p.w;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Grad $dPdx $dPdy
        };
    }
    }
}

// -------------------
// textureGradOffset
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
public vector<T.Element,4> textureGradOffset(Sampler1D<T> sampler, float p, float dPdx, float dPdy, constexpr int offset)
{
    return __vectorReshape2<T.Element,4>(sampler.SampleGrad(p, dPdx, dPdy, offset));
}

__generic<T:ITexelElement, Shape:__ITextureShape, let isArray:int, let sampleCount:int, let format:int>
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
[ForceInline]
public vector<T.Element,4> textureGradOffset(_Texture<
        T,
        Shape,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        0, // isShadow
        1, // isCombined
        format
    > sampler, vector<float,Shape.dimensions+isArray> p, vector<float,Shape.dimensions> dPdx, vector<float,Shape.dimensions> dPdy, constexpr vector<int,Shape.dimensions> offset)
{
    return __vectorReshape2<T.Element,4>(sampler.SampleGrad(p, dPdx, dPdy, offset));
}

[ForceInline]
[require(glsl_spirv, texture_shadowlod)]
public float textureGradOffset(sampler1DShadow sampler, vec3 p, float dPdx, float dPdy, constexpr int offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureGradOffset";
    case spirv:
    {
        float compareValue = p.z;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Grad|ConstOffset $dPdx $dPdy $offset
        };
    }
    }
}

[ForceInline]
[require(glsl_spirv, texture_shadowlod)]
public float textureGradOffset(sampler2DShadow sampler, vec3 p, vec2 dPdx, vec2 dPdy, constexpr ivec2 offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureGradOffset";
    case spirv:
    {
        float compareValue = p.z;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Grad|ConstOffset $dPdx $dPdy $offset
        };
    }
    }
}

[ForceInline]
[require(glsl_spirv, texture_shadowlod)]
public float textureGradOffset(sampler1DArrayShadow sampler, vec3 p, float dPdx, float dPdy, constexpr int offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureGradOffset";
    case spirv:
    {
        float compareValue = p.z;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Grad|ConstOffset $dPdx $dPdy $offset
        };
    }
    }
}

[ForceInline]
[require(glsl_spirv, texture_shadowlod)]
public float textureGradOffset(sampler2DArrayShadow sampler, vec4 p, vec2 dPdx, vec2 dPdy, constexpr ivec2 offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureGradOffset";
    case spirv:
    {
        float compareValue = p.w;
        return spirv_asm {
            result:$$float = OpImageSampleDrefExplicitLod $sampler $p $compareValue Grad|ConstOffset $dPdx $dPdy $offset
        };
    }
    }
}

// -------------------
// textureProjGrad
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
public vector<T.Element,4> textureProjGrad(Sampler1D<T> sampler, vec2 p, float dPdx, float dPdy)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjGrad";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $p Grad $dPdx $dPdy
    };
    default:
        return textureGrad(sampler, p.x / p.y, dPdx, dPdy);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
public vector<T.Element,4> textureProjGrad(Sampler1D<T> sampler, vec4 p, float dPdx, float dPdy)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjGrad";
    case spirv:
    {
        vec4 xw__ = p.xwww;
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $xw__ Grad $dPdx $dPdy
        };
    }
    default:
        return textureGrad(sampler, p.x / p.w, dPdx, dPdy);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
public vector<T.Element,4> textureProjGrad(Sampler2D<T> sampler, vec3 p, vec2 dPdx, vec2 dPdy)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjGrad";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $p Grad $dPdx $dPdy
    };
    default:
        return textureGrad(sampler, p.xy / p.z, dPdx, dPdy);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
public vector<T.Element,4> textureProjGrad(Sampler2D<T> sampler, vec4 p, vec2 dPdx, vec2 dPdy)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjGrad";
    case spirv:
    {
        vec4 xyw_ = p.xyww;
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $xyw_ Grad $dPdx $dPdy
        };
    }
    default:
        return textureGrad(sampler, p.xy / p.w, dPdx, dPdy);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
public vector<T.Element,4> textureProjGrad(Sampler3D<T> sampler, vec4 p, vec3 dPdx, vec3 dPdy)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjGrad";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $p Grad $dPdx $dPdy
    };
    default:
        return textureGrad(sampler, p.xyz / p.w, dPdx, dPdy);
    }
}

[ForceInline]
[require(glsl_spirv, texture_shadowlod)]
public float textureProjGrad(sampler1DShadow sampler, vec4 p, float dPdx, float dPdy)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjGrad";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xw__ = p.xwww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefExplicitLod $sampler $xw__ $compareValue Grad $dPdx $dPdy
        };
    }
    default:
        return textureGrad(sampler, p.xyz / p.w, dPdx, dPdy);
    }
}

[ForceInline]
[require(glsl_spirv, texture_shadowlod)]
public float textureProjGrad(sampler2DShadow sampler, vec4 p, vec2 dPdx, vec2 dPdy)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjGrad";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xyw_ = p.xyww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefExplicitLod $sampler $xyw_ $compareValue Grad $dPdx $dPdy
        };
    }
    default:
        return textureGrad(sampler, p.xyz / p.w, dPdx, dPdy);
    }
}

// -------------------
// textureProjGradOffset
// -------------------

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
public vector<T.Element,4> textureProjGradOffset(Sampler1D<T> sampler, vec2 p, float dPdx, float dPdy, constexpr int offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjGradOffset";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $p Grad|ConstOffset $dPdx $dPdy $offset
    };
    default:
        return textureGradOffset(sampler, p.x / p.y, dPdx, dPdy, offset);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
public vector<T.Element,4> textureProjGradOffset(Sampler1D<T> sampler, vec4 p, float dPdx, float dPdy, constexpr int offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjGradOffset";
    case spirv:
    {
        vec4 xw__ = p.xwww;
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $xw__ Grad|ConstOffset $dPdx $dPdy $offset
        };
    }
    default:
        return textureGradOffset(sampler, p.x / p.w, dPdx, dPdy, offset);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
public vector<T.Element,4> textureProjGradOffset(Sampler2D<T> sampler, vec3 p, vec2 dPdx, vec2 dPdy, constexpr ivec2 offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjGradOffset";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $p Grad|ConstOffset $dPdx $dPdy $offset
    };
    default:
        return textureGradOffset(sampler, p.xy / p.z, dPdx, dPdy, offset);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
public vector<T.Element,4> textureProjGradOffset(Sampler2D<T> sampler, vec4 p, vec2 dPdx, vec2 dPdy, constexpr ivec2 offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjGradOffset";
    case spirv:
    {
        vec4 xyw_ = p.xyww;
        return spirv_asm {
            result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $xyw_ Grad|ConstOffset $dPdx $dPdy $offset
        };
    }
    default:
        return textureGradOffset(sampler, p.xy / p.w, dPdx, dPdy, offset);
    }
}

__generic<T:ITexelElement>
[ForceInline]
[require(cpp_glsl_hlsl_spirv, texture_sm_4_1)]
public vector<T.Element,4> textureProjGradOffset(Sampler3D<T> sampler, vec4 p, vec3 dPdx, vec3 dPdy, constexpr ivec3 offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjGradOffset";
    case spirv: return spirv_asm {
        result:$$vector<T.Element,4> = OpImageSampleProjExplicitLod $sampler $p Grad|ConstOffset $dPdx $dPdy $offset
    };
    default:
        return textureGradOffset(sampler, p.xyz / p.w, dPdx, dPdy, offset);
    }
}

[ForceInline]
[require(glsl_spirv, texture_shadowlod)]
public float textureProjGradOffset(sampler1DShadow sampler, vec4 p, float dPdx, float dPdy, constexpr int offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjGradOffset";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xw__ = p.xwww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefExplicitLod $sampler $xw__ $compareValue Grad|ConstOffset $dPdx $dPdy $offset
        };
    }
    default:
        return textureGradOffset(sampler, p.xyz / p.w, dPdx, dPdy, offset);
    }
}

[ForceInline]
[require(glsl_spirv, texture_shadowlod)]
public float textureProjGradOffset(sampler2DShadow sampler, vec4 p, vec2 dPdx, vec2 dPdy, constexpr ivec2 offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "textureProjGradOffset";
    case spirv:
    {
        float compareValue = p.z;
        vec4 xyw_ = p.xyww;
        return spirv_asm {
            result:$$float = OpImageSampleProjDrefExplicitLod $sampler $xyw_ $compareValue Grad|ConstOffset $dPdx $dPdy $offset
        };
    }
    default:
        return textureGradOffset(sampler, p.xyz / p.w, dPdx, dPdy, offset);
    }
}

//
// Section 8.9.4. Texture Gather Functions
//

// -------------------
// textureGather
// -------------------

__generic<T:ITexelElement, Shape:__ITextureShape, let isArray:int, let sampleCount:int, let format:int>
[ForceInline]
[require(glsl_hlsl_spirv, texture_gather)]
public vector<T.Element,4> textureGather(_Texture<
        T,
        Shape,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        0, // isShadow
        1, // isCombined
        format
    > sampler, vector<float,Shape.dimensions+isArray> p, int comp = 0)
{
    switch (comp)
    {
        case 1: return sampler.GatherGreen(p);
        case 2: return sampler.GatherBlue(p);
        case 3: return sampler.GatherAlpha(p);
    }
    return sampler.GatherRed(p);
}

__generic<Shape:__ITextureShape, let isArray:int, let sampleCount:int, let format:int>
[ForceInline]
[require(glsl_hlsl_spirv, texture_gather)]
public vec4 textureGather(_Texture<
        float,
        Shape,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        1, // isShadow
        1, // isCombined
        format
    > sampler, vector<float,Shape.dimensions+isArray> p, float refZ)
{
    return sampler.GatherCmp(p, refZ);
}

// -------------------
// textureGatherOffset
// -------------------

__generic<T:ITexelElement, let isArray:int, let sampleCount:int, let format:int>
[ForceInline]
[require(glsl_hlsl_spirv, texture_gather)]
public vector<T.Element,4> textureGatherOffset(_Texture<
        T,
        __Shape2D,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        0, // isShadow
        1, // isCombined
        format
    > sampler, vector<float,2+isArray> p, constexpr vector<int,2> offset, int comp = 0)
{
    switch (comp)
    {
        case 1: return sampler.GatherGreen(p, offset);
        case 2: return sampler.GatherBlue(p, offset);
        case 3: return sampler.GatherAlpha(p, offset);
    }
    return sampler.Gather(p, offset);
}

__generic<let isArray:int, let sampleCount:int, let format:int>
[ForceInline]
[require(glsl_hlsl_spirv, texture_gather)]
public vec4 textureGatherOffset(_Texture<
        float,
        __Shape2D,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        1, // isShadow
        1, // isCombined
        format
    > sampler, vector<float,2+isArray> p, float refZ, constexpr vector<int,2> offset)
{
    return sampler.GatherCmp(p, refZ, offset);
}

// -------------------
// textureGatherOffsets
// -------------------

__generic<T:ITexelElement, let isArray:int, let sampleCount:int, let format:int>
[ForceInline]
[require(glsl_hlsl_spirv, texture_gather)]
public vector<T.Element,4> textureGatherOffsets(_Texture<
        T,
        __Shape2D,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        0, // isShadow
        1, // isCombined
        format
    > sampler, vector<float,2+isArray> p, constexpr vector<int,2> offsets[4], int comp = 0)
{
    switch (comp)
    {
        case 1: return sampler.GatherGreen(p, offsets[0], offsets[1], offsets[2], offsets[3]);
        case 2: return sampler.GatherBlue(p, offsets[0], offsets[1], offsets[2], offsets[3]);
        case 3: return sampler.GatherAlpha(p, offsets[0], offsets[1], offsets[2], offsets[3]);
    }
    return sampler.Gather(p, offsets[0], offsets[1], offsets[2], offsets[3]);
}

__generic<let isArray:int, let sampleCount:int, let format:int>
[ForceInline]
[require(glsl_hlsl_spirv, texture_gather)]
public vec4 textureGatherOffsets(_Texture<
        float,
        __Shape2D,
        isArray,
        0, // isMS
        sampleCount,
        0, // access
        1, // isShadow
        1, // isCombined
        format
    > sampler, vector<float,2+isArray> p, float refZ, constexpr vector<int,2> offsets[4])
{
    return sampler.GatherCmp(p, refZ, offsets[0], offsets[1], offsets[2], offsets[3]);
}

//
// Section 8.9.5. Compatibility Profile Texture Functions
//
// Note: the following functions exist for GLSL but not for SPIR-V.
// If we use `case glsl: __intrinsic_asm "XXX";`, it will cause an
// error when we try to translate the GLSL to SPIR-V.
// So we cannot use them.

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture1D(sampler1D sampler, float coord)
{
    return texture(sampler, coord);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture1D(sampler1D sampler, float coord, float bias)
{
    return texture(sampler, coord, bias);
}

[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture1DProj(sampler1D sampler, vec2 coord)
{
    return textureProj(sampler, coord);
}

[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture1DProj(sampler1D sampler, vec2 coord, float bias)
{
    return textureProj(sampler, coord, bias);
}

[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture1DProj(sampler1D sampler, vec4 coord)
{
    return textureProj(sampler, coord);
}

[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture1DProj(sampler1D sampler, vec4 coord, float bias)
{
    return textureProj(sampler, coord, bias);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture1DLod(sampler1D sampler, float coord, float lod)
{
    return textureLod(sampler, coord, lod);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture1DProjLod(sampler1D sampler, vec2 coord, float lod)
{
    return textureProjLod(sampler, coord, lod);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture1DProjLod(sampler1D sampler, vec4 coord, float lod)
{
    return textureProjLod(sampler, coord, lod);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture2D(sampler2D sampler, vec2 coord)
{
    return texture(sampler, coord);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture2D(sampler2D sampler, vec2 coord, float bias)
{
    return texture(sampler, coord, bias);
}

[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture2DProj(sampler2D sampler, vec3 coord)
{
    return textureProj(sampler, coord);
}

[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture2DProj(sampler2D sampler, vec3 coord, float bias)
{
    return textureProj(sampler, coord, bias);
}

[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture2DProj(sampler2D sampler, vec4 coord)
{
    return textureProj(sampler, coord);
}

[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture2DProj(sampler2D sampler, vec4 coord, float bias)
{
    return textureProj(sampler, coord, bias);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture2DLod(sampler2D sampler, vec2 coord, float lod)
{
    return textureLod(sampler, coord, lod);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture2DProjLod(sampler2D sampler, vec3 coord, float lod)
{
    return textureProjLod(sampler, coord, lod);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture2DProjLod(sampler2D sampler, vec4 coord, float lod)
{
    return textureProjLod(sampler, coord, lod);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture3D(sampler3D sampler, vec3 coord)
{
    return texture(sampler, coord);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture3D(sampler3D sampler, vec3 coord, float bias)
{
    return texture(sampler, coord, bias);
}

[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture3DProj(sampler3D sampler, vec4 coord)
{
    return textureProj(sampler, coord);
}

[require(glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture3DProj(sampler3D sampler, vec4 coord, float bias)
{
    return textureProj(sampler, coord, bias);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture3DLod(sampler3D sampler, vec3 coord, float lod)
{
    return textureLod(sampler, coord, lod);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 texture3DProjLod(sampler3D sampler, vec4 coord, float lod)
{
    return textureProjLod(sampler, coord, lod);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 textureCube(samplerCube sampler, vec3 coord)
{
    return texture(sampler, coord);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 textureCube(samplerCube sampler, vec3 coord, float bias)
{
    return texture(sampler, coord, bias);
}

[require(cpp_glsl_hlsl_spirv, texture_sm_4_0_fragment)]
public vec4 textureCubeLod(samplerCube sampler, vec3 coord, float lod)
{
    return textureLod(sampler, coord, lod);
}

[require(glsl_spirv, texture_shadowlod)]
public vec4 shadow1D(sampler1DShadow sampler, vec3 coord)
{
    return texture(sampler, coord);
}

[require(glsl_spirv, texture_shadowlod)]
public vec4 shadow1D(sampler1DShadow sampler, vec3 coord, float bias)
{
    return texture(sampler, coord, bias);
}

[require(glsl_spirv, texture_shadowlod)]
public vec4 shadow2D(sampler2DShadow sampler, vec3 coord)
{
    return texture(sampler, coord);
}

[require(glsl_spirv, texture_shadowlod)]
public vec4 shadow2D(sampler2DShadow sampler, vec3 coord, float bias)
{
    return texture(sampler, coord, bias);
}

[require(glsl_spirv, texture_shadowlod)]
public vec4 shadow1DProj(sampler1DShadow sampler, vec4 coord)
{
    return textureProj(sampler, coord);
}

[require(glsl_spirv, texture_shadowlod)]
public vec4 shadow1DProj(sampler1DShadow sampler, vec4 coord, float bias)
{
    return textureProj(sampler, coord, bias);
}

[require(glsl_spirv, texture_shadowlod)]
public vec4 shadow2DProj(sampler2DShadow sampler, vec4 coord)
{
    return textureProj(sampler, coord);
}

[require(glsl_spirv, texture_shadowlod)]
public vec4 shadow2DProj(sampler2DShadow sampler, vec4 coord, float bias)
{
    return textureProj(sampler, coord, bias);
}

[require(glsl_spirv, texture_shadowlod)]
public vec4 shadow1DLod(sampler1DShadow sampler, vec3 coord, float lod)
{
    return textureLod(sampler, coord, lod);
}

[require(glsl_spirv, texture_shadowlod)]
public vec4 shadow2DLod(sampler2DShadow sampler, vec3 coord, float lod)
{
    return textureLod(sampler, coord, lod);
}

[require(glsl_spirv, texture_shadowlod)]
public vec4 shadow1DProjLod(sampler1DShadow sampler, vec4 coord, float lod)
{
    return textureProjLod(sampler, coord, lod);
}

[require(glsl_spirv, texture_shadowlod)]
public vec4 shadow2DProjLod(sampler2DShadow sampler, vec4 coord, float lod)
{
    return textureProjLod(sampler, coord, lod);
}

//// gimageDim Implementation

public static const int GL_TEXTURE_CUBE_MAP_POSITIVE_X = 0;
public static const int GL_TEXTURE_CUBE_MAP_NEGATIVE_X = 1;
public static const int GL_TEXTURE_CUBE_MAP_POSITIVE_Y = 2;
public static const int GL_TEXTURE_CUBE_MAP_NEGATIVE_Y = 3;
public static const int GL_TEXTURE_CUBE_MAP_POSITIVE_Z = 4;
public static const int GL_TEXTURE_CUBE_MAP_NEGATIVE_Z = 5;

__generic<T : __BuiltinType>
[ForceInline] void typeRequireChecks_image_atomic_tier1()
{
    __target_switch
    {
    case glsl:
    {
        if (__type_equals<T, float>())
            __requireTargetExtension("GL_EXT_shader_atomic_float");
    }
    case spirv:
        if (__type_equals<T, float>())
        {
            spirv_asm
            {
                OpExtension "SPV_EXT_shader_atomic_float_add";
                OpCapability AtomicFloat32AddEXT
            };
        }
    }
}

__generic<T : __BuiltinType>
[ForceInline] void typeRequireChecks_image_atomic_tier2()
{
    __target_switch
    {
    case glsl:
    {
        if (__type_equals<T, float>())
            __requireTargetExtension("GL_EXT_shader_atomic_float2");
    }
    case spirv:
        if (__type_equals<T, float>())
        {
            spirv_asm
            {
                OpExtension "SPV_EXT_shader_atomic_float_min_max";
                OpCapability AtomicFloat32MinMaxEXT
            };
        }
    }
}

__generic<T : __BuiltinType>
[ForceInline] void typeRequireChecks_image_atomic_int64()
{
    __target_switch
    {
    case spirv:
        if (__type_equals<T, uint64_t>()
            || __type_equals<T, int64_t>())
        {
            spirv_asm
            {
                OpCapability Int64Atomics
            };
        }
        return;
    }
}

${{{{
{
    struct ImageTypeInfo
    {
        const char *prefix;
        const char *type;
        bool isInt;
        const char *SPVTypePrefix;
        const char *SPVSubTypePrefix;
        const char *SPVAtomicSuffix;
    };
    static const ImageTypeInfo kTypeWithPrefix[] =
        {
            { "", "float", false, "F", "F", "EXT"},
            { "u", "uint", true, "I", "U", ""},
            { "i", "int", true, "I", "S", ""},
            { "u64", "uint64_t", true, "I", "U", ""},
            { "i64", "int64_t", true, "I", "S", ""},
        };
    struct ShapeTypeInfo
    {
        const char *suffix;
        int imageSizeIVecDim;
        int imageCoordIVecIndexerDim;
        const char *shape;
        int isArray;
        int isMS;
        bool isRect;
    }
    static const kShapeType[] =
        {
            { "1D", 1, 1,"__Shape1D", 0, 0, 0},

            { "2D", 2, 2, "__Shape2D", 0, 0, 0},
            { "3D", 3, 3, "__Shape3D", 0, 0, 0},
            { "Cube", 2, 3, "__ShapeCube", 0, 0, 0},
            { "2DRect", 2, 2, "__Shape2D", 0, 0, 1},
            { "1DArray", 2, 2, "__Shape1D", 1, 0, 0},

            { "2DArray", 3, 3, "__Shape2D", 1, 0, 0},
            { "CubeArray", 3, 3, "__ShapeCube", 1, 0, 0},
            { "Buffer", 1, 1, "__ShapeBuffer", 0, 0, 0},
            { "2DMS", 2, 2, "__Shape2D", 0, 1, 0},
            { "2DMSArray", 3, 3, "__Shape2D", 1, 1, 0},
        };
    for (const auto& targetType : kTypeWithPrefix)
    for (const auto& targetShape : kShapeType)
    {
        // gimageDim is the format of image buffers, we
        // need to resolve a type name from the format:
        // targetType.prefix is the 'g'
        // targetShape.suffix is the 'Dim'
        StringBuilder fullTypeNameBuilder;
        fullTypeNameBuilder << targetType.prefix << "image" << targetShape.suffix;
        auto fullTypeName = fullTypeNameBuilder.toString();

        // We need a string to resolve for the imageSize return type
        // since each shape has a different return which may not be
        // a vec
        StringBuilder imageSizeIntOrIVecBuilder;
        if (targetShape.imageSizeIVecDim == 1)
            imageSizeIntOrIVecBuilder << "int";
        else
            imageSizeIntOrIVecBuilder << "ivec" << targetShape.imageSizeIVecDim;
        auto imageSizeIntOrIVec = imageSizeIntOrIVecBuilder.toString();

        // Following OpenGL/Vulkan semantics, IMAGE_PARAMS string represents
        // the changing list of parameters per gimageDim type
        // https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.pdf
        StringBuilder IMAGE_PARAMSBuilder;
        IMAGE_PARAMSBuilder << fullTypeName << " image,";
        int paramC = targetShape.imageCoordIVecIndexerDim;
        if (paramC == 1)
            IMAGE_PARAMSBuilder << "int";
        else
            IMAGE_PARAMSBuilder << "ivec" << paramC;
        IMAGE_PARAMSBuilder << " P";
        if (targetShape.isMS)
        {
            IMAGE_PARAMSBuilder << ", int sample";
        }
        auto IMAGE_PARAMS = IMAGE_PARAMSBuilder.toString();

        // SPIR-V requires a 'sample' parameter for all texture
        // functions. We need to handle a dynamic creation of
        // 'sample' if 'sample' parameter is not in IMAGE_PARAMS
        StringBuilder SPV_PREFIX_IMAGE_PARAMSBuilder;
        StringBuilder SPV_SUFFIX_IMAGE_PARAMSBuilder;
        StringBuilder SPV_DEFAULT_SAMPLE_VAR_IF_MISSINGBuilder;

        SPV_PREFIX_IMAGE_PARAMSBuilder << "$image $P";
        if (targetShape.isMS)
        {
            SPV_SUFFIX_IMAGE_PARAMSBuilder << " Sample $sample";
        }
        else
        {
            SPV_DEFAULT_SAMPLE_VAR_IF_MISSINGBuilder << "let sample = 0;";
        }

        auto SPV_PREFIX_IMAGE_PARAMS = SPV_PREFIX_IMAGE_PARAMSBuilder.toString();
        auto SPV_SUFFIX_IMAGE_PARAMS = SPV_SUFFIX_IMAGE_PARAMSBuilder.toString();
        auto SPV_DEFAULT_SAMPLE_VAR_IF_MISSING = SPV_DEFAULT_SAMPLE_VAR_IF_MISSINGBuilder.toString();
    }}}}

        __generic<let format:int=0>
        public typealias $(fullTypeName) = _Texture<
            $(targetType.type),
            $(targetShape.shape),
            $(targetShape.isArray), // isArray
            $(targetShape.isMS), // isMS
            0, // sampleCount
            1, // access
            0, // isShadow
            0, // isCombined
            format
        >;

${{{{
        if(targetShape.isRect)
            continue;
}}}}

        // readonly writeonly in GLSL means an object only allows information queries.
        [__readNone]
        [ForceInline]
        [require(glsl_spirv, image_size)]
        public $(imageSizeIntOrIVec) imageSize(readonly writeonly $(fullTypeName) image)
        {
            __target_switch
            {
            case glsl: __intrinsic_asm "imageSize($*0)";
            case spirv:
                return spirv_asm
                {
                    OpCapability ImageQuery;
                    result:$$$(imageSizeIntOrIVec) = OpImageQuerySize $image;
                };
            }
        }
        [__NoSideEffect]
        [ForceInline]
        [require(glsl_spirv, image_loadstore)]
        public $(targetType.prefix)vec4 imageLoad(readonly $(IMAGE_PARAMS))
        {
            __target_switch
            {
            case glsl: __intrinsic_asm "imageLoad($*0)";
            case spirv:
            {
                $(SPV_DEFAULT_SAMPLE_VAR_IF_MISSING)
                return spirv_asm
                {
                    result:$$$(targetType.prefix)vec4 = OpImageRead $(SPV_PREFIX_IMAGE_PARAMS) $(SPV_SUFFIX_IMAGE_PARAMS);
                };
            }
            }
        }
        [ForceInline]
        [require(glsl_spirv, image_loadstore)]
        public void imageStore(writeonly $(IMAGE_PARAMS), $(targetType.prefix)vec4 data)
        {
            __target_switch
            {
            case glsl: __intrinsic_asm "imageStore($*0)";
            case spirv:
            {
                $(SPV_DEFAULT_SAMPLE_VAR_IF_MISSING)
                return spirv_asm
                {
                    OpImageWrite $(SPV_PREFIX_IMAGE_PARAMS) $data $(SPV_SUFFIX_IMAGE_PARAMS);
                };
            }
            }
        }

${{{{
        if (targetShape.isMS)
        {
}}}}
            // readonly writeonly in GLSL means an object only allows information queries.
            [__readNone]
            [ForceInline]
            [require(glsl_spirv, image_samples)]
            public int imageSamples(readonly writeonly $(fullTypeName) image)
            {
                __target_switch
                {
                case glsl: __intrinsic_asm "imageSamples($*0)";
                case spirv:
                    return spirv_asm
                    {
                        OpCapability ImageQuery;
                        result:$$int = OpImageQuerySamples $image;
                    };
                }
            }
${{{{
        }
}}}}
        [ForceInline]
        [require(glsl_spirv, image_loadstore)]
        public $(targetType.type) imageAtomicAdd($(IMAGE_PARAMS), $(targetType.type) data)
        {
            typeRequireChecks_image_atomic_tier1<$(targetType.type)>();
            __target_switch
            {
            case glsl: __intrinsic_asm "imageAtomicAdd($*0)";
            case spirv:
            {
                let imageP = __getLegalizedSPIRVGlobalParamAddr(image);
                typeRequireChecks_image_atomic_int64<$(targetType.type)>();
                $(SPV_DEFAULT_SAMPLE_VAR_IF_MISSING)
                return spirv_asm
                {
                    OpCapability ImageQuery;
                    %ptrType = OpTypePointer Image $$$(targetType.type);
                    %ptr:%ptrType = OpImageTexelPointer $imageP $P $sample;
                    result:$$$(targetType.type) = OpAtomic$(targetType.SPVTypePrefix)Add$(targetType.SPVAtomicSuffix) %ptr Device None $data
                };
            }
            }
        }
        [ForceInline]
        [require(glsl_spirv, image_loadstore)]
        public $(targetType.type) imageAtomicExchange($(IMAGE_PARAMS), $(targetType.type) data)
        {
            typeRequireChecks_image_atomic_tier1<$(targetType.type)>();
            __target_switch
            {
            case glsl: __intrinsic_asm "imageAtomicExchange($*0)";
            case spirv:
            {
                let imageP = __getLegalizedSPIRVGlobalParamAddr(image);
                typeRequireChecks_image_atomic_int64<$(targetType.type)>();
                $(SPV_DEFAULT_SAMPLE_VAR_IF_MISSING)
                return spirv_asm
                {
                    OpCapability ImageQuery;
                    %ptrType = OpTypePointer Image $$$(targetType.type);
                    %ptr:%ptrType = OpImageTexelPointer $imageP $P $sample;
                    result:$$$(targetType.type) = OpAtomicExchange %ptr Device ImageMemory $data
                };
            }
            }
        }
        [ForceInline]
        [require(glsl_spirv, image_loadstore)]
        public $(targetType.type) imageAtomicMin($(IMAGE_PARAMS), $(targetType.type) data)
        {
            typeRequireChecks_image_atomic_tier2<$(targetType.type)>();
            __target_switch
            {
            case glsl: __intrinsic_asm "imageAtomicMin($*0)";
            case spirv:
            {
                let imageP = __getLegalizedSPIRVGlobalParamAddr(image);
                typeRequireChecks_image_atomic_int64<$(targetType.type)>();
                $(SPV_DEFAULT_SAMPLE_VAR_IF_MISSING)
                return spirv_asm
                {
                    OpCapability ImageQuery;
                    %ptrType = OpTypePointer Image $$$(targetType.type);
                    %ptr:%ptrType = OpImageTexelPointer $imageP $P $sample;
                    result:$$$(targetType.type) = OpAtomic$(targetType.SPVSubTypePrefix)Min$(targetType.SPVAtomicSuffix) %ptr Device ImageMemory $data
                };
            }
            }
        }
        [ForceInline]
        [require(glsl_spirv, image_loadstore)]
        public $(targetType.type) imageAtomicMax($(IMAGE_PARAMS), $(targetType.type) data)
        {
            typeRequireChecks_image_atomic_tier2<$(targetType.type)>();
            __target_switch
            {
            case glsl: __intrinsic_asm "imageAtomicMax($*0)";
            case spirv:
            {
                let imageP = __getLegalizedSPIRVGlobalParamAddr(image);
                typeRequireChecks_image_atomic_int64<$(targetType.type)>();
                $(SPV_DEFAULT_SAMPLE_VAR_IF_MISSING)
                return spirv_asm
                {
                    OpCapability ImageQuery;
                    %ptrType = OpTypePointer Image $$$(targetType.type);
                    %ptr:%ptrType = OpImageTexelPointer $imageP $P $sample;
                    result:$$$(targetType.type) = OpAtomic$(targetType.SPVSubTypePrefix)Max$(targetType.SPVAtomicSuffix) %ptr Device ImageMemory $data
                };
            }
            }
        }
${{{{
        if (!targetType.isInt)
            continue;
}}}}
        [ForceInline]
        [require(glsl_spirv, image_loadstore)]
        public $(targetType.type) imageAtomicAnd($(IMAGE_PARAMS), $(targetType.type) data)
        {
            __target_switch
            {
            case glsl: __intrinsic_asm "imageAtomicAnd($*0)";
            case spirv:
            {
                let imageP = __getLegalizedSPIRVGlobalParamAddr(image);
                typeRequireChecks_image_atomic_int64<$(targetType.type)>();
                $(SPV_DEFAULT_SAMPLE_VAR_IF_MISSING)
                return spirv_asm
                {
                    OpCapability ImageQuery;
                    %ptrType = OpTypePointer Image $$$(targetType.type);
                    %ptr:%ptrType = OpImageTexelPointer $imageP $P $sample;
                    result:$$$(targetType.type) = OpAtomicAnd %ptr Device ImageMemory $data
                };
            }
            }
        }
        [ForceInline]
        [require(glsl_spirv, image_loadstore)]
        public $(targetType.type) imageAtomicOr($(IMAGE_PARAMS), $(targetType.type) data)
        {
            __target_switch
            {
            case glsl: __intrinsic_asm "imageAtomicOr($*0)";
            case spirv:
            {
                let imageP = __getLegalizedSPIRVGlobalParamAddr(image);
                typeRequireChecks_image_atomic_int64<$(targetType.type)>();
                $(SPV_DEFAULT_SAMPLE_VAR_IF_MISSING)
                return spirv_asm
                {
                    OpCapability ImageQuery;
                    %ptrType = OpTypePointer Image $$$(targetType.type);
                    %ptr:%ptrType = OpImageTexelPointer $imageP $P $sample;
                    result:$$$(targetType.type) = OpAtomicOr %ptr Device ImageMemory $data
                };
            }
            }
        }
        
        [ForceInline]
        [require(glsl_spirv, image_loadstore)]
        public $(targetType.type) imageAtomicXor($(IMAGE_PARAMS), $(targetType.type) data)
        {
            __target_switch
            {
            case glsl: __intrinsic_asm "imageAtomicXor($*0)";
            case spirv:
            {
                let imageP = __getLegalizedSPIRVGlobalParamAddr(image);
                typeRequireChecks_image_atomic_int64<$(targetType.type)>();
                $(SPV_DEFAULT_SAMPLE_VAR_IF_MISSING)
                return spirv_asm
                {
                    OpCapability ImageQuery;
                    %ptrType = OpTypePointer Image $$$(targetType.type);
                    %ptr:%ptrType = OpImageTexelPointer $imageP $P $sample;
                    result:$$$(targetType.type) = OpAtomicXor %ptr Device ImageMemory $data
                };
            }
            }
        }
        [ForceInline]
        [require(glsl_spirv, image_loadstore)]
        public $(targetType.type) imageAtomicCompSwap($(IMAGE_PARAMS), $(targetType.type) compare, $(targetType.type) data)
        {
            __target_switch
            {
            case glsl: __intrinsic_asm "imageAtomicCompSwap($*0)";
            case spirv:
            {
                let imageP = __getLegalizedSPIRVGlobalParamAddr(image);
                typeRequireChecks_image_atomic_int64<$(targetType.type)>();
                $(SPV_DEFAULT_SAMPLE_VAR_IF_MISSING)
                return spirv_asm
                {
                    OpCapability ImageQuery;
                    %ptrType = OpTypePointer Image $$$(targetType.type);
                    %ptr:%ptrType = OpImageTexelPointer $imageP $P $sample;
                    result:$$$(targetType.type) = OpAtomicCompareExchange %ptr Device ImageMemory ImageMemory $data $compare
                };
            }
            }
        }
${{{{
    }
}
}}}}

//// RayTracing

// Ray Tracing variables

public typealias rayQueryEXT = RayQuery;
public typealias accelerationStructureEXT = RaytracingAccelerationStructure;
public typealias accelerationStructureNV = RaytracingAccelerationStructure;

public typealias hitObjectNV = HitObject;

//GL_EXT_ray_tracing BuiltIn's 

[require(any_target, raytracing_allstages)]
void requireGLSLExtForRayTracingBuiltin() 
{
    __target_switch
    {
    case glsl:
        __requireTargetExtension("GL_EXT_ray_tracing");
        __intrinsic_asm "";
    default:
        return;
    }
}

__spirv_version(1.4) 
[require(any_target, raytracing_allstages)]
void setupExtForRayTracingBuiltIn() 
{
    __target_switch
    {
    case glsl:
        requireGLSLExtForRayTracingBuiltin();
    default:
        return;
    }
}

public property uint3 gl_LaunchIDNV 
{
    [require(glsl_spirv, raytracing_allstages)]
    get 
    {
        setupExtForRayTracingBuiltIn();
        __target_switch
        {
        case glsl:
        {
            __intrinsic_asm "(gl_LaunchIDNV)";
        }
        case spirv:
        {
            return spirv_asm 
            {
                result:$$uint3 = OpLoad builtin(LaunchIdNV:uint3);
            };
        }
        }    
    }
}

public property uint3 gl_LaunchIDEXT 
{
    [require(cuda_glsl_hlsl_spirv, raytracing_allstages)]
    get 
    {
        setupExtForRayTracingBuiltIn();
        return DispatchRaysIndex();
    }
}

public property uint3 gl_LaunchSizeNV 
{
    [require(glsl_spirv, raytracing_allstages)]
    get 
    {
        setupExtForRayTracingBuiltIn();
        __target_switch
        {
        case glsl:
        {
            __intrinsic_asm "(gl_LaunchSizeNV)";
        }
        case spirv:
        {
            return spirv_asm 
            {
                result:$$uint3 = OpLoad builtin(LaunchSizeNV:uint3);
            };
        }
        }
    }
}

public property uint3 gl_LaunchSizeEXT 
{
    [require(cuda_glsl_hlsl_spirv, raytracing_allstages)]
    get 
    {
        setupExtForRayTracingBuiltIn();
        return DispatchRaysDimensions();
    }
}

internal in int __gl_PrimitiveID : SV_PrimitiveID;

public property int gl_PrimitiveID 
{
    [require(cuda_glsl_hlsl_spirv)]
    get 
    {
        __stage_switch
        {
        case anyhit:
        case closesthit:
        case intersection:
            setupExtForRayTracingBuiltIn();
            return PrimitiveIndex();
        default:
            return __gl_PrimitiveID;   
        }
    }
}

public property int gl_InstanceID 
{
    [require(cuda_glsl_hlsl_spirv)]
    get 
    {
        __stage_switch
        {
        case anyhit:
        case closesthit:
        case intersection:
            setupExtForRayTracingBuiltIn();
            return InstanceIndex();
        default:
            return gl_InstanceIndex;
        }
    }
}

public property int gl_InstanceCustomIndexEXT 
{
    [require(cuda_glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection)]
    get 
    {
        setupExtForRayTracingBuiltIn();
        return InstanceID();
    }
}

public property int gl_GeometryIndexEXT 
{
    [require(glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection)]
    get
    {
        setupExtForRayTracingBuiltIn();
        return GeometryIndex();
    }
}

public property vec3 gl_WorldRayOriginEXT 
{
    [require(glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection_miss)]
    get
    {
        setupExtForRayTracingBuiltIn();
        return WorldRayOrigin();
    }
}

public property vec3 gl_WorldRayDirectionEXT 
{
    [require(glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection_miss)]
    get
    {
        setupExtForRayTracingBuiltIn();
        return WorldRayDirection();
    }
}

public property vec3 gl_ObjectRayOriginEXT 
{
    [require(glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection)]
    get
    {
        setupExtForRayTracingBuiltIn();
        return ObjectRayOrigin();
    }
}

public property vec3 gl_ObjectRayDirectionEXT 
{
    [require(glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection)]
    get
    {
        setupExtForRayTracingBuiltIn();
        return ObjectRayDirection();
    }
}

public property float gl_RayTminEXT 
{
    [require(glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection_miss)]
    get
    {
        setupExtForRayTracingBuiltIn();
        return RayTMin();
    }
}

public property float gl_RayTmaxEXT 
{
    [require(glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection_miss)]
    get
    {
        setupExtForRayTracingBuiltIn();
        return RayTCurrent();
    }
}

public property uint gl_IncomingRayFlagsEXT 
{
    [require(glsl_hlsl_spirv, raytracing_anyhit_closesthit_intersection_miss)]
    get
    {
        setupExtForRayTracingBuiltIn();
        return RayFlags();
    }
}

public property float gl_HitTEXT 
{
    [require(glsl_spirv, raytracing_anyhit_closesthit)]
    get
    {
        setupExtForRayTracingBuiltIn();
        __target_switch
        {
        case glsl:
        {
            __intrinsic_asm "(gl_HitTEXT)";
        }
        case spirv:
        {
            return spirv_asm 
            {
                result:$$float = OpLoad builtin(RayTmaxKHR:float);
            };
        }
        }
    }
}

public property uint gl_HitKindEXT 
{

    [require(glsl_hlsl_spirv, raytracing_anyhit_closesthit)]
    get
    {
        setupExtForRayTracingBuiltIn();
        return HitKind();
    }
}

public property mat4x3 gl_ObjectToWorldEXT 
{
    [require(glsl_spirv, raytracing_anyhit_closesthit_intersection)]
    get
    {
        setupExtForRayTracingBuiltIn();
        __target_switch
        {
        case glsl:
        {
            __intrinsic_asm "(gl_ObjectToWorldEXT)";
        }
        case spirv:
        {
            return spirv_asm
            {
                result:$$mat4x3 = OpLoad builtin(ObjectToWorldKHR:mat4x3);
            };
        }
        }
    }
}

public property mat3x4 gl_ObjectToWorld3x4EXT 
{
    [require(glsl_spirv, raytracing_anyhit_closesthit_intersection)]
    get
    {
        setupExtForRayTracingBuiltIn();
        __target_switch
        {
        case glsl:
        {
            __intrinsic_asm "(gl_ObjectToWorld3x4EXT)";
        }
        case spirv:
        {
            return spirv_asm 
            {
                %mat:$$mat4x3 = OpLoad builtin(ObjectToWorldKHR:mat4x3);
                result:$$mat3x4 = OpTranspose %mat
            };
        }
        }
    }
}

public property mat4x3 gl_WorldToObjectEXT 
{
    [require(glsl_spirv, raytracing_anyhit_closesthit_intersection)]
    get
    {
        setupExtForRayTracingBuiltIn();
        __target_switch
        {
        case glsl:
        {
            __intrinsic_asm "(gl_WorldToObjectEXT)";
        }
        case spirv:
        {
            return spirv_asm 
            {
                result:$$mat4x3 = OpLoad builtin(WorldToObjectKHR:mat4x3);
            };
        }
        }
    }
}

public property mat3x4 gl_WorldToObject3x4EXT 
{
    [require(glsl_spirv, raytracing_anyhit_closesthit_intersection)]
    get
    {
        setupExtForRayTracingBuiltIn();
        __target_switch
        {
        case glsl:
        {
            __intrinsic_asm "(gl_WorldToObject3x4EXT)";
        }
        case spirv:
        {
            return spirv_asm 
            {
                %mat:$$mat4x3 = OpLoad builtin(WorldToObjectKHR:mat4x3);
                result:$$mat3x4 = OpTranspose %mat
            };
        }
        }
    }
}

// GL_EXT_ray_tracing functions

__glsl_extension(GL_EXT_ray_tracing)
[require(glsl_spirv, raytracing_raygen_closesthit_miss)]
public void traceRayEXT(
    accelerationStructureEXT topLevel,
    uint rayFlags,
    uint cullMask,
    uint sbtRecordOffset,
    uint sbtRecordStride,
    uint missIndex,
    vec3 origin,
    float Tmin,
    vec3 direction,
    float Tmax,
    constexpr int payload)
{
    __target_switch
    {
    case glsl:
    {
        __traceRay(
            topLevel,
            rayFlags,
            cullMask,
            sbtRecordOffset,
            sbtRecordStride,
            missIndex,
            origin,
            Tmin,
            direction,
            Tmax,
            payload);
    }
    case spirv:
    {
        spirv_asm 
        {
            OpTraceRayKHR 
                /**/ $topLevel
                /**/ $rayFlags
                /**/ $cullMask
                /**/ $sbtRecordOffset
                /**/ $sbtRecordStride
                /**/ $missIndex
                /**/ $origin
                /**/ $Tmin
                /**/ $direction
                /**/ $Tmax
                /**/ __rayPayloadFromLocation(payload);
        };
    }
    }
}

__glsl_extension(GL_EXT_ray_tracing)
[require(glsl_spirv, raytracing_intersection)]
public bool reportIntersectionEXT(float hitT, uint hitKind)
{
    return __reportIntersection(hitT, hitKind);
}

public property int terminateRayEXT
{
    [require(glsl_spirv, raytracing_anyhit)]
    [ForceInline]
    get
    {
        setupExtForRayTracingBuiltIn();
        AcceptHitAndEndSearch();
        return 0;
    }
}

public property int ignoreIntersectionEXT
{
    [require(glsl_spirv, raytracing_anyhit)]
    [ForceInline]
    get
    {
        setupExtForRayTracingBuiltIn();
        IgnoreHit();
        return 0;
    }
}

__glsl_extension(GL_EXT_ray_tracing)
[require(glsl_spirv, raytracing_raygen_closesthit_miss_callable)]
public void executeCallableEXT(
    uint sbtRecordIndex, 
    int callable /*callableDataEXT and callableDataInEXT*/)
{
    __target_switch
    {
    case glsl:
    {
        __executeCallable(sbtRecordIndex, callable);
    }
    case spirv:
    {
        spirv_asm {
            OpExecuteCallableKHR $sbtRecordIndex __rayCallableFromLocation(callable)
        };
    }
    }
}

// GL_EXT_ray_tracing constants

public static const uint gl_HitKindFrontFacingTriangleEXT = 0xFEU;
public static const uint gl_HitKindBackFacingTriangleEXT = 0xFFU;

/// GL_EXT_ray_query

// GL_EXT_ray_query constants

public static const uint gl_RayFlagsNoneEXT = 0U;
public static const uint gl_RayFlagsOpaqueEXT = 1U;
public static const uint gl_RayFlagsNoOpaqueEXT = 2U;
public static const uint gl_RayFlagsTerminateOnFirstHitEXT = 4U;
public static const uint gl_RayFlagsSkipClosestHitShaderEXT = 8U;
public static const uint gl_RayFlagsCullBackFacingTrianglesEXT = 16U;
public static const uint gl_RayFlagsCullFrontFacingTrianglesEXT = 32U;
public static const uint gl_RayFlagsCullOpaqueEXT = 64U;
public static const uint gl_RayFlagsCullNoOpaqueEXT = 128U;

public static const uint gl_RayQueryCommittedIntersectionNoneEXT = 0U;
public static const uint gl_RayQueryCommittedIntersectionTriangleEXT = 1U;
public static const uint gl_RayQueryCommittedIntersectionGeneratedEXT = 2U;

public static const uint gl_RayQueryCandidateIntersectionTriangleEXT = 0U;
public static const uint gl_RayQueryCandidateIntersectionAABBEXT = 1U;

// GL_EXT_ray_query functions

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public void rayQueryInitializeEXT(
    inout rayQueryEXT q,
    accelerationStructureEXT topLevel,
    uint rayFlags, uint cullMask, vec3 origin,
    float tMin, vec3 direction, float tMax)
{
    q.TraceRayInline(
        topLevel, 
        rayFlags, 
        cullMask, 
        { origin, tMin, direction, tMax });
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public bool rayQueryProceedEXT(inout rayQueryEXT q)
{
    return q.Proceed();
}

__glsl_extension(GL_EXT_ray_query)
[require(glsl_spirv, rayquery)]
[mutating]
[ForceInline]
public void rayQueryTerminateEXT(inout rayQueryEXT q)
{
    q.Abort();
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public void rayQueryGenerateIntersectionEXT(inout rayQueryEXT q, float tHit)
{
    q.CommitProceduralPrimitiveHit(tHit);
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public void rayQueryConfirmIntersectionEXT(inout rayQueryEXT q)
{
    q.CommitNonOpaqueTriangleHit();
}

__glsl_extension(GL_EXT_ray_query)
[__NoSideEffect]
[ForceInline]
[require(glsl_spirv, rayquery)]
public uint rayQueryGetIntersectionTypeEXT(rayQueryEXT q, bool committed)
{
    if (committed)
    {
        return q.CommittedStatus();
    }
    else
    {
        return q.CandidateType();
    }
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public float rayQueryGetRayTMinEXT(rayQueryEXT q)
{
    return q.RayTMin();
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public uint rayQueryGetRayFlagsEXT(rayQueryEXT q)
{
    return q.RayFlags();
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public vec3 rayQueryGetWorldRayOriginEXT(rayQueryEXT q)
{
    return q.WorldRayOrigin();
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public vec3 rayQueryGetWorldRayDirectionEXT(rayQueryEXT q)
{
    return q.WorldRayDirection();
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public float rayQueryGetIntersectionTEXT(rayQueryEXT q, bool committed) 
{
    if (committed) 
    {
        return q.CommittedRayT();
    }
    else 
    {
        return q.CandidateTriangleRayT();
    }
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public int rayQueryGetIntersectionInstanceCustomIndexEXT(rayQueryEXT q, bool committed) 
{
    if (committed) 
    {
        return q.CommittedRayInstanceCustomIndex();
    }
    else 
    {
        return q.CandidateRayInstanceCustomIndex();;
    }
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public int rayQueryGetIntersectionInstanceIdEXT(rayQueryEXT q, bool committed) 
{
    if (committed) 
    {
        return q.CommittedRayInstanceId();
    }
    else 
    {
        return q.CandidateRayInstanceId();
    }
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public uint rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(rayQueryEXT q, bool committed) 
{
    if (committed) 
    {
        return q.CommittedRayInstanceShaderBindingTableRecordOffset();
    }
    else 
    {
        return q.CandidateRayInstanceShaderBindingTableRecordOffset();    
    }
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public int rayQueryGetIntersectionGeometryIndexEXT(rayQueryEXT q, bool committed) 
{
    if (committed) 
    {
        return q.CommittedRayGeometryIndex();
    }
    else 
    {
        return q.CandidateRayGeometryIndex();
    }
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public int rayQueryGetIntersectionPrimitiveIndexEXT(rayQueryEXT q, bool committed) 
{
    if (committed) 
    {
        return q.CommittedRayPrimitiveIndex();
    }
    else 
    {
        return q.CandidateRayPrimitiveIndex();
    }
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public vec2 rayQueryGetIntersectionBarycentricsEXT(rayQueryEXT q, bool committed) 
{
    if (committed) 
    {
        return q.CommittedRayBarycentrics();
    }
    else 
    {
        return q.CandidateRayBarycentrics();
    }
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public bool rayQueryGetIntersectionFrontFaceEXT(rayQueryEXT q, bool committed) 
{
    if (committed) 
    {
        return q.CommittedRayFrontFace();
    }
    else 
    {
        return q.CandidateRayFrontFace();
    }
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public bool rayQueryGetIntersectionCandidateAABBOpaqueEXT(rayQueryEXT q) 
{
    return q.CandidateProceduralPrimitiveNonOpaque();
}

[ForceInline]
[require(glsl_spirv, rayquery)]
public vec3 rayQueryGetIntersectionObjectRayDirectionEXT(rayQueryEXT q, bool committed) 
{
    if (committed) 
    {
        return q.CommittedRayObjectRayDirection();
    }
    else 
    {
        return q.CandidateRayObjectRayDirection();
    }
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public vec3 rayQueryGetIntersectionObjectRayOriginEXT(rayQueryEXT q, bool committed) 
{
    if (committed) 
    {
        return q.CommittedRayObjectRayOrigin();
    }
    else 
    {
        return q.CandidateRayObjectRayOrigin();
    }
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public mat4x3 rayQueryGetIntersectionObjectToWorldEXT(rayQueryEXT q, bool committed) 
{
    if (committed) 
    {
        return q.CommittedRayObjectToWorld();
    }
    else 
    {
        return q.CandidateRayObjectToWorld();
    }
}

__glsl_extension(GL_EXT_ray_query)
[ForceInline]
[require(glsl_spirv, rayquery)]
public mat4x3 rayQueryGetIntersectionWorldToObjectEXT(rayQueryEXT q, bool committed) 
{
    if (committed) 
    {
        return q.CommittedRayWorldToObject();
    }
    else 
    {
        return q.CandidateRayWorldToObject();
    }
}

/// GL_NV_shader_invocation_reorder

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public void hitObjectTraceRayNV(
    inout hitObjectNV hitObject,
    accelerationStructureEXT topLevel,
    uint rayFlags,
    uint cullMask,
    uint sbtRecordOffset,
    uint sbtRecordStride,
    uint missIndex,
    vec3 origin,
    float Tmin,
    vec3 direction,
    float Tmax,
    constexpr int payload) 
    {
    __target_switch
    {
    case glsl:
    {
        HitObject::__glslTraceRay(
            hitObject,
            topLevel,
            rayFlags,
            cullMask,
            sbtRecordOffset,
            sbtRecordStride,
            missIndex,
            origin,
            Tmin,
            direction,
            Tmax,
            payload);
    }
    case spirv:
    {
        spirv_asm
        {
            OpHitObjectTraceRayNV
                &hitObject
                /**/ $topLevel
                /**/ $rayFlags
                /**/ $cullMask
                /**/ $sbtRecordOffset
                /**/ $sbtRecordStride
                /**/ $missIndex
                /**/ $origin
                /**/ $Tmin
                /**/ $direction
                /**/ $Tmax
                /**/ __rayPayloadFromLocation(payload)
        };
    }
    }
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
__glsl_extension(GL_NV_ray_tracing_motion_blur)
[ForceInline]
[require(glsl_spirv, ser_motion_raygen_closesthit_miss)]
public void hitObjectTraceRayMotionNV(
    inout hitObjectNV hitObject, 
    accelerationStructureEXT topLevel,
    uint rayFlags,
    uint cullMask,
    uint sbtRecordOffset,
    uint sbtRecordStride,
    uint missIndex,
    vec3 origin,
    float Tmin,
    vec3 direction,
    float Tmax,
    float currentTime,
    constexpr int payload)
    {
    __target_switch
    {
    case glsl:
    {
        HitObject::__glslTraceMotionRay(
            hitObject,
            topLevel,
            rayFlags,
            cullMask,
            sbtRecordOffset,
            sbtRecordStride,
            missIndex,
            origin,
            Tmin,
            direction,
            Tmax,
            currentTime,
            payload);
    }
    case spirv:
    {
        spirv_asm
        {
            OpCapability RayTracingMotionBlurNV;
            OpExtension "SPV_NV_ray_tracing_motion_blur";
            OpHitObjectTraceRayMotionNV
                /**/ &hitObject 
                /**/ $topLevel
                /**/ $rayFlags
                /**/ $cullMask
                /**/ $sbtRecordOffset
                /**/ $sbtRecordStride
                /**/ $missIndex
                /**/ $origin
                /**/ $Tmin
                /**/ $direction
                /**/ $Tmax
                /**/ $currentTime
                /**/ __rayPayloadFromLocation(payload)
        };
    }
    }
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public void hitObjectRecordHitNV(
    inout hitObjectNV hitObject,
    accelerationStructureEXT topLevel,
    int instanceid,
    int primitiveid,
    int geometryindex,
    uint hitKind,
    uint sbtRecordOffset,
    uint sbtRecordStride,
    vec3 origin,
    float Tmin,
    vec3 direction,
    float Tmax,
    constexpr int attributeLocation)
    {
    __target_switch
    {
    case glsl:
    {
        HitObject::__glslMakeHit(
            hitObject,
            topLevel,
            instanceid,
            primitiveid,
            geometryindex,
            hitKind,
            sbtRecordOffset,
            sbtRecordStride,
            origin,
            Tmin,
            direction,
            Tmax,
            attributeLocation);
    }
    case spirv:
    {
        spirv_asm
        {
            OpHitObjectRecordHitNV
                /**/ &hitObject
                /**/ $topLevel
                /**/ $instanceid
                /**/ $primitiveid
                /**/ $geometryindex
                /**/ $hitKind
                /**/ $sbtRecordOffset
                /**/ $sbtRecordStride
                /**/ $origin
                /**/ $Tmin
                /**/ $direction
                /**/ $Tmax
                /**/ __rayAttributeFromLocation(attributeLocation)
        };
    }
    }
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
__glsl_extension(GL_NV_ray_tracing_motion_blur)
[ForceInline]
[require(glsl_spirv, ser_motion_raygen_closesthit_miss)]
public void hitObjectRecordHitMotionNV(
    inout hitObjectNV hitObject,
    accelerationStructureEXT topLevel,
    int instanceid,
    int primitiveid,
    int geometryindex,
    uint hitKind,
    uint sbtRecordOffset,
    uint sbtRecordStride,
    vec3 origin,
    float Tmin,
    vec3 direction,
    float Tmax,
    float currentTime,
    constexpr int attributeLocation)
    {
    __target_switch
    {
    case glsl:
    {
        HitObject::__glslMakeMotionHit(
            hitObject,
            topLevel,
            instanceid,
            primitiveid,
            geometryindex,
            hitKind,
            sbtRecordOffset,
            sbtRecordStride,
            origin,
            Tmin,
            direction,
            Tmax,
            currentTime,
            attributeLocation);
    }
    case spirv:
    {
        spirv_asm
        {
            OpHitObjectRecordHitMotionNV
                /**/ &hitObject
                /**/ $topLevel
                /**/ $instanceid
                /**/ $primitiveid
                /**/ $geometryindex
                /**/ $hitKind
                /**/ $sbtRecordOffset
                /**/ $sbtRecordStride
                /**/ $origin
                /**/ $Tmin
                /**/ $direction
                /**/ $Tmax
                /**/ $currentTime
                /**/ __rayAttributeFromLocation(attributeLocation)
        };
    }
    }
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public void hitObjectRecordHitWithIndexNV(
    inout hitObjectNV hitObject,
    accelerationStructureEXT topLevel,
    int instanceid,
    int primitiveid,
    int geometryindex,
    uint hitKind,
    uint sbtRecordIndex,
    vec3 origin,
    float Tmin,
    vec3 direction,
    float Tmax,
    constexpr int attributeLocation)
{
    __target_switch
    {
    case glsl:
    {
        HitObject::__glslMakeHitWithIndex(
            hitObject,
            topLevel,
            instanceid,
            primitiveid,
            geometryindex,
            hitKind,
            sbtRecordIndex,
            origin,
            Tmin,
            direction,
            Tmax,
            attributeLocation);
    }
    case spirv:
    {
        spirv_asm
        {
            OpHitObjectRecordHitWithIndexNV
                /**/ &hitObject
                /**/ $topLevel
                /**/ $instanceid
                /**/ $primitiveid
                /**/ $geometryindex
                /**/ $hitKind
                /**/ $sbtRecordIndex
                /**/ $origin
                /**/ $Tmin
                /**/ $direction
                /**/ $Tmax
                /**/ __rayAttributeFromLocation(attributeLocation)
        };
    }
    }
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
__glsl_extension(GL_NV_ray_tracing_motion_blur)
[ForceInline]
[require(glsl_spirv, ser_motion_raygen_closesthit_miss)]
public void hitObjectRecordHitWithIndexMotionNV(
    inout hitObjectNV hitObject,
    accelerationStructureEXT topLevel,
    int instanceid,
    int primitiveid,
    int geometryindex,
    uint hitKind,
    uint sbtRecordIndex,
    vec3 origin,
    float Tmin,
    vec3 direction,
    float Tmax,
    float currentTime,
    constexpr int attributeLocation)
    {
    __target_switch
    {
    case glsl:
    {
        HitObject::__glslMakeMotionHitWithIndex(
            hitObject,
            topLevel,
            instanceid,
            primitiveid,
            geometryindex,
            hitKind,
            sbtRecordIndex,
            origin,
            Tmin,
            direction,
            Tmax,
            currentTime,
            attributeLocation);
    }
    case spirv:
    {
        spirv_asm
        {
            OpCapability RayTracingMotionBlurNV;
            OpExtension "SPV_NV_ray_tracing_motion_blur";
            OpHitObjectRecordHitWithIndexMotionNV
                /**/ &hitObject
                /**/ $topLevel
                /**/ $instanceid
                /**/ $primitiveid
                /**/ $geometryindex
                /**/ $hitKind
                /**/ $sbtRecordIndex
                /**/ $origin
                /**/ $Tmin
                /**/ $direction
                /**/ $Tmax
                /**/ $currentTime
                /**/ __rayAttributeFromLocation(attributeLocation)
        };
    }
    }
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public void hitObjectRecordMissNV(
    inout hitObjectNV hitObject, 
    uint sbtRecordIndex, 
    vec3 origin,
    float Tmin,
    vec3 direction,
    float Tmax)
{
    hitObject = HitObject::MakeMiss(
        sbtRecordIndex,
        { origin, Tmin, direction, Tmax }
        );
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
__glsl_extension(GL_NV_ray_tracing_motion_blur)
[ForceInline]
[require(glsl_spirv, ser_motion_raygen_closesthit_miss)]
public void hitObjectRecordMissMotionNV(
    inout hitObjectNV hitObject, 
    uint sbtRecordIndex, 
    vec3 origin,
    float Tmin,
    vec3 direction,
    float Tmax,
    float currentTime)
{
    hitObject = HitObject::MakeMotionMiss(
        sbtRecordIndex,
        { origin, Tmin, direction, Tmax },
        currentTime
        );
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public void hitObjectRecordEmptyNV(hitObjectNV hitObject)
{
    hitObject = HitObject::MakeNop();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public void hitObjectExecuteShaderNV(
    inout hitObjectNV hitObject, 
    constexpr int payload)
{    
    __target_switch 
    {
    case glsl:
    {
        HitObject::__glslInvoke(hitObject, payload);
    }
    case spirv:
    {
        spirv_asm 
        {
            OpHitObjectExecuteShaderNV &hitObject __rayPayloadFromLocation(payload)
        };
    }
    }
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public bool hitObjectIsEmptyNV(hitObjectNV hitObject)
{
    return hitObject.IsNop();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public bool hitObjectIsMissNV(hitObjectNV hitObject)
{
    return hitObject.IsMiss();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public bool hitObjectIsHitNV(hitObjectNV hitObject)
{
    return hitObject.IsHit();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public float hitObjectGetRayTMinNV(hitObjectNV hitObject)
{
    return hitObject.GetRayDesc().TMin;  
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public float hitObjectGetRayTMaxNV(hitObjectNV hitObject)
{
    return hitObject.GetRayDesc().TMax; 
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public vec3 hitObjectGetWorldRayOriginNV(hitObjectNV hitObject)
{
    return hitObject.GetRayDesc().Origin;
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public vec3 hitObjectGetWorldRayDirectionNV(hitObjectNV hitObject)
{
    return hitObject.GetRayDesc().Direction; 
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public vec3 hitObjectGetObjectRayOriginNV(hitObjectNV hitObject)
{
    return hitObject.GetObjectRayOrigin();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public vec3 hitObjectGetObjectRayDirectionNV(hitObjectNV hitObject)
{
    return hitObject.GetObjectRayDirection();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public mat4x3 hitObjectGetObjectToWorldNV(hitObjectNV hitObject)
{
    return hitObject.GetObjectToWorld();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public mat4x3 hitObjectGetWorldToObjectNV(hitObjectNV hitObject)
{
    return hitObject.GetWorldToObject(); 
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public int hitObjectGetInstanceCustomIndexNV(hitObjectNV hitObject)
{
    return hitObject.GetInstanceID();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public int hitObjectGetInstanceIdNV(hitObjectNV hitObject)
{
    return hitObject.GetInstanceIndex();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public int hitObjectGetGeometryIndexNV(hitObjectNV hitObject)
{
    return hitObject.GetGeometryIndex();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public int hitObjectGetPrimitiveIndexNV(hitObjectNV hitObject)
{
    return hitObject.GetPrimitiveIndex();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public uint hitObjectGetHitKindNV(hitObjectNV hitObject)
{
    return hitObject.GetHitKind();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public void hitObjectGetAttributesNV(
    inout hitObjectNV hitObject, 
    constexpr int attributeLocation)
{
    __target_switch
    {
    case glsl:
    {
        __intrinsic_asm "hitObjectGetAttributesNV($0, $1)";
    }
    case spirv:
    {
        spirv_asm 
        {
            OpCapability ShaderInvocationReorderNV; 
            OpHitObjectGetAttributesNV &hitObject __rayAttributeFromLocation(attributeLocation)
        };
    }
    }
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public uvec2 hitObjectGetShaderRecordBufferHandleNV(hitObjectNV hitObject)
{
    return hitObject.GetShaderRecordBufferHandle();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public uint hitObjectGetShaderBindingTableRecordIndexNV(hitObjectNV hitObject)
{
    return hitObject.GetShaderTableIndex();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen_closesthit_miss)]
public float hitObjectGetCurrentTimeNV(hitObjectNV hitObject)
{
    return hitObject.GetCurrentTime();
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen)]
public void reorderThreadNV(uint hint, uint bits)
{
    ReorderThread(hint, bits);
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen)]
public void reorderThreadNV(hitObjectNV hitObject)
{
    ReorderThread(hitObject);
}

__glsl_extension(GL_EXT_ray_tracing)
__glsl_extension(GL_NV_shader_invocation_reorder)
__glsl_extension(GLSL_EXT_buffer_reference_uvec2)
[ForceInline]
[require(glsl_spirv, ser_raygen)]
public void reorderThreadNV(hitObjectNV hitObject, uint hint, uint bits)
{
    ReorderThread(hitObject, hint, bits);
}

/// GL_NV_ray_tracing_motion_blur

public property float gl_CurrentRayTimeNV
{
    [require(glsl_spirv, raytracing_motionblur_anyhit_closesthit_intersection_miss)]
    get
    {
        __target_switch
        {
        case glsl:
            __intrinsic_asm "(gl_CurrentRayTimeNV)";
        case spirv:
            return spirv_asm
            {
                OpCapability RayTracingMotionBlurNV;
                OpExtension "SPV_NV_ray_tracing_motion_blur";
                result:$$float = OpLoad builtin(CurrentRayTimeNV:float);
            };
        }
    }
}

__glsl_extension(GL_EXT_ray_tracing) 
__glsl_extension(GL_NV_ray_tracing_motion_blur)
[ForceInline]
[require(glsl_spirv, raytracing_motionblur_raygen_closesthit_miss)]
public void traceRayMotionNV(
    accelerationStructureEXT topLevel,
    uint rayFlags,
    uint cullMask,
    uint sbtRecordOffset,
    uint sbtRecordStride,
    uint missIndex,
    vec3 origin,
    float Tmin,
    vec3 direction,
    float Tmax,
    float currentTime,
    constexpr int payload)
{
    __target_switch 
    {
    case glsl:
    {
        __traceMotionRay(
            topLevel,
            rayFlags,
            cullMask,
            sbtRecordOffset,
            sbtRecordStride,
            missIndex,
            origin,
            Tmin,
            direction,
            Tmax,
            currentTime,
            payload);
    }
    case spirv:
    {
        spirv_asm 
        {
            OpCapability RayTracingMotionBlurNV;
            OpExtension "SPV_NV_ray_tracing_motion_blur";
            OpTraceRayMotionNV
                /**/ $topLevel
                /**/ $rayFlags
                /**/ $cullMask
                /**/ $sbtRecordOffset
                /**/ $sbtRecordStride
                /**/ $missIndex
                /**/ $origin
                /**/ $Tmin
                /**/ $direction
                /**/ $Tmax
                /**/ $currentTime
                /**/ __rayPayloadFromLocation(payload)
        };
    }
    }
}

// GL_KHR_shader_subgroup_basic Built-in Variables

[require(cpp_cuda_glsl_hlsl_spirv_wgsl, subgroup_basic)]
void requireGLSLExtForSubgroupBasicBuiltin() {
    __target_switch
    {
    case glsl:
        __requireTargetExtension("GL_KHR_shader_subgroup_basic");
        __intrinsic_asm "";
    default:
        return;
    }
}

[require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_basic)]
void setupExtForSubgroupBasicBuiltIn() {
    __target_switch
    {
    case glsl:
        requireGLSLExtForSubgroupBasicBuiltin();
    default:
        return;
    }
}

__spirv_version(1.3)
[require(cpp_cuda_glsl_hlsl_spirv_wgsl, subgroup_ballot)]
void requireGLSLExtForSubgroupBallotBuiltin() {
    __target_switch
    {
    case glsl:
        __requireTargetExtension("GL_KHR_shader_subgroup_ballot");
        __intrinsic_asm "";
    default:
        return;
    }
}

__spirv_version(1.3)
[require(cpp_cuda_glsl_hlsl_spirv_wgsl, subgroup_ballot)]
void setupExtForSubgroupBallotBuiltIn() {
    __target_switch
    {
    case glsl:
        requireGLSLExtForSubgroupBallotBuiltin();
    default:
        return;
    }
}

public property uint gl_NumSubgroups {
    
    [require(glsl_spirv, subgroup_basic)]
    get {
        setupExtForSubgroupBasicBuiltIn();
        __target_switch
        {
        case glsl:
            __intrinsic_asm "(gl_NumSubgroups)";
        case spirv:
            return spirv_asm {
                    OpCapability GroupNonUniform;
                    result:$$uint = OpLoad builtin(NumSubgroups:uint);
                };
        }

    }
}

public property uint gl_SubgroupID
{
    [require(glsl_spirv, subgroup_basic)]
    get {
        setupExtForSubgroupBasicBuiltIn();
        __target_switch
        {
        case glsl:
            __intrinsic_asm "(gl_SubgroupID)";
        case spirv:
            return spirv_asm {
                        OpCapability GroupNonUniform;
                        result:$$uint = OpLoad builtin(SubgroupId:uint);
                    };
        }
    }
}

public property uint gl_SubgroupSize
{
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_basic)]
    get {
        setupExtForSubgroupBasicBuiltIn();
        return WaveGetLaneCount();
    }
}

public property uint gl_SubgroupInvocationID
{
    [ForceInline]
    [require(cpp_cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_basic)]
    get {
        setupExtForSubgroupBasicBuiltIn();
        return WaveGetLaneIndex();
    }
}

public property uvec4 gl_SubgroupEqMask
{
    [require(glsl_spirv, subgroup_basic_ballot)]
    get {
        setupExtForSubgroupBasicBuiltIn();
        setupExtForSubgroupBallotBuiltIn();
        __target_switch
        {
        case glsl:
            __intrinsic_asm "(gl_SubgroupEqMask)";
        case spirv:
            return spirv_asm {
                        OpCapability GroupNonUniformBallot;
                        result:$$uvec4 = OpLoad builtin(SubgroupEqMask:uvec4);
                    };
        }
    }
}

public property uvec4 gl_SubgroupGeMask
{
    [require(glsl_spirv, subgroup_basic_ballot)]
    get {
        setupExtForSubgroupBasicBuiltIn();
        setupExtForSubgroupBallotBuiltIn();
        __target_switch
        {
        case glsl:
            __intrinsic_asm "(gl_SubgroupGeMask)";
        case spirv:
            return spirv_asm {
                        OpCapability GroupNonUniformBallot;
                        result:$$uvec4 = OpLoad builtin(SubgroupGeMask:uvec4);
                    };
        }
    }
}

public property uvec4 gl_SubgroupGtMask
{
    [require(glsl_spirv, subgroup_basic_ballot)]
    get {
        setupExtForSubgroupBasicBuiltIn();
        setupExtForSubgroupBallotBuiltIn();
        __target_switch
        {
        case glsl:
            __intrinsic_asm "(gl_SubgroupGtMask)";
        case spirv:
            return spirv_asm {
                        OpCapability GroupNonUniformBallot;
                        result:$$uvec4 = OpLoad builtin(SubgroupGtMask:uvec4);
                    };
        }
    }
}

public property uvec4 gl_SubgroupLeMask
{
    [require(glsl_spirv, subgroup_basic_ballot)]
    get {
        setupExtForSubgroupBasicBuiltIn();
        setupExtForSubgroupBallotBuiltIn();
        __target_switch
        {
        case glsl:
            __intrinsic_asm "(gl_SubgroupLeMask)";
        case spirv:
            return spirv_asm {
                        OpCapability GroupNonUniformBallot;
                        result:$$uvec4 = OpLoad builtin(SubgroupLeMask:uvec4);
                    };
        }
    }
}

public property uvec4 gl_SubgroupLtMask
{
    [require(glsl_spirv, subgroup_basic_ballot)]
    get {
        setupExtForSubgroupBasicBuiltIn();
        setupExtForSubgroupBallotBuiltIn();
        __target_switch
        {
        case glsl:
            __intrinsic_asm "(gl_SubgroupLtMask)";
        case spirv:
            return spirv_asm {
                        OpCapability GroupNonUniformBallot;
                        result:$$uvec4 = OpLoad builtin(SubgroupLtMask:uvec4);
                    };
        }
    }
}

// GL_KHR_shader_subgroup_basic

__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_basic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv, subgroup_basic)]
public void subgroupBarrier()
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__syncwarp()";
    case hlsl:
        __intrinsic_asm "AllMemoryBarrierWithGroupSync()";
    case glsl:
        __intrinsic_asm "subgroupBarrier()";
    case metal:
        __intrinsic_asm "simdgroup_barrier(mem_flags::mem_none)";
    case spirv:
        spirv_asm {
            OpCapability Shader;
            OpControlBarrier Subgroup Subgroup AcquireRelease|SubgroupMemory|ImageMemory|UniformMemory
        };

    }
}

__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_basic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv, subgroup_basic)]
public void subgroupMemoryBarrier()
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__threadfence_block()";
    case hlsl:
        __intrinsic_asm "AllMemoryBarrier()";
    case glsl:
        __intrinsic_asm "subgroupMemoryBarrier()";
    case metal:
        __intrinsic_asm "simdgroup_barrier(mem_flags::mem_device)";
    case spirv:
        spirv_asm {
            OpCapability Shader;
            OpMemoryBarrier Subgroup AcquireRelease|SubgroupMemory|ImageMemory|UniformMemory
        };

    }
}

__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_basic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv, subgroup_basic)]
public void subgroupMemoryBarrierBuffer()
{
    // the following implementation is NOT the same as DeviceMemoryBarrier
    // HLSL lacks the same granularity of blocking on subgroup memory within a subgroup
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__threadfence_block()";
    case hlsl:
        __intrinsic_asm "DeviceMemoryBarrier()";
    case glsl:
        __intrinsic_asm "subgroupMemoryBarrierBuffer()";
    case metal:
        __intrinsic_asm "simdgroup_barrier(mem_flags::mem_device)";
    case spirv:
        spirv_asm {
            OpCapability Shader;
            OpMemoryBarrier Subgroup AcquireRelease|UniformMemory
        };

    }
}

__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_basic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv, subgroup_basic)]
public void subgroupMemoryBarrierImage()
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__threadfence_block()";
    case hlsl:
        __intrinsic_asm "DeviceMemoryBarrier()";
    case glsl:
        __intrinsic_asm "subgroupMemoryBarrierImage()";
    case metal:
        __intrinsic_asm "simdgroup_barrier(mem_flags::mem_texture)";
    case spirv:
        spirv_asm {
            OpMemoryBarrier Subgroup AcquireRelease|ImageMemory
        };

    }
}

__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_basic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv, subgroup_basic)]
public void subgroupMemoryBarrierShared()
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "__threadfence_block()";
    case hlsl:
        __intrinsic_asm "GroupMemoryBarrier()";
    case glsl:
        __intrinsic_asm "subgroupMemoryBarrierShared()";
    case metal:
        __intrinsic_asm "simdgroup_barrier(mem_flags::mem_threadgroup)";
    case spirv:
        spirv_asm {
            // SubgroupMemory triggers vulkan validation layer error; 
            // WorkgroupMemory is the next level of granularity 
            OpMemoryBarrier Subgroup AcquireRelease|WorkgroupMemory
        };

    }
}

__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_basic) 
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_basic)]
public bool subgroupElect()
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "( (__activemask() & (__activemask()*-1)) == _getLaneId())";    
    default:
        return WaveIsFirstLane();

    }
}

// GL_KHR_shader_subgroup_vote

__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_vote)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_vote)]
public bool subgroupAll(bool value)
{
    return WaveActiveAllTrue(value);
}

__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_vote)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_vote)]
public bool subgroupAny(bool value)
{
    return WaveActiveAnyTrue(value);
}

__generic<T : __BuiltinType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_vote)
[ForceInline]
[require(cuda_glsl_hlsl_spirv, subgroup_vote)]
public bool subgroupAllEqual(T value)
{
    shader_subgroup_preamble<T>();
    return WaveActiveAllEqual(value);
}

__generic<T : __BuiltinType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_vote)
[ForceInline]
[require(cuda_glsl_hlsl_spirv, subgroup_vote)]
public bool subgroupAllEqual(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    return WaveActiveAllEqual(value);
}

// GL_KHR_shader_subgroup_arithmetic

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
public T subgroupAdd(T value)
{
    shader_subgroup_preamble<T>();
    return WaveActiveSum(value);
}

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
public T subgroupMul(T value)
{
    shader_subgroup_preamble<T>();
    return WaveActiveProduct(value);
}

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
public T subgroupMin(T value)
{
    shader_subgroup_preamble<T>();
    return WaveActiveMin(value);
}

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
public T subgroupMax(T value)
{
    shader_subgroup_preamble<T>();
    return WaveActiveMax(value);
}

__generic<T : __BuiltinLogicalType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_arithmetic)]
public T subgroupAnd(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupAnd($0)";
    case metal:
        __intrinsic_asm "simd_and";
    case spirv:
        if (__isBool<T>()) {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformLogicalAnd $$T result Subgroup 0 $value
            };
        }
        else {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformBitwiseAnd $$T result Subgroup 0 $value
            };
        }
    }
}

__generic<T : __BuiltinLogicalType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_arithmetic)]
public T subgroupOr(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupOr($0)";
    case metal:
        __intrinsic_asm "simd_or";
    case spirv:
        if (__isBool<T>()) {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformLogicalOr $$T result Subgroup 0 $value
            };
        }
        else {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformBitwiseOr $$T result Subgroup 0 $value
            };
        }
    }
}

__generic<T : __BuiltinLogicalType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_arithmetic)]
public T subgroupXor(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupXor($0)";
    case metal:
        __intrinsic_asm "simd_xor";
    case spirv:
        if (__isBool<T>()) {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformLogicalXor $$T result Subgroup 0 $value
            };
        }
        else {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformBitwiseXor $$T result Subgroup 0 $value
            };
        }
    }
}

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_arithmetic)]
public T subgroupInclusiveAdd(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupInclusiveAdd($0)";
    case metal:
        __intrinsic_asm "simd_prefix_inclusive_sum";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFAdd $$T result Subgroup InclusiveScan $value};
        else if (__isInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformIAdd $$T result Subgroup InclusiveScan $value};
        else return value;
    }
}

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_arithmetic)]
public T subgroupInclusiveMul(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupInclusiveMul($0)";
    case metal:
        __intrinsic_asm "simd_prefix_inclusive_product";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMul $$T result Subgroup InclusiveScan $value};
        else if (__isInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformIMul $$T result Subgroup InclusiveScan $value};
        else return value;
    }
}

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public T subgroupInclusiveMin(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupInclusiveMin($0)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMin $$T result Subgroup InclusiveScan $value};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformSMin $$T result Subgroup InclusiveScan $value};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformUMin $$T result Subgroup InclusiveScan $value};
        else return value;
    }
}

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public T subgroupInclusiveMax(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupInclusiveMax($0)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMax $$T result Subgroup InclusiveScan $value};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformSMax $$T result Subgroup InclusiveScan $value};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformUMax $$T result Subgroup InclusiveScan $value};
        else return value;
    }
}

__generic<T : __BuiltinLogicalType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public T subgroupInclusiveAnd(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupInclusiveAnd($0)";
    case spirv:
        if (__isBool<T>()) {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformLogicalAnd $$T result Subgroup InclusiveScan $value
            };
        }
        else {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformBitwiseAnd $$T result Subgroup InclusiveScan $value
            };
        }
    }
}

__generic<T : __BuiltinLogicalType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public T subgroupInclusiveOr(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupInclusiveOr($0)";
    case spirv:
        if (__isBool<T>()) {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformLogicalOr $$T result Subgroup InclusiveScan $value
            };
        }
        else {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformBitwiseOr $$T result Subgroup InclusiveScan $value
            };
        }
    }
}

__generic<T : __BuiltinLogicalType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public T subgroupInclusiveXor(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupInclusiveXor($0)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformLogicalXor $$T result Subgroup InclusiveScan $value};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseXor $$T result Subgroup InclusiveScan $value};
    }
    return T(0);
}

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
public T subgroupExclusiveAdd(T value)
{
    shader_subgroup_preamble<T>();
    return WavePrefixSum(value);
}


__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
public T subgroupExclusiveMul(T value)
{
    shader_subgroup_preamble<T>();
    return WavePrefixProduct(value);
}

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public T subgroupExclusiveMin(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupExclusiveMin($0)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMin $$T result Subgroup ExclusiveScan $value};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformSMin $$T result Subgroup ExclusiveScan $value};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformUMin $$T result Subgroup ExclusiveScan $value};
        else return value;
    }
}

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public T subgroupExclusiveMax(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupExclusiveMax($0)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMax $$T result Subgroup ExclusiveScan $value};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformSMax $$T result Subgroup ExclusiveScan $value};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformUMax $$T result Subgroup ExclusiveScan $value};
        else return value;
    }
}

__generic<T : __BuiltinLogicalType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public T subgroupExclusiveAnd(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupExclusiveAnd($0)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformLogicalAnd $$T result Subgroup ExclusiveScan $value};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseAnd $$T result Subgroup ExclusiveScan $value};
    }
}

__generic<T : __BuiltinLogicalType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public T subgroupExclusiveOr(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupExclusiveOr($0)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformLogicalOr $$T result Subgroup ExclusiveScan $value};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseOr $$T result Subgroup ExclusiveScan $value};
    }
}

__generic<T : __BuiltinLogicalType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public T subgroupExclusiveXor(T value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupExclusiveXor($0)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformLogicalXor $$T result Subgroup ExclusiveScan $value};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseXor $$T result Subgroup ExclusiveScan $value};
    }
}

// GL_KHR_shader_subgroup_arithmetic
//note: this is a seperate section because it is so huge that the only reasonable way to implement this is to just regex replace code

__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
public vector<T,N> subgroupAdd(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    return WaveActiveSum(value);
}

__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
public vector<T,N> subgroupMul(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    return WaveActiveProduct(value);
}

__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
public vector<T,N> subgroupMin(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    return WaveActiveMin(value);
}

__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
public vector<T,N> subgroupMax(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    return WaveActiveMax(value);
}

__generic<T : __BuiltinLogicalType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_arithmetic)]
public vector<T,N> subgroupAnd(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        // TODO: Bool inputs are invalid for Metal and WGSL, cast them to int or don't allow them to compile.
        __intrinsic_asm "subgroupAnd($0)";
    case metal:
        __intrinsic_asm "simd_and";
    case spirv:
        if (__isBool<T>()) {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformLogicalAnd $$vector<T,N> result Subgroup 0 $value
            };
        }
        else {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformBitwiseAnd $$vector<T,N> result Subgroup 0 $value
            };
        }

    }
}

__generic<T : __BuiltinLogicalType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_arithmetic)]
public vector<T,N> subgroupOr(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupOr($0)";
    case metal:
        __intrinsic_asm "simd_or";
    case spirv:
        if (__isBool<T>()) {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformLogicalOr $$vector<T,N> result Subgroup 0 $value
            };
        }
        else {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformBitwiseOr $$vector<T,N> result Subgroup 0 $value
            };
        }

    }
}

__generic<T : __BuiltinLogicalType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_arithmetic)]
public vector<T,N> subgroupXor(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupXor($0)";
    case metal:
        __intrinsic_asm "simd_xor";
    case spirv:
        if (__isBool<T>()) {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformLogicalXor $$vector<T,N> result Subgroup 0 $value
            };
        }
        else {
            return spirv_asm {
                OpCapability GroupNonUniformArithmetic;
                OpGroupNonUniformBitwiseXor $$vector<T,N> result Subgroup 0 $value
            };
        }
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_arithmetic)]
public vector<T,N> subgroupInclusiveAdd(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupInclusiveAdd($0)";
    case metal:
        __intrinsic_asm "simd_prefix_inclusive_sum";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFAdd $$vector<T,N> result Subgroup InclusiveScan $value};
        else if (__isInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformIAdd $$vector<T,N> result Subgroup InclusiveScan $value};
        else return value;
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_arithmetic)]
public vector<T,N> subgroupInclusiveMul(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupInclusiveMul($0)";
    case metal:
        __intrinsic_asm "simd_prefix_inclusive_product";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMul $$vector<T,N> result Subgroup InclusiveScan $value};
        else if (__isInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformIMul $$vector<T,N> result Subgroup InclusiveScan $value};
        else return value;
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public vector<T,N> subgroupInclusiveMin(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupInclusiveMin($0)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMin $$vector<T,N> result Subgroup InclusiveScan $value};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformSMin $$vector<T,N> result Subgroup InclusiveScan $value};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformUMin $$vector<T,N> result Subgroup InclusiveScan $value};
        else return value;
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public vector<T,N> subgroupInclusiveMax(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupInclusiveMax($0)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMax $$vector<T,N> result Subgroup InclusiveScan $value};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformSMax $$vector<T,N> result Subgroup InclusiveScan $value};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformUMax $$vector<T,N> result Subgroup InclusiveScan $value};
        else return value;
    }
}

__generic<T : __BuiltinLogicalType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public vector<T,N> subgroupInclusiveAnd(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupInclusiveAnd($0)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformLogicalAnd $$vector<T,N> result Subgroup InclusiveScan $value};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseAnd $$vector<T,N> result Subgroup InclusiveScan $value};
    }
}

__generic<T : __BuiltinLogicalType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public vector<T,N> subgroupInclusiveOr(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupInclusiveOr($0)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformLogicalOr $$vector<T,N> result Subgroup InclusiveScan $value};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseOr $$vector<T,N> result Subgroup InclusiveScan $value};
    }
}

__generic<T : __BuiltinLogicalType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public vector<T,N> subgroupInclusiveXor(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupInclusiveXor($0)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformLogicalXor $$vector<T,N> result Subgroup InclusiveScan $value};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseXor $$vector<T,N> result Subgroup InclusiveScan $value};
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
public vector<T,N> subgroupExclusiveAdd(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    return WavePrefixSum(value);
}


__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_arithmetic)]
public vector<T,N> subgroupExclusiveMul(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    return WavePrefixProduct(value);
}

__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public vector<T,N> subgroupExclusiveMin(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupExclusiveMin($0)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMin $$vector<T,N> result Subgroup ExclusiveScan $value};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformSMin $$vector<T,N> result Subgroup ExclusiveScan $value};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformUMin $$vector<T,N> result Subgroup ExclusiveScan $value};
        else return value;
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public vector<T,N> subgroupExclusiveMax(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupExclusiveMax($0)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformFMax $$vector<T,N> result Subgroup ExclusiveScan $value};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformSMax $$vector<T,N> result Subgroup ExclusiveScan $value};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformUMax $$vector<T,N> result Subgroup ExclusiveScan $value};
        else return value;
    }
}

__generic<T : __BuiltinLogicalType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public vector<T,N> subgroupExclusiveAnd(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupExclusiveAnd($0)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformLogicalAnd $$vector<T,N> result Subgroup ExclusiveScan $value};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseAnd $$vector<T,N> result Subgroup ExclusiveScan $value};
    }
}

__generic<T : __BuiltinLogicalType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public vector<T,N> subgroupExclusiveOr(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupExclusiveOr($0)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformLogicalOr $$vector<T,N> result Subgroup ExclusiveScan $value};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseOr $$vector<T,N> result Subgroup ExclusiveScan $value};
    }
}

__generic<T : __BuiltinLogicalType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_arithmetic)
[ForceInline]
[require(glsl_spirv, subgroup_arithmetic)]
public vector<T,N> subgroupExclusiveXor(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl: __intrinsic_asm "subgroupExclusiveXor($0)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformLogicalXor $$vector<T,N> result Subgroup ExclusiveScan $value};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpGroupNonUniformBitwiseXor $$vector<T,N> result Subgroup ExclusiveScan $value};
    }
}

// GL_KHR_shader_subgroup_ballot

__generic<T : __BuiltinType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_ballot)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_ballot)]
public T subgroupBroadcast(T value, uint id)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case wgsl:
        // WGSL's intrinsic does not accept non-const ids, do shuffle instead.
        __intrinsic_asm "subgroupShuffle";
    default:
        return WaveBroadcastLaneAt(value, id);
    }
}

__generic<T : __BuiltinType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_ballot)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_ballot)]
public vector<T,N> subgroupBroadcast(vector<T,N> value, uint id)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case wgsl:
        // WGSL's intrinsic does not accept non-const ids, do shuffle instead.
        __intrinsic_asm "subgroupShuffle";
    default:
        return WaveBroadcastLaneAt(value, id);
    }
}

__generic<T : __BuiltinType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_ballot)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_ballot)]
public T subgroupBroadcastFirst(T value)
{
    shader_subgroup_preamble<T>();
    return WaveReadLaneFirst(value);
}

__generic<T : __BuiltinType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_ballot)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_ballot)]
public vector<T,N> subgroupBroadcastFirst(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    return WaveReadLaneFirst(value);
}

// WaveMaskBallot is not the same; it force trunc's
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_ballot)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_ballot)]
public uvec4 subgroupBallot(bool value)
{
    return WaveActiveBallot(value);
}

// logic for HLSL and CUDA which lack InverseBalloc
// CUDA: works exclusivly 32 waves, therefore only need comp x
// HLSL:{
// 1. index into comp I want: index = trunc(float(lane)*(1/32))
// 2. lane & value[index]
// note: 1/32 wil be converted to multiplication
// we do 1/32 since 1 uint stores 32 threads 
// note 2: we have a waveLaneCount check because based on wave lane count we can determine if we can do a 
// fast path or slow path (know index is 0 or non 0)
// }
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_ballot)
[ForceInline]
[require(cuda_glsl_hlsl_spirv, subgroup_ballot)]
public bool subgroupInverseBallot(uvec4 value)
{
    __target_switch
    {
    case cuda:
        // only has 32 warps
        __intrinsic_asm "(($0).x >> (_getLaneId()) & 1)";
    case hlsl:
        // much like _WaveCountBits, but here we hope that we hit case 0; we can then avoid the expensive logic
        const uint waveLaneCount = WaveGetLaneCount();
        switch ((waveLaneCount - 1) / 32)
        {
        case 0:
            __intrinsic_asm "(($0)[0] >> WaveGetLaneIndex()) & 1)";
        case 1:
        case 2:
        case 3:
            __intrinsic_asm "((($0)[uint(float(WaveGetLaneIndex())*0.03125f)] >> WaveGetLaneIndex()) & 1)";
        }
    case glsl:
        __intrinsic_asm "subgroupInverseBallot($0)";
    case spirv:
        return spirv_asm {
                OpCapability GroupNonUniformBallot; 
                OpGroupNonUniformInverseBallot $$bool result Subgroup $value
        };
    }
    return false;
}

// same logic as subgroupInverseBallot
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_ballot)
[ForceInline]
[require(cuda_glsl_hlsl_spirv, subgroup_ballot)]
public bool subgroupBallotBitExtract(uvec4 value, uint index)
{
    __target_switch
    {
    case cuda:
        __intrinsic_asm "($1 & ($0).x) != 0";
    case hlsl:
        const uint waveLaneCount = WaveGetLaneCount();
        switch ((waveLaneCount - 1) / 32)
        {
        case 0:
            __intrinsic_asm "($0)[0] & ($1)";
        case 1:
        case 2:
        case 3:
            __intrinsic_asm "($0)[uint(float($1)*0.03125f)] & ($1)";
        }
    case glsl:
        __intrinsic_asm "subgroupBallotBitExtract($0, $1)";
    case spirv:
        return spirv_asm {
                OpCapability GroupNonUniformBallot; 
                OpGroupNonUniformBallotBitExtract $$bool result Subgroup $value $index
        };
    }
    return false;
}


// the count is only supposed to use uvec4 values within bottom bits of subgroup launched, not a simple countbits
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_ballot)
[ForceInline]
[require(glsl_spirv, subgroup_ballot)]
public uint subgroupBallotBitCount(uvec4 value)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupBallotBitCount($0)";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformBallot; 
            OpGroupNonUniformBallotBitCount $$uint result Subgroup Reduce $value
        };
    }
}

__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_ballot)
[ForceInline]
[require(glsl_spirv, subgroup_ballot)]
public uint subgroupBallotInclusiveBitCount(uvec4 value)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupBallotInclusiveBitCount($0)";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformBallot; 
            OpGroupNonUniformBallotBitCount $$uint result Subgroup InclusiveScan $value
        };
    }
}

__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_ballot)
[ForceInline]
[require(glsl_spirv, subgroup_ballot)]
public uint subgroupBallotExclusiveBitCount(uvec4 value)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupBallotExclusiveBitCount($0)";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformBallot; 
            OpGroupNonUniformBallotBitCount $$uint result Subgroup ExclusiveScan $value
        };
    }
}

__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_ballot)
[ForceInline]
[require(glsl_spirv, subgroup_ballot)]
public uint subgroupBallotFindLSB(uvec4 value)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupBallotFindLSB($0)";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformBallot; 
            OpGroupNonUniformBallotFindLSB $$uint result Subgroup $value
        };
    }
}

__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_ballot)
[ForceInline]
[require(glsl_spirv, subgroup_ballot)]
public uint subgroupBallotFindMSB(uvec4 value)
{
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupBallotFindMSB($0)";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformBallot; 
            OpGroupNonUniformBallotFindMSB $$uint result Subgroup $value
        };
    }
}

// GL_KHR_shader_subgroup_shuffle

__generic<T : __BuiltinType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_shuffle)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_shuffle)]
public T subgroupShuffle(T value, uint index)
{
    shader_subgroup_preamble<T>();
    return WaveShuffle(value, index);
}

__generic<T : __BuiltinType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_shuffle)
__wgsl_extension(subgroups)
[require(glsl_metal_spirv_wgsl, subgroup_shuffle)]
[ForceInline] public T subgroupShuffleXor(T value, uint mask)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupShuffleXor($0,$1)";
    case metal:
        __intrinsic_asm "simd_shuffle_xor($0, ushort($1))";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformBallot; 
            OpGroupNonUniformShuffleXor $$T result Subgroup $value $mask
        };
    }
}

__generic<T : __BuiltinType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_shuffle)
[ForceInline]
[require(cuda_glsl_hlsl_metal_spirv_wgsl, subgroup_shuffle)]
public vector<T,N> subgroupShuffle(vector<T,N> value, uint index)
{
    shader_subgroup_preamble<T>();
    return WaveShuffle(value, index);
}

__generic<T : __BuiltinType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_shuffle)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_shuffle)]
public vector<T,N> subgroupShuffleXor(vector<T,N> value, uint mask)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupShuffleXor($0,$1)";
    case metal:
        __intrinsic_asm "simd_shuffle_xor($0, ushort($1))";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformBallot; 
            OpGroupNonUniformShuffleXor $$vector<T,N> result Subgroup $value $mask
        };
    }
}


// GL_KHR_shader_subgroup_shuffle_relative

__generic<T : __BuiltinType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_shuffle_relative)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_shufflerelative)]
public T subgroupShuffleUp(T value, uint delta)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupShuffleUp($0, $1)";
    case metal:
        __intrinsic_asm "simd_shuffle_up($0, ushort($1))";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformShuffleRelative;
            OpGroupNonUniformShuffleUp $$T result Subgroup $value $delta
        };
    }
}

__generic<T : __BuiltinType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_shuffle_relative)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_shufflerelative)]
public T subgroupShuffleDown(T value, uint delta)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupShuffleDown($0, $1)";
    case metal:
        __intrinsic_asm "simd_shuffle_down($0, ushort($1))";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformShuffleRelative; 
            OpGroupNonUniformShuffleDown $$T result Subgroup $value $delta
        };
    }
}


__generic<T : __BuiltinType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_shuffle_relative)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_shufflerelative)]
public vector<T,N> subgroupShuffleUp(vector<T,N> value, uint delta)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupShuffleUp($0, $1)";
    case metal:
        __intrinsic_asm "simd_shuffle_up($0, ushort($1))";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformShuffleRelative;
            OpGroupNonUniformShuffleUp $$vector<T,N> result Subgroup $value $delta
        };
    }
}

__generic<T : __BuiltinType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_shuffle_relative)
__wgsl_extension(subgroups)
[ForceInline]
[require(glsl_metal_spirv_wgsl, subgroup_shufflerelative)]
public vector<T,N> subgroupShuffleDown(vector<T,N> value, uint delta)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
    case wgsl:
        __intrinsic_asm "subgroupShuffleDown($0, $1)";
    case metal:
        __intrinsic_asm "simd_shuffle_down($0, ushort($1))";
    case spirv:
        return spirv_asm {
            OpCapability GroupNonUniformShuffleRelative;
            OpGroupNonUniformShuffleDown $$vector<T,N> result Subgroup $value $delta
        };
    }
}
// GL_KHR_shader_subgroup_clustered

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_clustered)
[ForceInline]
[require(glsl_spirv, subgroup_clustered)]
public T subgroupClusteredAdd(T value, uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredAdd($0, $1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformFAdd $$T result Subgroup ClusteredReduce $value $clusterSize};
        else if (__isInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformIAdd $$T result Subgroup ClusteredReduce $value $clusterSize};
        else return value;
    }
}

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_clustered)
[ForceInline]
[require(glsl_spirv, subgroup_clustered)]
public T subgroupClusteredMul(T value, uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredMul($0, $1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformFMul $$T result Subgroup ClusteredReduce $value $clusterSize};
        else if (__isInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformIMul $$T result Subgroup ClusteredReduce $value $clusterSize};
        else return value;
    } 
}

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_clustered)
[ForceInline]
[require(glsl_spirv, subgroup_clustered)]
public T subgroupClusteredMin(T value, uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredMin($0, $1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformFMin $$T result Subgroup ClusteredReduce $value $clusterSize};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformSMin $$T result Subgroup ClusteredReduce $value $clusterSize};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformUMin $$T result Subgroup ClusteredReduce $value $clusterSize};
        else return value;
    }
}

__generic<T : __BuiltinArithmeticType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_clustered)
[ForceInline]
[require(glsl_spirv, subgroup_clustered)]
public T subgroupClusteredMax(T value, uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredMax($0, $1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformFMax $$T result Subgroup ClusteredReduce $value $clusterSize};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformSMax $$T result Subgroup ClusteredReduce $value $clusterSize};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered;  OpGroupNonUniformUMax $$T result Subgroup ClusteredReduce $value $clusterSize};
        else return value;
    }
}

__generic<T : __BuiltinLogicalType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_clustered)
[ForceInline]
[require(glsl_spirv, subgroup_clustered)]
public T subgroupClusteredAnd(T value, uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredAnd($0, $1)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformLogicalAnd $$T result Subgroup ClusteredReduce $value $clusterSize};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformBitwiseAnd $$T result Subgroup ClusteredReduce $value $clusterSize};
    }
}

__generic<T : __BuiltinLogicalType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_clustered)
[ForceInline]
[require(glsl_spirv, subgroup_clustered)]
public T subgroupClusteredOr(T value, uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredOr($0, $1)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformLogicalOr $$T result Subgroup ClusteredReduce $value $clusterSize};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformBitwiseOr $$T result Subgroup ClusteredReduce $value $clusterSize};
    }
}



__generic<T : __BuiltinLogicalType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_clustered)
[ForceInline]
[require(glsl_spirv, subgroup_clustered)]
public T subgroupClusteredXor(T value, uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredXor($0, $1)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformLogicalXor $$T result Subgroup ClusteredReduce $value $clusterSize};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformBitwiseXor $$T result Subgroup ClusteredReduce $value $clusterSize};
    }
}



__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_clustered)
[ForceInline]
[require(glsl_spirv, subgroup_clustered)]
public vector<T,N> subgroupClusteredAdd(vector<T,N> value, uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredAdd($0, $1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; 
            OpGroupNonUniformFAdd $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
        else if (__isInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformIAdd $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
        else return value;
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_clustered)
[ForceInline]
[require(glsl_spirv, subgroup_clustered)]
public vector<T,N> subgroupClusteredMul(vector<T,N> value, uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredMul($0, $1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformFMul $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
        else if (__isInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformIMul $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
        else return value;
    } 
}

__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_clustered)
[ForceInline]
[require(glsl_spirv, subgroup_clustered)]
public vector<T,N> subgroupClusteredMin(vector<T,N> value, uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredMin($0, $1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformFMin $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformSMin $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformUMin $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
        else return value;
    }
}

__generic<T : __BuiltinArithmeticType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_clustered)
[ForceInline]
[require(glsl_spirv, subgroup_clustered)]
public vector<T,N> subgroupClusteredMax(vector<T,N> value, uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredMax($0, $1)";
    case spirv:
        if (__isFloat<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformFMax $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
        else if (__isSignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformSMax $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
        else if (__isUnsignedInt<T>())
            return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered;  OpGroupNonUniformUMax $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
        else return value;
    }
}

__generic<T : __BuiltinLogicalType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_clustered)
[ForceInline]
[require(glsl_spirv, subgroup_clustered)]
public vector<T,N> subgroupClusteredAnd(vector<T,N> value, uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredAnd($0, $1)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformLogicalAnd $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformBitwiseAnd $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
    }
}

__generic<T : __BuiltinLogicalType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_clustered)
[ForceInline]
[require(glsl_spirv, subgroup_clustered)]
public vector<T,N> subgroupClusteredOr(vector<T,N> value, uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredOr($0, $1)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformLogicalOr $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformBitwiseOr $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
    }
}

__generic<T : __BuiltinLogicalType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_clustered)
[ForceInline]
[require(glsl_spirv, subgroup_clustered)]
public vector<T,N> subgroupClusteredXor(vector<T,N> value, uint clusterSize)
{
    shader_subgroup_preamble<T>();
    __target_switch
    {
    case glsl:
        __intrinsic_asm "subgroupClusteredXor($0, $1)";
    case spirv:
        if (__isBool<T>()) return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformLogicalXor $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
        else return spirv_asm {OpCapability GroupNonUniformArithmetic; OpCapability GroupNonUniformClustered; OpGroupNonUniformBitwiseXor $$vector<T,N> result Subgroup ClusteredReduce $value $clusterSize};
    }
}

// GL_KHR_shader_subgroup_quad

__generic<T : __BuiltinType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_quad)
[ForceInline]
[require(glsl_hlsl_spirv, subgroup_quad)]
public T subgroupQuadBroadcast(T value, uint id)
{
    shader_subgroup_preamble<T>();
    return QuadReadLaneAt(value, id);
}

__generic<T : __BuiltinType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_quad)
[ForceInline]
[require(glsl_hlsl_metal_spirv_wgsl, subgroup_quad)]
public T subgroupQuadSwapHorizontal(T value)
{
    shader_subgroup_preamble<T>();
    return QuadReadAcrossX(value);
}

__generic<T : __BuiltinType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_quad)
[ForceInline]
[require(glsl_hlsl_metal_spirv_wgsl, subgroup_quad)]
public T subgroupQuadSwapVertical(T value)
{
    shader_subgroup_preamble<T>();
    return QuadReadAcrossY(value);
}

__generic<T : __BuiltinType>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_quad)
[ForceInline]
[require(glsl_hlsl_metal_spirv_wgsl, subgroup_quad)]
public T subgroupQuadSwapDiagonal(T value)
{
    shader_subgroup_preamble<T>();
    return QuadReadAcrossDiagonal(value);
}


__generic<T : __BuiltinType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_quad)
[ForceInline]
[require(glsl_hlsl_spirv, subgroup_quad)]
public vector<T,N> subgroupQuadBroadcast(vector<T,N> value, uint id)
{
    shader_subgroup_preamble<T>();
    return QuadReadLaneAt(value, id);
}

__generic<T : __BuiltinType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_quad)
[ForceInline]
[require(glsl_hlsl_metal_spirv_wgsl, subgroup_quad)]
public vector<T,N> subgroupQuadSwapHorizontal(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    return QuadReadAcrossX(value);
}

__generic<T : __BuiltinType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_quad)
[ForceInline]
[require(glsl_hlsl_metal_spirv_wgsl, subgroup_quad)]
public vector<T,N> subgroupQuadSwapVertical(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    return QuadReadAcrossY(value);
}

__generic<T : __BuiltinType, let N : int>
__spirv_version(1.3)
__glsl_extension(GL_KHR_shader_subgroup_quad)
[ForceInline]
[require(glsl_hlsl_metal_spirv_wgsl, subgroup_quad)]
public vector<T,N> subgroupQuadSwapDiagonal(vector<T,N> value)
{
    shader_subgroup_preamble<T>();
    return QuadReadAcrossDiagonal(value);
}

// GL_KHR_shader_subgroup_rotate

__generic<T : __BuiltinType>
[require(glsl_metal_spirv, subgroup_rotate)]
public T subgroupRotate(T value, uint delta)
{
    return WaveRotate(value, delta);
}

__generic<T : __BuiltinType, let N : int>
[require(glsl_metal_spirv, subgroup_rotate)]
public vector<T, N> subgroupRotate(vector<T, N> value, uint delta)
{
    return WaveRotate(value, delta);
}

__generic<T : __BuiltinType>
[require(glsl_spirv, subgroup_rotate)]
public T subgroupClusteredRotate(T value, uint delta, constexpr uint clusterSize)
{
    return WaveClusteredRotate(value, delta, clusterSize);

}

__generic<T : __BuiltinType, let N : int>
[require(glsl_spirv, subgroup_rotate)]
public vector<T, N> subgroupClusteredRotate(vector<T, N> value, uint delta, constexpr uint clusterSize)
{
    return WaveClusteredRotate(value, delta, clusterSize);
}

//// GLSL atomic

// The following type internally is a Shader Storage Buffer 
// as per GL_EXT_vulkan_glsl_relaxed
__magic_type(GLSLAtomicUintType)
__intrinsic_type($(kIROp_GLSLAtomicUintType))
public struct atomic_uint
{
};

// tier of float refers to atomic extension support of float1 or float2.
// if we are inside a atomic_float function we will run the check for float1 tier
// types and operations to enable the according ext needed for these operations

__generic<T : __BuiltinType>
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
void typeRequireChecks_atomic_using_float0_tier() 
{
    __target_switch
    {
    case glsl:
    {
        if (__type_equals<T, uint64_t>() || __type_equals<T, int64_t>())
            __requireTargetExtension("GL_EXT_shader_atomic_int64");
    }
    case spirv:
        return; 
    }
}
__generic<T : __BuiltinType>
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
void typeRequireChecks_atomic_using_float1_tier() 
{
    __target_switch
    {
    case glsl:
    {
        if (__type_equals<T, float>()) 
            __requireTargetExtension("GL_EXT_shader_atomic_float");
        else if (__type_equals<T, half>() || __type_equals<T, float16_t>())
        {
            __requireTargetExtension("GL_EXT_shader_atomic_float2");
            __requireTargetExtension("GL_EXT_shader_explicit_arithmetic_types");
        }
        else if (__type_equals<T, double>()) 
            __requireTargetExtension("GL_EXT_shader_atomic_float");
        else if (__type_equals<T, uint64_t>() || __type_equals<T, int64_t>())
            __requireTargetExtension("GL_EXT_shader_atomic_int64");
    }
    case spirv:
        return;
    }
}
__generic<T : __BuiltinType>
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
void typeRequireChecks_atomic_using_float2_tier() 
{
    __target_switch
    {
    case glsl:
    {    
        if (__type_equals<T, float>()) 
            __requireTargetExtension("GL_EXT_shader_atomic_float2");
        else if (__type_equals<T, half>() || __type_equals<T, float16_t>())
        {
            __requireTargetExtension("GL_EXT_shader_atomic_float2");
            __requireTargetExtension("GL_EXT_shader_explicit_arithmetic_types");
        }
        else if (__type_equals<T, double>()) 
            __requireTargetExtension("GL_EXT_shader_atomic_float2");
        else if (__type_equals<T, uint64_t>() || __type_equals<T, int64_t>())
             __requireTargetExtension("GL_EXT_shader_atomic_int64");
    }
    case spirv: 
        return; 
    }
}

__generic<T : __BuiltinType>
[require(glsl_spirv, atomic_glsl)]
void typeRequireChecks_atomic_using_add()
{
    __target_switch
    {
    case glsl:
        return;
    case spirv:
    {
        if (__type_equals<T, float>())
        {
            spirv_asm
            {
                OpExtension "SPV_EXT_shader_atomic_float_add"; 
                OpCapability AtomicFloat32AddEXT
            };
        }
        else if (__type_equals<T, half>() 
            || __type_equals<T, float16_t>())
        {
            spirv_asm
            {
                OpExtension "SPV_EXT_shader_atomic_float_add"; 
                OpCapability AtomicFloat16AddEXT
            };
        }
        else if (__type_equals<T, double>())
        {
            spirv_asm
            {
                OpExtension "SPV_EXT_shader_atomic_float_add"; 
                OpCapability AtomicFloat64AddEXT
            };
        }
        else if (__type_equals<T, uint64_t>()
            || __type_equals<T, int64_t>())
        {
            spirv_asm
            {
                OpCapability Int64Atomics
            };
        }
    }    
    }
}
__generic<T : __BuiltinType>
[require(glsl_spirv, atomic_glsl)]
void typeRequireChecks_atomic_using_MinMax()
{
    __target_switch
    {
    case glsl:
        return;
    case spirv:
    {
        if (__type_equals<T, float>())
        {
            spirv_asm
            {
                OpExtension "SPV_EXT_shader_atomic_float_add"; 
                OpCapability AtomicFloat32MinMaxEXT
            };
        }
        else if (__type_equals<T, half>() 
            || __type_equals<T, float16_t>())
        {
            spirv_asm
            {
                OpExtension "SPV_EXT_shader_atomic_float_add"; 
                OpCapability AtomicFloat16MinMaxEXT
            };
        }
        else if (__type_equals<T, double>())
        {
            spirv_asm
            {
                OpExtension "SPV_EXT_shader_atomic_float_add"; 
                OpCapability AtomicFloat64MinMaxEXT
            };
        }
        else if (__type_equals<T, uint64_t>()
            || __type_equals<T, int64_t>())
        {
            spirv_asm
            { 
                OpCapability Int64Atomics
            };
        }
    }
    }
}
__generic<T : __BuiltinType>
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
void typeRequireChecks_atomic_using_Logical_CAS() 
{
    __target_switch
    {
    case glsl:
        return;
    case spirv:
    {
        if (__type_equals<T, uint64_t>()
            || __type_equals<T, int64_t>())
        {
            spirv_asm
            { 
                OpCapability Int64Atomics
            };
        }
    }
    }
}

${{{{
static const struct {
    const char* name;
    const char* classType;
    const char *subclassType;
    const char *suffix;
    const bool isFloat;
} atomics[] =
    {
        { 
            "uint", "I", "U", "", false
        },
        { 
            "uint64_t", "I", "U", "", false
        },
        { 
            "int", "I", "S", "", false
        },
        { 
            "int64_t", "I", "S", "", false
        },
        { 
            "float16_t", "F", "F", "EXT", true
        },
        { 
            "float", "F", "F", "EXT", true
        },
        { 
            "double", "F", "F", "EXT", true
        },
    };
for (const auto& item : atomics) 
{
}}}}


__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
__intrinsic_op($(kIROp_AtomicAdd))
$(item.name) atomicAddWithOrder(inout $(item.name) mem, $(item.name) data, MemoryOrder order);

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public $(item.name) atomicAdd(inout $(item.name) mem, $(item.name) data)
{
    typeRequireChecks_atomic_using_float1_tier<$(item.name)>();
    typeRequireChecks_atomic_using_add<$(item.name)>();
    return atomicAddWithOrder(mem, data, MemoryOrder::Relaxed);
}

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
__intrinsic_op($(kIROp_AtomicMin))
$(item.name) atomicMinWithOrder(inout $(item.name) mem, $(item.name) data, MemoryOrder order);

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public $(item.name) atomicMin(inout $(item.name) mem, $(item.name) data)
{
    typeRequireChecks_atomic_using_float2_tier<$(item.name)>();
    typeRequireChecks_atomic_using_MinMax<$(item.name)>();
    return atomicMinWithOrder(mem, data, MemoryOrder::Relaxed);
}

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
__intrinsic_op($(kIROp_AtomicMax))
$(item.name) atomicMaxWithOrder(inout $(item.name) mem, $(item.name) data, MemoryOrder order);

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public $(item.name) atomicMax(inout $(item.name) mem, $(item.name) data)
{
    typeRequireChecks_atomic_using_float2_tier<$(item.name)>();
    typeRequireChecks_atomic_using_MinMax<$(item.name)>();
    return atomicMaxWithOrder(mem, data, MemoryOrder::Relaxed);
}

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
__intrinsic_op($(kIROp_AtomicExchange))
$(item.name) atomicExchangeWithOrder(inout $(item.name) mem, $(item.name) data, MemoryOrder order);

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public $(item.name) atomicExchange(inout $(item.name) mem, $(item.name) data)
{
    typeRequireChecks_atomic_using_float1_tier<$(item.name)>();
    return atomicExchangeWithOrder(mem, data, MemoryOrder::Relaxed);
}

${{{{
if(item.isFloat) 
    continue;
}}}}


__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
__intrinsic_op($(kIROp_AtomicAnd))
$(item.name) atomicAndWithOrder(inout $(item.name) mem, $(item.name) data, MemoryOrder order);

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public $(item.name) atomicAnd(inout $(item.name) mem, $(item.name) data)
{
    typeRequireChecks_atomic_using_float0_tier<$(item.name)>();
    typeRequireChecks_atomic_using_Logical_CAS<$(item.name)>();
    return atomicAndWithOrder(mem, data, MemoryOrder::Relaxed);
}

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
__intrinsic_op($(kIROp_AtomicOr))
$(item.name) atomicOrWithOrder(inout $(item.name) mem, $(item.name) data, MemoryOrder order);

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public $(item.name) atomicOr(inout $(item.name) mem, $(item.name) data)
{
    typeRequireChecks_atomic_using_float0_tier<$(item.name)>();
    typeRequireChecks_atomic_using_Logical_CAS<$(item.name)>();
    return atomicOrWithOrder(mem, data, MemoryOrder::Relaxed);
}


__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
__intrinsic_op($(kIROp_AtomicXor))
$(item.name) atomicXorWithOrder(inout $(item.name) mem, $(item.name) data, MemoryOrder order);

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public $(item.name) atomicXor(inout $(item.name) mem, $(item.name) data)
{
    typeRequireChecks_atomic_using_float0_tier<$(item.name)>();
    typeRequireChecks_atomic_using_Logical_CAS<$(item.name)>();
    return atomicXorWithOrder(mem, data, MemoryOrder::Relaxed);
}

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
__intrinsic_op($(kIROp_AtomicCompareExchange))
$(item.name) atomicCompSwapWithOrder(inout $(item.name) mem, $(item.name) compare, $(item.name) data, MemoryOrder successOrder, MemoryOrder failOrder);

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public $(item.name) atomicCompSwap(inout $(item.name) mem, $(item.name) compare, $(item.name) data)
{
    typeRequireChecks_atomic_using_float0_tier<$(item.name)>();
    typeRequireChecks_atomic_using_Logical_CAS<$(item.name)>();
    return atomicCompSwapWithOrder(mem, compare, data, MemoryOrder::Relaxed, MemoryOrder::Relaxed);
}

${{{{
}
}}}}

// all atomic_uint functions are mangled at compile time, 
// all types are converted into a field address of a 'uint'
// relative to the layout(offset) of the atomic_uint 

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public uint atomicCounterIncrement(atomic_uint c)
{

    __target_switch
    {
    case glsl: __intrinsic_asm "atomicAdd($0, 1)";
    case spirv:
    {
        return spirv_asm
        {
            OpAtomicIIncrement $$uint result $c Device UniformMemory
        };
    }
    }
}

__glsl_version(430) 
[require(glsl, atomic_glsl)]
[ForceInline] public uint atomicCounterDecrement_GLSL_helper(atomic_uint c)
{
    __target_switch
    {
    case glsl:
    {
        __intrinsic_asm "atomicExchange($0,$0-1)";
    }
    }
}

__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public uint atomicCounter(atomic_uint c)
{
    __target_switch
    {
    case glsl:
    {
        __intrinsic_asm "($0)";
    }
    case spirv:
    {
        return spirv_asm
        {
            OpLoad $$uint result $c 
        };
    }
    }
}


__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public uint atomicCounterDecrement(atomic_uint c)
{
    __target_switch
    {
    case glsl:
    {
        atomicCounterDecrement_GLSL_helper(c);
        return atomicCounter(c);
    }
    case spirv:
    {
        // spirv OpAtomicIDecrement returns pre-sub-1, glsl returns the new value
        // we want a discarded side effect and then return the new value
        return spirv_asm
        {
            %discardedValue:$$uint = OpAtomicIDecrement $c Device UniformMemory;
            OpLoad $$uint result $c
        };
    }
    }   
}


__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public uint atomicCounterAdd(atomic_uint c, uint data)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "atomicAdd($0, $1)";
    case spirv:
    {
        return spirv_asm
        {
            OpAtomicIAdd $$uint result $c Device UniformMemory $data
        };
    }
    }
}


__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public uint atomicCounterSubtract(atomic_uint c, uint data)
{
    __target_switch
    {
    case glsl:
    {
        __intrinsic_asm "atomicExchange($0,$0-$1)";
    }
    case spirv:
    {
        return spirv_asm
        {
            OpAtomicISub $$uint result $c Device UniformMemory $data
        };
    }
    }   
}


__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public uint atomicCounterMin(atomic_uint c, uint data)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "atomicMin($0, $1)";
    case spirv:
    {
        return spirv_asm
        {
            OpAtomicUMin $$uint result $c Device UniformMemory $data
        };
    }
    }
}


__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public uint atomicCounterMax(atomic_uint c, uint data)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "atomicMax($0, $1)";
    case spirv:
    {
        return spirv_asm
        {
            OpAtomicUMax $$uint result $c Device UniformMemory $data
        };
    }
    }
}


__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public uint atomicCounterAnd(atomic_uint c, uint data)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "atomicAnd($0, $1)";
    case spirv:
    {
        return spirv_asm
        {
            OpAtomicAnd $$uint result $c Device UniformMemory $data
        };
    }
    }   
}


__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public uint atomicCounterOr(atomic_uint c, uint data)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "atomicOr($0, $1)";
    case spirv:
    {
        return spirv_asm
        {
            OpAtomicOr $$uint result $c Device UniformMemory $data
        };
    }
    }
}


__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public uint atomicCounterXor(atomic_uint c, uint data)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "atomicXor($0, $1)";
    case spirv:
    {
        return spirv_asm
        {
            OpAtomicXor $$uint result $c Device UniformMemory $data
        };
    }
    }
}


__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public uint atomicCounterExchange(atomic_uint c, uint data)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "atomicExchange($0, $1)";
    case spirv:
    {
        return spirv_asm
        {
            OpAtomicExchange $$uint result $c Device UniformMemory $data
        };
    }
    }
}


__glsl_version(430)
[ForceInline]
[require(glsl_spirv, atomic_glsl)]
public uint atomicCounterCompSwap(atomic_uint c, uint compare, uint data)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "atomicCompSwap($0, $1, $2)";
    case spirv:
    {
        return spirv_asm
        {
            OpAtomicCompareExchange $$uint result $c Device UniformMemory UniformMemory $data $compare
        };
    }
    }
}

/// Section 8.14. Fragment Processing Functions

[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing)]
public float dFdx(float p)
{
    return ddx(p);
}
__generic<let N : int>
[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing)]
public vector<float, N> dFdx(vector<float, N> p)
{
    return ddx(p);
}

[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing)]
public float dFdy(float p)
{
    return ddy(p);
}
__generic<let N : int>
[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing)]
public vector<float, N> dFdy(vector<float, N> p)
{
    return ddy(p);
}

[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
public float dFdxFine(float p)
{
    return ddx_fine(p);
}
__generic<let N : int>
[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
public vector<float, N> dFdxFine(vector<float, N> p)
{
    return ddx_fine(p);
}

[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
public float dFdyFine(float p)
{
    return ddy_fine(p);
}
__generic<let N : int>
[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
public vector<float, N> dFdyFine(vector<float, N> p)
{
    return ddy_fine(p);
}

[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
public float dFdxCoarse(float p)
{
    return ddx_coarse(p);
}
__generic<let N : int>
[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
public vector<float, N> dFdxCoarse(vector<float, N> p)
{
    return ddx_coarse(p);
}

[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
public float dFdyCoarse(float p)
{
    return ddy_coarse(p);
}
__generic<let N : int>
[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
public vector<float, N> dFdyCoarse(vector<float, N> p)
{
    return ddy_coarse(p);
}

[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
public float fwidthFine(float p)
{
    return fwidth_fine(p);
}
__generic<let N : int>
[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
public vector<float, N> fwidthFine(vector<float, N> p)
{
    return fwidth_fine(p);
}

[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
public float fwidthCoarse(float p)
{
    return fwidth_coarse(p);
}
__generic<let N : int>
[__NoSideEffect]
[ForceInline]
[require(glsl_hlsl_spirv, fragmentprocessing_derivativecontrol)]
public vector<float, N> fwidthCoarse(vector<float, N> p)
{
    return fwidth_coarse(p);
}

[__NoSideEffect]
[__GLSLRequireShaderInputParameter(0)]
[require(glsl_spirv, fragmentprocessing)]
public float interpolateAtCentroid(__constref float interpolant)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "interpolateAtCentroid($0)";
    case spirv:
    {
        return spirv_asm
        {
            OpCapability InterpolationFunction;
            OpExtInst $$float result glsl450 InterpolateAtCentroid &interpolant
        };
    }
    }
}
__generic<let N : int>
[__NoSideEffect]
[__GLSLRequireShaderInputParameter(0)]
[require(glsl_spirv, fragmentprocessing)]
public vector<float, N> interpolateAtCentroid(__constref vector<float, N> interpolant)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "interpolateAtCentroid($0)";
    case spirv:
    {
        return spirv_asm
        {
            OpCapability InterpolationFunction;
            OpExtInst $$vector<float, N> result glsl450 InterpolateAtCentroid &interpolant
        };
    }
    }
}

[__NoSideEffect]
[__GLSLRequireShaderInputParameter(0)]
[require(glsl_spirv, fragmentprocessing)]
public float interpolateAtSample(__constref float interpolant, int sample)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "interpolateAtSample($0,$1)";
    case spirv:
    {
        return spirv_asm
        {
            OpCapability InterpolationFunction;
            OpExtInst $$float result glsl450 InterpolateAtSample &interpolant $sample
        };
    }
    }
}
__generic<let N : int>
[__NoSideEffect]
[__GLSLRequireShaderInputParameter(0)]
[require(glsl_spirv, fragmentprocessing)]
public vector<float, N> interpolateAtSample(__constref vector<float, N> interpolant, int sample)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "interpolateAtSample($0,$1)";
    case spirv:
    {
        return spirv_asm
        {
            OpCapability InterpolationFunction;
            OpExtInst $$vector<float, N> result glsl450 InterpolateAtSample &interpolant $sample
        };
    }
    }
}

[__NoSideEffect]
[__GLSLRequireShaderInputParameter(0)]
[require(glsl_spirv, fragmentprocessing)]
public float interpolateAtOffset(__constref float interpolant, vec2 offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "interpolateAtOffset($0,$1)";
    case spirv:
    {
        return spirv_asm
        {
            OpCapability InterpolationFunction;
            OpExtInst $$float result glsl450 InterpolateAtOffset &interpolant $offset
        };
    }
    }
}
__generic<let N : int>
[__NoSideEffect]
[__GLSLRequireShaderInputParameter(0)]
[require(glsl_spirv, fragmentprocessing)]
public vector<float, N> interpolateAtOffset(__constref vector<float, N> interpolant, vec2 offset)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "interpolateAtOffset($0,$1)";
    case spirv:
    {
        return spirv_asm
        {
            OpCapability InterpolationFunction;
            OpExtInst $$vector<float, N> result glsl450 InterpolateAtOffset &interpolant $offset
        };
    }
    }
}

/// Section 8.15. Noise Functions (deprecated)

[deprecated("Always returns 0")]
[__NoSideEffect]
[ForceInline]
public float noise1(float x)
{
    return 0.0f;
}
[deprecated("Always returns 0")]
__generic<let N : int>
[__NoSideEffect]
[ForceInline]
public float noise1(vector<float, N> x)
{
    return 0.0f;
}

[deprecated("Always returns 0")]
[__NoSideEffect]
[ForceInline]
public vec2 noise2(float x)
{
    return vec2(0.0f);
}
[deprecated("Always returns 0")]
__generic<let N : int>
[__NoSideEffect]
[ForceInline]
public vec2 noise2(vector<float, N> x)
{
    return vec2(0.0f);
}

[deprecated("Always returns 0")]
[__NoSideEffect]
[ForceInline]
public vec3 noise3(float x)
{
    return vec3(0.0f);
}
[deprecated("Always returns 0")]
__generic<let N : int>
[__NoSideEffect]
[ForceInline]
public vec3 noise3(vector<float, N> x)
{
    return vec3(0.0f);
}

[deprecated("Always returns 0")]
[__NoSideEffect]
[ForceInline]
public vec4 noise4(float x)
{
    return vec4(0.0f);
}
[deprecated("Always returns 0")]
__generic<let N : int>
[__NoSideEffect]
[ForceInline]
public vec4 noise4(vector<float, N> x)
{
    return vec4(0.0f);
}

/// Section 8.16. Shader Invocation Control Functions

// TODO: for tessellation control shader: barrier() function may only
// be placed inside the main() of the shader and may not be called within
// control flow.
// TODO: if called after a return, error.

[ForceInline]
[require(glsl_hlsl_spirv, memorybarrier)]
public void barrier()
{
    __target_switch
    {
    case hlsl:
    {
        GroupMemoryBarrier();
    }
    case glsl: __intrinsic_asm "barrier()";
    case spirv:
    {
        spirv_asm
        {
            OpControlBarrier Workgroup Workgroup AcquireRelease|WorkgroupMemory
        };
    }
    }
}

/// Section 8.17. Shader Memory Control Functions

[ForceInline]
[require(glsl_hlsl_spirv, shadermemorycontrol)]
public void memoryBarrier()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "AllMemoryBarrier()";
    case glsl: __intrinsic_asm "memoryBarrier()";
    case spirv:
    {
        spirv_asm
        {
            OpCapability Shader;
            OpMemoryBarrier Device AcquireRelease|WorkgroupMemory|ImageMemory|UniformMemory
        };
    }
    }
}

// Does not exist in vulkan. The equivlent is now `memoryBarrierBuffer`
// for vulkan since GL_EXT_vulkan_glsl_relaxed states that a atomic_counter
// is a storage buffer object in implementation.

// glslang will compile with `AtomicCounterMemory` since it does not use
// the `AtomicStorage` OpCapability which is required for `AtomicCounterMemory`.
// this is invalid/undefined spir-v for vulkan targets and should not be followed
[ForceInline]
[require(glsl_hlsl_spirv, shadermemorycontrol_compute)]
public void memoryBarrierAtomicCounter()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "AllMemoryBarrier()";
    case glsl: __intrinsic_asm "memoryBarrierBuffer()";
    case spirv:
    {
        spirv_asm
        {
            OpCapability Shader;
            OpMemoryBarrier Device AcquireRelease|UniformMemory
        };
    }
    }
}
[ForceInline]
[require(glsl_hlsl_spirv, shadermemorycontrol)]
public void memoryBarrierBuffer()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "DeviceMemoryBarrier()";
    case glsl: __intrinsic_asm "memoryBarrierBuffer()";
    case spirv:
    {
        spirv_asm
        {
            OpCapability Shader;
            OpMemoryBarrier Device AcquireRelease|UniformMemory
        };
    }
    }
}

[ForceInline]
[require(glsl_hlsl_spirv, shadermemorycontrol_compute)]
public void memoryBarrierShared()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "GroupMemoryBarrier()";
    case glsl: __intrinsic_asm "memoryBarrierShared()";
    case spirv:
    {
        spirv_asm
        {
            OpCapability Shader;
            OpMemoryBarrier Device AcquireRelease|WorkgroupMemory
        };
    }
    }
}
[ForceInline]
[require(glsl_hlsl_spirv, shadermemorycontrol)]
public void memoryBarrierImage()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "DeviceMemoryBarrier()";
    case glsl: __intrinsic_asm "memoryBarrierImage()";
    case spirv:
    {
        spirv_asm
        {
            OpMemoryBarrier Device AcquireRelease|ImageMemory
        };
    }
    }
}
[ForceInline]
[require(glsl_hlsl_spirv, shadermemorycontrol_compute)]
public void groupMemoryBarrier()
{
    __target_switch
    {
    case hlsl: __intrinsic_asm "GroupMemoryBarrier()";
    case glsl: __intrinsic_asm "groupMemoryBarrier()";
    case spirv:
    {
        spirv_asm
        {
            OpMemoryBarrier Workgroup AcquireRelease|WorkgroupMemory|ImageMemory|UniformMemory
        };
    }
    }
}

/// Section 8.18. Subpass-Input Functions

${{{{
struct SubpassLoadEntriesType
{
    const char* typeName;
    const char* prefix;
};
static const SubpassLoadEntriesType kSubpassLoadEntries[] =
    {
        { "vec4", "" },
        { "uvec4", "u" },
        { "ivec4", "i" },
    };
for (const auto& subpassLoadEntry : kSubpassLoadEntries)
{
}}}}
    [require(glsl_spirv, subpass)]
    public typealias $(subpassLoadEntry.prefix)subpassInput = __SubpassImpl<
        $(subpassLoadEntry.typeName),
        0 // isMS
    >;

    [require(glsl_spirv, subpass)]
    public typealias $(subpassLoadEntry.prefix)subpassInputMS = __SubpassImpl<
        $(subpassLoadEntry.typeName),
        1 // isMS
    >;
${{{{
}
}}}}

    __generic<T>
    [__NoSideEffect]
    [ForceInline]
    [require(glsl_hlsl_spirv, subpass)]
    public T subpassLoad(__SubpassImpl<T,0> subpass)
    {
        return subpass.SubpassLoad();
    }
    __generic<T>
    [__NoSideEffect]
    [ForceInline]
    [require(glsl_hlsl_spirv, subpass)]
    public T subpassLoad(__SubpassImpl<T,1> subpass, int sample)
    {
        return subpass.SubpassLoad(sample);
    }

/// Section 8.19. Shader Invocation Group Functions

// Invocation functions compiles into SubgroupVoteKHR operations.
// Instead of using these, we directly emit the non deprecated
// alternatives
__glsl_version(460)
__spirv_version(1.3)
[NonUniformReturn]
[ForceInline]
[require(glsl_hlsl_spirv, shaderinvocationgroup)]
public bool anyInvocation(bool value)
{
    return WaveActiveAnyTrue(value);
}
__glsl_version(460)
__spirv_version(1.3)
[NonUniformReturn]
[ForceInline]
[require(glsl_hlsl_spirv, shaderinvocationgroup)]
public bool allInvocations(bool value)
{
    return WaveActiveAllTrue(value);
}
__glsl_version(460)
__spirv_version(1.3)
[NonUniformReturn]
[ForceInline]
[require(glsl_hlsl_spirv, shaderinvocationgroup)]
public bool allInvocationsEqual(bool value)
{
    return WaveActiveAllEqual(value);
}


/// extensions
/// https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GL_EXT_nonuniform_qualifier.txt

__generic<T:__BuiltinArithmeticType>
__intrinsic_op($(kIROp_NonUniformResourceIndex))
[require(cpp_cuda_glsl_hlsl_spirv, nonuniformqualifier)]
public T nonuniformEXT(T index);

/// Debug output printing
[ForceInline]
public void debugPrintfEXT<each T>(NativeString format, expand each T args)
{
    printf(format, args);
}
