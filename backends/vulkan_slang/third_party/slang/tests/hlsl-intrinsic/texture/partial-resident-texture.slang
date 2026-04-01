//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -compute -shaderobj -output-using-type -use-dxil -profile cs_6_7 -dx12
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-vk -compute -shaderobj -output-using-type -emit-spirv-directly -render-feature hardware-device -xslang -DVK

//TEST_INPUT: ubuffer(data=[2], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT:ubuffer(data=[1 1 1 1]):name=iBuf
RWByteAddressBuffer iBuf;

//TEST_INPUT: Sampler:name samplerState
SamplerState samplerState;

//TEST_INPUT: Sampler:name samplerCmpState
SamplerComparisonState samplerCmpState;

//
// Textures.
//

//TEST_INPUT: Texture1D(size=4, content = one):name t1D_f32v3
Texture1D<float3> t1D_f32v3;
//TEST_INPUT: Texture2D(size=4, content = one):name t2D_f32v3
Texture2D<float3> t2D_f32v3;
//TEST_INPUT: Texture3D(size=4, content = one):name t3D_f32v3
Texture3D<float3> t3D_f32v3;
//TEST_INPUT: TextureCube(size=4, content = one):name tCube_f32v3
TextureCube<float3> tCube_f32v3;
//TEST_INPUT: Texture1D(size=4, content = one, arrayLength=2):name t1DArray_f32v3
Texture1DArray<float3> t1DArray_f32v3;
//TEST_INPUT: Texture2D(size=4, content = one, arrayLength=2):name t2DArray_f32v3
Texture2DArray<float3> t2DArray_f32v3;
//TEST_INPUT: TextureCube(size=4, content = one, arrayLength=2):name tCubeArray_f32v3
TextureCubeArray<float3> tCubeArray_f32v3;

//TEST_INPUT: Texture1D(size=4, content = one):name t1D_f32v4
Texture1D<float4> t1D_f32v4;
//TEST_INPUT: Texture2D(size=4, content = one):name t2D_f32v4
Texture2D<float4> t2D_f32v4;
//TEST_INPUT: Texture3D(size=4, content = one):name t3D_f32v4
Texture3D<float4> t3D_f32v4;
//TEST_INPUT: TextureCube(size=4, content = one):name tCube_f32v4
TextureCube<float4> tCube_f32v4;

//TEST_INPUT: Texture1D(size=4, content = one, arrayLength=2):name t1DArray_f32v4
Texture1DArray<float4> t1DArray_f32v4;
//TEST_INPUT: Texture2D(size=4, content = one, arrayLength=2):name t2DArray_f32v4
Texture2DArray<float4> t2DArray_f32v4;
//TEST_INPUT: TextureCube(size=4, content = one, arrayLength=2):name tCubeArray_f32v4
TextureCubeArray<float4> tCubeArray_f32v4;

//TEST_INPUT: Texture2D(size=4, content = one):name t2DMS_f32v4
Texture2DMS<float4> t2DMS_f32v4;

//
// Depth textures.
//
__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias depth2d = _Texture<
    T,
    __Shape2D,
    0, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    0, // isCombined
    format
>;

__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias depth2d_array = _Texture<
    T,
    __Shape2D,
    1, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    0, // isCombined
    format
>;

__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias depthcube = _Texture<
    T,
    __ShapeCube,
    0, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    0, // isCombined
    format
>;

__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias depthcube_array = _Texture<
    T,
    __ShapeCube,
    1, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    0, // isCombined
    format
>;

//TEST_INPUT: Texture2D(size=4, content = zero):name d2D
depth2d<float> d2D;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name d2DArray
depth2d_array<float> d2DArray;

uint getNotMapped()
{
    // We want to return a status uint that causes `CheckAccessFullyMapped` to return false.
    // These are just educated guesses - actual implementation differ between platforms and drivers.
#if defined(VK)
    return 0xFFFFFFFFU;
#else
    return 0;
#endif
}

bool TEST_depthTexture()
{
    float u = 0.0;
    int offset = 0;
    float clamp = 0.0;
    float slice = 0.0;
    float level = 0.0;
    float compareValue = 0.0;

    uint status;

    return true
        // =================
        // float SampleCmp()
        // =================
        && (status = getNotMapped(), all(0.0 == d2D.SampleCmp(samplerCmpState, float2(u), compareValue, int2(offset), clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(0.0 == d2DArray.SampleCmp(samplerCmpState, float3(u, u, slice), compareValue, int2(offset), clamp, status))) && CheckAccessFullyMapped(status)

        // ==========================
        // float SampleCmpLevelZero()
        // ==========================
        && (status = getNotMapped(), all(0.0 == d2D.SampleCmpLevelZero(samplerCmpState, float2(u), compareValue, int2(offset), status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(0.0 == d2DArray.SampleCmpLevelZero(samplerCmpState, float3(u, u, slice), compareValue, int2(offset), status))) && CheckAccessFullyMapped(status)

        // ======================
        // float SampleCmpLevel()
        // ======================
        && (status = getNotMapped(), all(0.0 == d2D.SampleCmpLevel(samplerCmpState, float2(u), compareValue, level, int2(offset), status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(0.0 == d2DArray.SampleCmpLevel(samplerCmpState, float3(u, u, slice), compareValue, level, int2(offset), status))) && CheckAccessFullyMapped(status)
        ;
}

bool TEST_texture<T>(
    Texture1D<T> t1D,
    Texture2D<T> t2D,
    Texture3D<T> t3D,
    TextureCube<T> tCube,
    Texture1DArray<T> t1DArray,
    Texture2DArray<T> t2DArray,
    TextureCubeArray<T> tCubeArray,
    )
    where T : ITexelElement, IArithmetic
{
    typealias TN = T;

    float u = 0.0;
    int offset = 0;
    float clamp = 0.0;
    float slice = 0.0;
    float bias = 0.0;
    float grad = 0.0;
    float level = 0.0;
    constexpr const float ddx = 0.0f;
    constexpr const float ddy = 0.0f;

    uint status;

    return true
        // ==========
        // T Sample()
        // ==========
        && (status = getNotMapped(), all(TN(T(1)) == t1D.Sample(samplerState, u, offset, clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2D.Sample(samplerState, float2(u), int2(offset), clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t3D.Sample(samplerState, float3(u), int3(offset), clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t1DArray.Sample(samplerState, float2(u, slice), offset, clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2DArray.Sample(samplerState, float3(u, u, slice), offset, clamp, status))) && CheckAccessFullyMapped(status)

        // ==============
        // T SampleBias()
        // ==============
        && (status = getNotMapped(), all(TN(T(1)) == t1D.SampleBias(samplerState, u, bias, offset, clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2D.SampleBias(samplerState, float2(u), bias, int2(offset), clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t3D.SampleBias(samplerState, float3(u), bias, int3(offset), clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t1DArray.SampleBias(samplerState, float2(u, slice), bias, offset, clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2DArray.SampleBias(samplerState, float3(u, u, slice), bias, offset, clamp, status))) && CheckAccessFullyMapped(status)

        // ==============
        // T SampleGrad()
        // ==============
        && (status = getNotMapped(), all(TN(T(1)) == t1D.SampleGrad(samplerState, u, ddx, ddy, offset, clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2D.SampleGrad(samplerState, float2(u), ddx, ddy, int2(offset), clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t3D.SampleGrad(samplerState, float3(u), ddx, ddy, int3(offset), clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t1DArray.SampleGrad(samplerState, float2(u, slice), ddx, ddy, offset, clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2DArray.SampleGrad(samplerState, float3(u, u, slice), ddx, ddy, offset, clamp, status))) && CheckAccessFullyMapped(status)

        // ==============
        // T SampleLevel()
        // ==============
        && (status = getNotMapped(), all(TN(T(1)) == t1D.SampleLevel(samplerState, u, level, offset, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2D.SampleLevel(samplerState, float2(u), level, int2(offset), status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t3D.SampleLevel(samplerState, float3(u), level, int3(offset), status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t1DArray.SampleLevel(samplerState, float2(u, slice), level, offset, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2DArray.SampleLevel(samplerState, float3(u, u, slice), level, offset, status))) && CheckAccessFullyMapped(status)
        ;
}

bool TEST_load<T>(
    Texture2D<T> t2D,
    Texture2DMS<T> t2DMS)
    where T : ITexelElement, IArithmetic
{
    typealias TN = T;
    constexpr const int2 offset = int2(0, 0);
    uint status;
    float clamp = 0;

    float2 uv = float2(0.5f, 0.5f);

    int sampleIndex = 0;
    int2 iuv = int2(1, 1);
    int3 iuvs = int3(iuv, sampleIndex);

    return true
        && (status = getNotMapped(), all(TN(T(1)) == t2DMS.Load(iuv, sampleIndex, offset, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2D.Load(iuvs, offset, status))) && CheckAccessFullyMapped(status)
            // SPIRV does not support sparse buffer loads.
#if !defined(VK)
        && (status = getNotMapped(), 2 == outputBuffer.Load(0, status)) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), 1 == iBuf.Load(0, status)) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(int2(1) == iBuf.Load2(0, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(int3(1) == iBuf.Load3(0, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(int4(1) == iBuf.Load4(0, status))) && CheckAccessFullyMapped(status)
#endif
        ;
}

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    bool result = true
        // Make sure CheckAccessFullyMapped can return false
        && (!CheckAccessFullyMapped(getNotMapped()))
        && TEST_texture(
            t1D_f32v3,
            t2D_f32v3,
            t3D_f32v3,
            tCube_f32v3,
            t1DArray_f32v3,
            t2DArray_f32v3,
            tCubeArray_f32v3,
            )
        &&  TEST_texture(
            t1D_f32v4,
            t2D_f32v4,
            t3D_f32v4,
            tCube_f32v4,
            t1DArray_f32v4,
            t2DArray_f32v4,
            tCubeArray_f32v4,
            )
        && TEST_depthTexture()
        && TEST_load(t2D_f32v4, t2DMS_f32v4)
        ;

    //CHK:1
    outputBuffer[0] = int(result);
}
