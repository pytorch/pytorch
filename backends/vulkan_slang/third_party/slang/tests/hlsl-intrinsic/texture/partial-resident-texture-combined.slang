// TODO: There are issues running tests combined texture samplers in DX12. (Github issue #6982)
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -compute -shaderobj -output-using-type -use-dxil -profile cs_6_7 -dx12 -Xslang -DTARGET_DX12

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-vk -compute -shaderobj -output-using-type -emit-spirv-directly -render-feature hardware-device -xslang -DVK

//TEST_INPUT: ubuffer(data=[2], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT:ubuffer(data=[1 1 1 1]):name=iBuf
RWByteAddressBuffer iBuf;

//
// Combined texture samplers.
//

//TEST_INPUT: TextureSampler1D(size=4, content = one):name st1D_f32v3
Sampler1D<float3> st1D_f32v3;
//TEST_INPUT: TextureSampler2D(size=4, content = one):name st2D_f32v3
Sampler2D<float3> st2D_f32v3;
//TEST_INPUT: TextureSampler3D(size=4, content = one):name st3D_f32v3
Sampler3D<float3> st3D_f32v3;

//TEST_INPUT: TextureSampler1D(size=4, content = one, arrayLength=2):name st1DArray_f32v3
Sampler1DArray<float3> st1DArray_f32v3;
//TEST_INPUT: TextureSampler2D(size=4, content = one, arrayLength=2):name st2DArray_f32v3
Sampler2DArray<float3> st2DArray_f32v3;

//TEST_INPUT: TextureSampler1D(size=4, content = one):name st1D_f32v4
Sampler1D<float4> st1D_f32v4;
//TEST_INPUT: TextureSampler2D(size=4, content = one):name st2D_f32v4
Sampler2D<float4> st2D_f32v4;
//TEST_INPUT: TextureSampler3D(size=4, content = one):name st3D_f32v4
Sampler3D<float4> st3D_f32v4;

//TEST_INPUT: TextureSampler1D(size=4, content = one, arrayLength=2):name st1DArray_f32v4
Sampler1DArray<float4> st1DArray_f32v4;
//TEST_INPUT: TextureSampler2D(size=4, content = one, arrayLength=2):name st2DArray_f32v4
Sampler2DArray<float4> st2DArray_f32v4;

//TEST_INPUT: TextureSampler2D(size=4, content = one):name st2DMS_f32v4
 Sampler2DMS<float4> st2DMS_f32v4;

//
// Combined depth texture samplers.
//

__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias CombinedDepth2d = _Texture<
    T,
    __Shape2D,
    0, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    1, // isCombined
    format
>;

__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias CombinedDepth2d_array = _Texture<
    T,
    __Shape2D,
    1, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    1, // isCombined
    format
>;


//TEST_INPUT: TextureSampler2D(size=4, content = zero):name cd2D
CombinedDepth2d<float> cd2D;
//TEST_INPUT: TextureSampler2D(size=4, content = zero, arrayLength=2):name cd2DArray
CombinedDepth2d_array<float> cd2DArray;

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

bool TEST_combinedDepth()
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
        && (status = getNotMapped(), all(0.0 == cd2D.SampleCmp(float2(u), compareValue, int2(offset), clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(0.0 == cd2DArray.SampleCmp(float3(u, u, slice), compareValue, int2(offset), clamp, status))) && CheckAccessFullyMapped(status)

        // ==========================
        // float SampleCmpLevelZero()
        // ==========================
        && (status = getNotMapped(), all(0.0 == cd2D.SampleCmpLevelZero(float2(u), compareValue, int2(offset), status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(0.0 == cd2DArray.SampleCmpLevelZero(float3(u, u, slice), compareValue, int2(offset), status))) && CheckAccessFullyMapped(status)

        // ======================
        // float SampleCmpLevel()
        // ======================
        && (status = getNotMapped(), all(0.0 == cd2D.SampleCmpLevel(float2(u), compareValue, level, int2(offset), status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(0.0 == cd2DArray.SampleCmpLevel(float3(u, u, slice), compareValue, level, int2(offset), status))) && CheckAccessFullyMapped(status)
        ;
}

bool TEST_sampler<T>(
    Sampler1D<T> t1D,
    Sampler2D<T> t2D,
    Sampler3D<T> t3D,
    Sampler1DArray<T> t1DArray,
    Sampler2DArray<T> t2DArray,
) where T : ITexelElement, IArithmetic
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
        && (status = getNotMapped(), all(TN(T(1)) == t1D.Sample(u, offset, clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2D.Sample(float2(u), int2(offset), clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t3D.Sample(float3(u), int3(offset), clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t1DArray.Sample(float2(u, slice), offset, clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2DArray.Sample(float3(u, u, slice), offset, clamp, status))) && CheckAccessFullyMapped(status)

        // ==============
        // T SampleBias()
        // ==============
        && (status = getNotMapped(), all(TN(T(1)) == t1D.SampleBias(u, bias, offset, clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2D.SampleBias(float2(u), bias, int2(offset), clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t3D.SampleBias(float3(u), bias, int3(offset), clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t1DArray.SampleBias(float2(u, slice), bias, offset, clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2DArray.SampleBias(float3(u, u, slice), bias, offset, clamp, status))) && CheckAccessFullyMapped(status)

        // ==============
        // T SampleGrad()
        // ==============
        && (status = getNotMapped(), all(TN(T(1)) == t1D.SampleGrad(u, ddx, ddy, offset, clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2D.SampleGrad(float2(u), ddx, ddy, int2(offset), clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t3D.SampleGrad(float3(u), ddx, ddy, int3(offset), clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t1DArray.SampleGrad(float2(u, slice), ddx, ddy, offset, clamp, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2DArray.SampleGrad(float3(u, u, slice), ddx, ddy, offset, clamp, status))) && CheckAccessFullyMapped(status)

        // ==============
        // T SampleLevel()
        // ==============
        && (status = getNotMapped(), all(TN(T(1)) == t1D.SampleLevel(u, level, offset, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2D.SampleLevel(float2(u), level, int2(offset), status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t3D.SampleLevel(float3(u), level, int3(offset), status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t1DArray.SampleLevel(float2(u, slice), level, offset, status))) && CheckAccessFullyMapped(status)
        && (status = getNotMapped(), all(TN(T(1)) == t2DArray.SampleLevel(float3(u, u, slice), level, offset, status))) && CheckAccessFullyMapped(status)
        ;
}

bool TEST_sparse<T>(
     Sampler2D<T> s2D,
     Sampler2DMS<T> s2DMS)
     where T : ITexelElement, IArithmetic
 {
     typealias TN = T;
     constexpr const int2 offset = int2(0, 0);
     uint status;
 
     int sampleIndex = 0;
     int2 iuv = int2(1, 1);
     int3 iuvs = int3(iuv, sampleIndex);
 
     return true
         && (status = getNotMapped(), all(TN(T(1)) == s2D.Load(iuvs, offset, status))) && CheckAccessFullyMapped(status)
         && (status = getNotMapped(), all(TN(T(1)) == s2DMS.Load(iuv, sampleIndex, offset, status))) && CheckAccessFullyMapped(status)
         ;
 }

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    bool result = true
        // Make sure CheckAccessFullyMapped can return false
        && (!CheckAccessFullyMapped(getNotMapped()))
        && TEST_sampler(
            st1D_f32v3,
            st2D_f32v3,
            st3D_f32v3,
            st1DArray_f32v3,
            st2DArray_f32v3,
            )
        && TEST_sampler(
            st1D_f32v4,
            st2D_f32v4,
            st3D_f32v4,
            st1DArray_f32v4,
            st2DArray_f32v4,
            )
        && TEST_combinedDepth()
#if !defined(TARGET_DX12) // HLSL doesn't support samplers for `Load` functions.
        && TEST_sparse(st2D_f32v4, st2DMS_f32v4)
#endif
        ;

    //CHK:1
    outputBuffer[0] = int(result);
}
