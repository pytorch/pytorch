//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -shaderobj -emit-spirv-via-glsl -allow-glsl
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -shaderobj -emit-spirv-directly -allow-glsl
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute -profile cs_6_6 -dx12 -use-dxil -shaderobj -render-feature hardware-device -allow-glsl
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-metal -compute -shaderobj -allow-glsl
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cpu -compute -shaderobj -allow-glsl
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-wgpu -compute -shaderobj -allow-glsl
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cuda -compute -g0 -allow-glsl

//TEST_INPUT:ubuffer(data=[0x12345678], stride=4):name inputBuffer
StructuredBuffer<uint> inputBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<uint> outputBuffer;

bool verifyResult<T: IFloat> (T expected, T actual, T tolerance)
{
    return (expected - tolerance) <= actual && actual <= (expected + tolerance);
}

bool verifyResultVector<T: IFloat, let N: int>(vector<T, N> expected, vector<T, N> actual, T tolerance = T(0.01))
{
    bool isValid = true;
    for (int i = 0; i < N; ++i)
        isValid = isValid && verifyResult(expected[i], actual[i], tolerance);
    return isValid;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint packed = inputBuffer[0];
    uint index = 0U;

    //
    // Test GLSL intrinsics for unpacking floating points.
    // Packing intrinsics are tested in `tests/hlsl-intrinsic/packed/pack-unpack-float.slang`.
    //

    float4 u4x8Expected = float4(0.4706, 0.3373, 0.2039, 0.0706);
    float4 u4x8Float = unpackUnorm4x8(packed);
    // BUF: 1
    outputBuffer[index++] = verifyResultVector(u4x8Expected, u4x8Float);

    float4 s4x8Expected = float4(0.9449, 0.6772, 0.4094, 0.1417);
    float4 s4x8Float = unpackSnorm4x8(packed);
    // BUF-NEXT: 1
    outputBuffer[index++] = verifyResultVector(s4x8Expected, s4x8Float);

    float2 u2x16Expected = float2(0.3377, 0.0711);
    float2 u2x16Float = unpackUnorm2x16(packed);
    // BUF-NEXT: 1
    outputBuffer[index++] = verifyResultVector(u2x16Expected, u2x16Float);

    float2 s2x16Expected = float2(0.6756, 0.1422);
    float2 s2x16Float = unpackSnorm2x16(packed);
    // BUF-NEXT: 1
    outputBuffer[index++] = verifyResultVector(s2x16Expected, s2x16Float);

    float2 h2x16Expected = float2(103.5, 0.000757);
    float2 h2x16Float = unpackHalf2x16(packed);
    // BUF-NEXT: 1
    outputBuffer[index++] = verifyResultVector(h2x16Expected, h2x16Float);
}
