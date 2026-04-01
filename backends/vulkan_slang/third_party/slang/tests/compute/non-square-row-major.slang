// non-square-row-major.slang

// Note! This test doesn't work on CUDA or CPU targets, because both these targets 
// assume matrices are tightly packed, whereas GPU targets align rows to 16 bytes.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=ALIGNED):-slang -compute -wgpu -output-using-type -xslang -matrix-layout-row-major -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=PACKED):-cpu -compute -output-using-type -compile-arg -O3 -xslang -matrix-layout-row-major -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=ALIGNED):-slang -compute -output-using-type -xslang -matrix-layout-row-major -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=ALIGNED):-slang -compute -dx12 -output-using-type -xslang -matrix-layout-row-major -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=ALIGNED):-vk -compute -output-using-type -xslang -matrix-layout-row-major -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=PACKED):-cuda -compute -output-using-type -xslang -matrix-layout-row-major -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=PACKED): -slang -output-using-type -shaderobj -mtl

// matrix<R, C>
//TEST_INPUT:cbuffer(data=[1.0 2.0 3.0 4.0 5.0 6.0 0.0 0.0  10.0 20.0 0.0 0.0 ]):name matrixBuffer
ConstantBuffer<float3x2> matrixBuffer;

//TEST_INPUT:ubuffer(data=[0 0], stride=4):out,name output
RWStructuredBuffer<float> output;

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    float3 v = float3(1, 2, 1);

    float3x2 M = matrixBuffer;
    
    float2 r = mul(v, M);

    // ALIGNED: 21
    // ALIGNED: 34

    // PACKED: 12
    // PACKED: 16
    output[0] = r.x;
    output[1] = r.y;
}
