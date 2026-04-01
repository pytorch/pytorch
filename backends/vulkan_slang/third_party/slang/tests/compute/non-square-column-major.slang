// non-square-column-major.slang

// Note! This test doesn't work on CUDA or CPU targets, because both these targets ignore
// row/column major setting, as well as they handle alignment differently.

//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -compile-arg -O3 -xslang -matrix-layout-column-major -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type -xslang -matrix-layout-column-major -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -output-using-type -xslang -matrix-layout-column-major -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -xslang -matrix-layout-column-major -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type -xslang -matrix-layout-column-major -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=METAL):-slang  -output-using-type -shaderobj -mtl

// matrix<R, C>
//TEST_INPUT:cbuffer(data=[1.0 0.0 10.0 0.0  0.0 1.0 20.0 0.0]):name matrixBuffer
ConstantBuffer<float3x2> matrixBuffer;

//TEST_INPUT:ubuffer(data=[0 0], stride=4):out,name output
RWStructuredBuffer<float> output;

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    float3 v = float3(1, 2, 1);

    float3x2 M = matrixBuffer;
    
    float2 r = mul(v, M);

    // Metal always uses scalar layout + row major, so it is computing
    //              [1  0]
    // [ 1 2 1]  *  [10 0]  = [ 21 1 ]
    //              [0  1]
    // METAL: 21
    // METAL: 1
    output[0] = r.x;
    output[1] = r.y;
}
