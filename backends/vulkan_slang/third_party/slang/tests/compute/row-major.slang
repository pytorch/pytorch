// row-major.slang

//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -compile-arg -O3 -xslang -matrix-layout-row-major -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type -xslang -matrix-layout-row-major -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -output-using-type -xslang -matrix-layout-row-major -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -xslang -matrix-layout-row-major -shaderobj

//TEST_INPUT:cbuffer(data=[1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0 10.0 20.0 30.0 1.0]):name matrixBuffer
ConstantBuffer<float4x4> matrixBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name output
RWStructuredBuffer<float> output;

[numthreads(1, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    float4 v = float4(1, 2, 3, 1);

    float4x4 M = matrixBuffer;
    
    float4 r = mul(v, M);

    output[0] = r.x;
    output[1] = r.y;
    output[2] = r.z;
    output[3] = r.w;
}
