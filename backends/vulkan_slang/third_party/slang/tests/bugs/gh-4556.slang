//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -compute -output-using-type -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -compute -output-using-type -shaderobj
//TEST(compute):SIMPLE(filecheck=SPIRV): -target spirv-asm -stage compute
//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-mtl -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -cpu -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0.0 0.0 0.0 0.0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

//SPIRV-NOT: Validation of generated SPIR-V failed

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    float3x4 a = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    double3x4 b = (double3x4)a;

    // CHECK: 1.000000
    outputBuffer[0] = (float)b[0][0];
    // CHECK: 2.000000
    outputBuffer[1] = (float)b[0][1];

    // CHECK: 7.000000
    outputBuffer[2] = (float)b[1][2];

    // CHECK: 12.000000
    outputBuffer[3] = (float)b[2][3];
}
