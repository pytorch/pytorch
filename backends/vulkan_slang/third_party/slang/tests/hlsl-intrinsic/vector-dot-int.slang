//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj -output-using-type

// No 16-bit and 64-bit integer support on DX11.
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type -xslang -DDX11

//TEST(compute, vulkan):SIMPLE(filecheck=CHECK_SPV): -stage compute -entry computeMain -target spirv

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = int(dispatchThreadID.x);
    uint outIndex = 0;

    // CHECK_SPV: OpSDot
    int3 a = { index - 1, index - 2, index - 3};
    int3 b = { 1, 2, 3};
    outputBuffer[outIndex++] = dot(a, b);

    // CHECK_SPV: OpUDot
    uint3 c = { index + 1, index + 2, index + 3};
    uint3 d = { 2, 4, 6};
    outputBuffer[outIndex++] = int(dot(c, d));

#if !defined(DX11)
    // CHECK_SPV: OpUDot
    uint64_t2 e = { index + 1, index + 2};
    uint64_t2 f = { 4, 8};
    outputBuffer[outIndex++] = int(dot(e, f));

    // CHECK_SPV: OpSDot
    int16_t4 g = { int16_t(index + 1), int16_t(index + 2), int16_t(index + 3), int16_t(index + 4)};
    int16_t4 h = { -1, 2, 2, -1};
    outputBuffer[outIndex++] = int(dot(g, h));
#else
    outputBuffer[outIndex++] = 20;
    outputBuffer[outIndex++] = 5;
#endif
}
