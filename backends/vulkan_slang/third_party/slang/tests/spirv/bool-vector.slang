// bool-vector.slang

//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -output-using-type

// Test direct SPIR-V emit on arrays in uniforms.

//TEST_INPUT:set resultBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<uint> resultBuffer;

struct Data
{
    bool4 bv;
};

//TEST_INPUT:set dataBuffer = ubuffer(data=[0 1 1 0], stride=4)
RWStructuredBuffer<Data> dataBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    // CHECK: 0
    // CHECK: 100
    resultBuffer[0] = dataBuffer[0].bv.x;
    resultBuffer[1] = dataBuffer[0].bv.y ? 100 : 0;
}
