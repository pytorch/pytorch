//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK): -vk -compute

float getData<L:IBufferDataLayout>(StructuredBuffer<float3, L> p, int index)
{
    return p[index].x;
}

// GLSL: layout({{.*}}scalar{{.*}})

//TEST_INPUT:set scalarBuffer = ubuffer(data=[1 2 3 4 5 6], stride=12)
StructuredBuffer<float3, ScalarDataLayout> scalarBuffer;
//TEST_INPUT:set outputBuffer = out ubuffer(data=[0 0 0 0 0 0], stride=12)
RWStructuredBuffer<float3, ScalarDataLayout> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 1
    // CHECK: 1
    // CHECK: 1
    // CHECK: 4
    // CHECK: 4
    // CHECK: 4
    outputBuffer[0] = float3(getData(scalarBuffer, 0));
    outputBuffer[1] = float3(getData(scalarBuffer, 1));
}
