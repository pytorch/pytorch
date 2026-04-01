//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -vk -emit-spirv-directly
groupshared Atomic<uint> values[10];

//TEST_INPUT: set outputAddr = out ubuffer(data=[0 0 0 0], stride=4)
uniform uint64_t outputAddr;

[numthreads(4,1,1)]
void computeMain(int i : SV_DispatchThreadID)
{
    uint* output = (uint*)(outputAddr);
    values[i] = 0;
    values[i] += 1;
    output[i] = values[i].load();

    // CHECK: 1
    // CHECK: 1
    // CHECK: 1
    // CHECK: 1
}