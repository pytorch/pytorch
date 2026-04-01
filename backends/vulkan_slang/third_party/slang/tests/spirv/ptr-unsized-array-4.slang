//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -emit-spirv-directly
//DISABLED_TEST: COMPARE_COMPUTE(filecheck-buffer=CHECK): -wgpu
//DISABLED_TEST: COMPARE_COMPUTE(filecheck-buffer=CHECK): -d3d11
//DISABLED_TEST: COMPARE_COMPUTE(filecheck-buffer=CHECK): -d3d12
//DISABLED_TEST: COMPARE_COMPUTE(filecheck-buffer=CHECK): -metal

// Test a pointer to a struct that has only one field and is an unsized array.
struct MeshStorage {
    uint64_t QuadData[];
};

//TEST_INPUT: set pStorage = ubuffer(data=[1 2 3 4 5 6 7 8],stride=4);
uniform MeshStorage* pStorage;

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0],stride=4);
uniform RWStructuredBuffer<uint> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 3
    // CHECK: 4
    outputBuffer[0] = (int)(pStorage.QuadData[1]&0xFFFFFFFF);
    outputBuffer[1] = (int)(pStorage.QuadData[1]>>32);
}