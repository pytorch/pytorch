//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -emit-spirv-directly
//DISABLED_TEST: COMPARE_COMPUTE(filecheck-buffer=CHECK): -wgpu
//DISABLED_TEST: COMPARE_COMPUTE(filecheck-buffer=CHECK): -d3d11
//DISABLED_TEST: COMPARE_COMPUTE(filecheck-buffer=CHECK): -d3d12
//DISABLED_TEST: COMPARE_COMPUTE(filecheck-buffer=CHECK): -metal

struct Obj
{
    int val;

    [mutating]
    void addOne() { val++; }

    int getValPlusOne() { return val + 1; }
}

//TEST_INPUT: set pObj = ubuffer(data=[2 0 0 0], stride=4);
uniform Obj* pObj;

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0],stride=4);
uniform RWStructuredBuffer<uint> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    pObj->addOne();
    // CHECK: 4
    outputBuffer[0] = pObj->getValPlusOne();
}