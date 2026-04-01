// For some reason CI fails this compute test by not loading data correctly
//IGNORE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -xslang -fvk-use-dx-layout
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry computeMain -stage compute -fvk-use-dx-layout

//TEST_INPUT: set Test.v0 = 1;
//TEST_INPUT: set Test.v1 = {2,3};
//TEST_INPUT: set Test.v2 = 4;

struct Int_wrapper
{
//SPIRV: Offset 0
    uint v0;
//SPIRV: Offset 4
    uint v1;
};

cbuffer Test
{
//SPIRV: Offset 0
    uint v0;

//SPIRV: Offset 16
// struct's always start on a new register
    Int_wrapper v1;
//SPIRV: Offset 24
// Non-struct can pack with a partially filled register
    uint v2;
};

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[shader("compute")]
[numthreads(2, 2, 1)]
void computeMain()
{
    // CHECK: 64

    outputBuffer[1] = v0;
    outputBuffer[2] = v1.v0;
    outputBuffer[3] = v1.v1;
    outputBuffer[4] = v2;
    
    

    outputBuffer[0] = (true
            && v0 == 1
            && v1.v0 == 2
            && v1.v1 == 3
            && v2 == 4
        ) ? 100 : 0;
}