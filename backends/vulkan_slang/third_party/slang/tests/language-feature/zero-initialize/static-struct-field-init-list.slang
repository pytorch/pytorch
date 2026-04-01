//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -xslang -zero-initialize

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

struct structData
{
    uint data;
};

struct structOuterData
{
    static structData data1 = { 1 };
};

[numthreads(1,1,1)]
void computeMain()
{
    // BUF: 1
    structOuterData mat;
    outputBuffer[0] = true
        && mat.data1.data == 1
        ;
}