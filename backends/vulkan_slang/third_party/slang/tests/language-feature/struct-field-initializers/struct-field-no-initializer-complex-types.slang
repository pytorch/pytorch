//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -emit-spirv-directly
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-cpu -compute -entry computeMain
//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-dx12 -use-dxil -compute -entry computeMain

//TEST_INPUT:ubuffer(data=[0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct DefaultData : IDefaultInitializable
{
    static const int2 val = int2(0, 1);
    float2 size;
    float scale;
    float bias;
};

extension DefaultData
{
    int someGet() { return val.x; }
}

int loadDefaultData(DefaultData noInit)
{
    outputBuffer[1] = 1;
    return noInit.someGet();
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    DefaultData noInit;
    outputBuffer[0] = true
        && loadDefaultData(noInit) == 0;
        ;
    // BUF: 1
    // BUF-NEXT: 1
}