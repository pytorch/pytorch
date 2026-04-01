//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-dx12 -use-dxil -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-cpu -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -shaderobj

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct NestedStruct
{
    int data;
};

struct TopStruct
{
    NestedStruct nestedStruct = {};
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TopStruct val = {};
// BUF: 1
    outputBuffer[0] = val.nestedStruct.data == 0;
}
