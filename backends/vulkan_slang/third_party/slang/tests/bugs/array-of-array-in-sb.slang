//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk

struct S
{
    int vertices[3][3];
}

//TEST_INPUT:set cs = cbuffer(data=[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1])
ConstantBuffer<S> cs;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0], stride=4),name=s_buf
RWStructuredBuffer<S> s_buf;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK: 6

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    s_buf[0].vertices[1][1] = 5;
    outputBuffer[0] = s_buf[0].vertices[1][1] + cs.vertices[1][0];
}