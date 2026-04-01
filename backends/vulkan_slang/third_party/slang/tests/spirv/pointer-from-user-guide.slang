//TEST(compute, vulkan):SIMPLE(filecheck=SPV): -stage compute -entry computeMain -target spirv -emit-spirv-directly

// This test is to verify the example on the user guide
// and prevent any regressions.

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct MyType
{
    float a;
};

float test(MyType* pObj)
{
    //SPV: OpTypePointer
    MyType* pNext = pObj + 1;
    MyType* pNext2 = &pNext[1];
    return pNext.a + pNext->a + (*pNext2).a + pNext2[0].a;
}

cbuffer Constants
{
    MyType *ptr;
};

[numthreads(1,1,1)]
void computeMain()
{
    outputBuffer[0] = test(ptr);
}
