// Test that we can zero-init a struct with interface typed member.

//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER): -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER): -vk -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface IFoo
{
    int method();
}

//TEST_INPUT: type_conformance Impl1:IFoo = 0
struct Impl1 : IFoo
{
    int data;
    int method() { return data + 1; }
}

struct MyType
{
    IFoo foo;
}


[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    MyType t = {};
    // BUFFER: 1
    outputBuffer[0] = t.foo.method();
}
