// is-as.slang

// Test that `is` and `as` operators works as intended.

//TEST(compute):COMPARE_COMPUTE: -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface IFoo
{
    int method();
}

struct Impl1 : IFoo
{
    int data;
    int method() { return data; }
}

struct Impl2 : IFoo
{
    int data1;
    int data2;
    int method() { return data1 + data2; }
}

int getData(IFoo foo)
{
    let castResult = foo as Impl2;
    if (castResult.hasValue)
    {
        return castResult.value.method();
    }
    return 0;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    Impl2 obj;
    obj.data1 = 1;
    obj.data2 = 2;
    int outVal = getData(obj);
    outputBuffer[0] = outVal;
}
