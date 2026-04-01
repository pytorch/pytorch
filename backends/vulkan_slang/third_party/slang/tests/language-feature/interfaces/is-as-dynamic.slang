// is-as-dynamic.slang

// Test that `is` and `as` operators works as intended in dynamic dispatch.

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -vk -compute
//TEST(compute):COMPARE_COMPUTE_EX:-slang -cpu -compute

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[anyValueSize(8)]
interface IFoo
{
    int method();
}

//TEST_INPUT: type_conformance Impl1:IFoo = 0
struct Impl1 : IFoo
{
    int data;
    int method() { return data; }
}

//TEST_INPUT: type_conformance Impl2:IFoo = 1
struct Impl2 : IFoo
{
    int data1;
    int data2;
    int method() { return data1 + data2; }
}

int getData(IFoo foo)
{
    let castResult = foo as Impl2;
    if (castResult.hasValue && foo is Impl2 && !(foo is Impl1))
    {
        return castResult.value.method();
    }
    return 0;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    int2 data = int2(1, 2);
    IFoo dynamicObject = createDynamicObject<IFoo, int2>(1, data);
    int outVal = getData(dynamicObject);
    outputBuffer[0] = outVal;
}
