// optional.slang

// Test that `Optional` construction and conversion works.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute
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

Optional<Impl1> getVal(bool shouldHaveVal)
{
    if (shouldHaveVal)
    {
        Impl1 val;
        val.data = 1;
        return val;
    }
    return none;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    let v1 = getVal(true);
    let v2 = getVal(false);
    int result = 0;
    if (v1.hasValue)
        result += v1.value.data;
    if (v2.hasValue)
        result += v2.value.data;
    outputBuffer[0] = result;
    if (v1 != none)
        result += v1.value.data;
    if (!(v2 == none))
        result += v2.value.data;
    outputBuffer[1] = result;

}
