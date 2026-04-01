//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

// Confirm that generics syntax can be used in user
// code and generates valid output.

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float4> outputBuffer;

interface IElement
{
    float4 getValue();
};

struct SingleElement : IElement
{
    float4 value;
    float4 getValue()
    {
        return value;
    }
};

struct ListElement<THead : IElement, TTail : IElement> : IElement
{
    THead head;
    TTail tail;
    float4 getValue()
    {
        return head.getValue() + tail.getValue();
    }
};

float4 test<T : IElement>(T val)
{
    return val.getValue();
}


[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    ListElement<SingleElement, ListElement<SingleElement, ListElement<SingleElement, SingleElement> > > list;
    list.head.value = float4(1.0);
    list.tail.head.value = float4(2.0);
    list.tail.tail.head.value = float4(3.0);
    list.tail.tail.tail.value = float4(4.0);
    float4 outVal = test(list);
    outputBuffer[0] = outVal;
}