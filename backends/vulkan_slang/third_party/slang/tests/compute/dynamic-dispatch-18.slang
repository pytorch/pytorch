// Test using generic interface methods with dynamic dispatch.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_0 -use-dxil -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx11 -profile sm_5_0 -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute  -output-using-type

[anyValueSize(12)]
interface IReturnsZero
{
    float get();
}

[anyValueSize(16)]
interface IInterface
{
    associatedtype Getter : IReturnsZero;
    Getter createGetter();
}

struct Impl : IInterface
{
    int data;
    struct Getter : IReturnsZero
    {
        bool data;
        float get() { if (data) return 0.0; else return 1.0;}
    }
    Getter createGetter() { Getter g; g.data = true; return g; }
};


float test(IReturnsZero r)
{
    return r.get();
}

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<float> gOutputBuffer;

//TEST_INPUT: set gObj = new StructuredBuffer<IInterface>[new Impl{1}];
RWStructuredBuffer<IInterface> gObj;

//TEST_INPUT: type_conformance Impl:IInterface = 3

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float result = 0.0;
    let i = createDynamicObject<IInterface, int>(3, 0);
    IReturnsZero iobj = i.createGetter();
    result = test(iobj);
    gOutputBuffer[0] = result;
}
