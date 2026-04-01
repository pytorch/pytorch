// Test use of unrelated generics in a dynamically dispatched generic function can be specialized.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_0 -use-dxil -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx11 -profile sm_5_0 -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute  -output-using-type

interface IValueGetter
{
    float get();
}

struct MyWrapper<T:IValueGetter>
{
    T getter;
    float getVal()
    {
        return getter.get();
    }
}

interface IFoo
{
    int getIntVal(int x);
}

[anyValueSize(16)]
interface IInterface
{
    float callGetter<T:IFoo>(T foo);
}

struct Impl : IInterface
{
    int data;
    struct GetterImpl : IValueGetter { float val; float get(){return val;} }
    float callGetter<T:IFoo>(T foo)
    {
        // We should be able to specialize `wrapper` even if it is
        // used in this dynamically dispatched method.
        MyWrapper<GetterImpl> wrapper;
        wrapper.getter.val = 2.0f;
        return wrapper.getVal() * foo.getIntVal(1);
    }
};

struct IdentityFoo : IFoo { int getIntVal(int x) { return x; } }

float test()
{
    IInterface obj = createDynamicObject<IInterface, uint4>(3, uint4(1,2,3,4));
    IdentityFoo foo;
    // Call via dynamic dispatch
    return obj.callGetter(foo);
}

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<float> gOutputBuffer;

//TEST_INPUT: type_conformance Impl:IInterface = 3

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gOutputBuffer[0] = test();
}
