// Test using generic interface methods with dynamic dispatch.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -vk -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_0 -use-dxil -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx11 -profile sm_5_0 -output-using-type

interface IReturnsZero
{
    float get();
}

[anyValueSize(16)]
interface IInterface
{
    float run<Z:IReturnsZero>();
}

struct UserDefinedPackedType
{
    float3 val;
    uint flags;
};

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<float> gOutputBuffer;

//TEST_INPUT: set gObj = new StructuredBuffer<UserDefinedPackedType>[new UserDefinedPackedType{[1.0, 2.0, 3.0], 3}, new UserDefinedPackedType{[2.0, 3.0, 4.0], 4}];
RWStructuredBuffer<UserDefinedPackedType> gObj;

//TEST_INPUT: type_conformance FloatVal:IInterface = 3
//TEST_INPUT: type_conformance Float4Val:IInterface = 4

[numthreads(1, 1, 1)]
void computeMain(uint3       dispatchThreadID : SV_DispatchThreadID)
{
    float result = 0.0;
    for (int i = 0; i < 2; i++)
    {
        var rawObj = gObj.Load(i);
        IInterface dynamicObj = createDynamicObject<IInterface, UserDefinedPackedType>(rawObj.flags, rawObj);
        result += dynamicObj.run<ReturnsZero>();
    }
    gOutputBuffer[0] = result;
}

struct ReturnsZero : IReturnsZero
{
    float get() { return 0.0; }
}
struct FloatVal : IInterface
{
    float val;
    float run<Z:IReturnsZero>()
    {
        Z z = Z();
        return val + z.get();
    }
};
struct Float4Struct { float4 val; }
struct Float4Val : IInterface
{
    Float4Struct val;
    float run<Z:IReturnsZero>()
    {
        Z z = Z();
        return val.val.x + val.val.y + z.get();
    }
};
