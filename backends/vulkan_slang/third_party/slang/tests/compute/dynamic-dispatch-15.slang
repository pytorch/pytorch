// Test packing/unpacking different types of fields into `AnyValue`s.

//dTEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cpu -output-using-type
//dTEST(compute):COMPARE_COMPUTE_EX:-slang -compute -vk -output-using-type
//dTEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_0 -use-dxil -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx11 -profile sm_5_0 -output-using-type

[anyValueSize(16)]
interface IInterface
{
    float run();
}

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<float> gOutputBuffer;

//TEST_INPUT: set gObj = new StructuredBuffer<IInterface>[new FloatVal{1.0}, new Float4Val{{[2.0, 3.0, 4.0, 5.0]}}, new IntVal{6}, new Int4Val{[7,8,9,10]}];
RWStructuredBuffer<IInterface> gObj;

[numthreads(1, 1, 1)]
void computeMain(uint3       dispatchThreadID : SV_DispatchThreadID)
{
    // Test unpacking.
    float result = 0.0;
    for (int i = 0; i < 4; i++)
    {
        IInterface v0 = gObj.Load(i);
        result += v0.run();
    }

    // Test packing.
    IInterface v[2];
    Float4Val cval;
    cval.val.val = float4(1,2,3,4);
    v[0] = cval;

    Int4Val ival;
    ival.val = int4(2,3,4,5);
    v[1] = ival;

    for (int i = 0; i < 2; i++)
    {
        result += v[i].run();
    }
    gOutputBuffer[0] = result;
}

// Type must be marked `public` to ensure it is visible in the generated DLL.
export struct FloatVal : IInterface
{
    float val;
    float run()
    {
        return val;
    }
};
interface ISomething{void g();}
struct Float4Struct : ISomething { float4 val; void g() {} }
export struct Float4Val : IInterface
{
    Float4Struct val;
    float run()
    {
        return val.val.x;
    }
};
export struct IntVal : IInterface
{
    int val;
    float run()
    {
        return val;
    }
};
export struct Int4Val : IInterface
{
    int4 val;
    float run()
    {
        return val.x;
    }
};
