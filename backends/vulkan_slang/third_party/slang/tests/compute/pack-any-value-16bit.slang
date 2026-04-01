// Test anyvalue packing of 16bit types.

//TEST_DISABLED(compute):COMPARE_COMPUTE_EX:-slang -compute -cuda -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -render-feature int16,half
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_2 -use-dxil -output-using-type

[anyValueSize(32)]
interface IInterface
{
    float run();
}

struct Val : IInterface
{
    int16_t v0;
    float f0;
    uint16_t v1;
    float16_t v2;
    half v3;
    uint16_t v4;
    float run()
    {
        return v0 + f0 + v1 + v2 + v3 + v4;
    }
};

struct UserDefinedPackedType
{
    uint4 values[2];
};

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<float> gOutputBuffer;

//TEST_INPUT: type_conformance Val:IInterface = 11

[numthreads(1, 1, 1)]
void computeMain(uint3       dispatchThreadID : SV_DispatchThreadID)
{
    uint half_4_0 = 0x4400; // 4.0f
    uint half_5_0 = 0x4500; // 5.0f

    UserDefinedPackedType objStorage;
    objStorage.values[0] = uint4(1, asuint(2.0), (3U | (half_4_0<<16)), (half_5_0 | (6<<16)));
    objStorage.values[1] = 0;

    IInterface dynamicObj = createDynamicObject<IInterface, UserDefinedPackedType>(11, objStorage);
    float result = dynamicObj.run();
    gOutputBuffer[0] = result;

    Val v;
    v.v0 = 1;
    v.f0 = 2;
    v.v1 = 3;
    v.v2 = half(4);
    v.v3 = half(5);
    v.v4 = 6;
    IInterface dynamicObj1 = createDynamicObject<IInterface, Val>(11, v);;
    gOutputBuffer[1] = dynamicObj1.run();

    var packed = reinterpret<UserDefinedPackedType, Val>(v);
    var unpacked = reinterpret<Val, UserDefinedPackedType>(packed);
    gOutputBuffer[2] = unpacked.run();
}
