// Test anyvalue packing of 8bit types.

//TEST_DISABLED(compute):COMPARE_COMPUTE_EX:-slang -compute -cuda -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -render-feature int16
//TEST_DISABLED(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_2 -use-dxil -output-using-type

[anyValueSize(20)]
interface IInterface
{
    float run();
}

struct Val : IInterface
{
    int8_t v0;
    uint8_t v1;
    float f0;
    uint16_t v2;
    uint8_t v3;
    uint32_t v4;
    uint8_t v5;
    float run()
    {
        return v0 + v1 + f0 + v2 + v3 + v4 + v5;
    }
};

struct UserDefinedPackedType
{
    uint values[5];
};

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<float> gOutputBuffer;

//TEST_INPUT: type_conformance Val:IInterface = 11

[numthreads(1, 1, 1)]
void computeMain(uint3       dispatchThreadID : SV_DispatchThreadID)
{
    UserDefinedPackedType objStorage;
    objStorage.values[0] = 0xA5A50201;
    objStorage.values[1] = asuint(3.0f);
    objStorage.values[2] = 4u|(0xA505u<<16u);
    objStorage.values[3] = 6;
    objStorage.values[4] = 7;

    IInterface dynamicObj = createDynamicObject<IInterface, UserDefinedPackedType>(11, objStorage);
    float result = dynamicObj.run();
    gOutputBuffer[0] = result;

    Val v;
    v.v0 = 1;
    v.v1 = 2;
    v.f0 = 3;
    v.v2 = 4;
    v.v3 = 5;
    v.v4 = 6;
    v.v5 = 7;

    IInterface dynamicObj1 = createDynamicObject<IInterface, Val>(11, v);;
    gOutputBuffer[1] = dynamicObj1.run();

    var packed = reinterpret<UserDefinedPackedType, Val>(v);
    var unpacked = reinterpret<Val, UserDefinedPackedType>(packed);
    gOutputBuffer[2] = unpacked.run();
}

