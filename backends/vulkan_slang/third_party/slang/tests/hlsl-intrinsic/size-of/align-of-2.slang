
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer

RWStructuredBuffer<int> outputBuffer;

// Check Enum without base type

enum Enum 
{
    A = 10,
    B = 20,
};

enum Enum8 : uint8_t
{
    A = 20,
    B = 19,
};

// Check type aliases

typedef Enum EnumTypeDef;
typealias EnumTypeAlias = Enum;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int idx = asint(dispatchThreadID.x);

    int size = 0;
    switch (idx)
    {   
        default:
        case 0:     size = alignof(Enum); break;
        case 1:     size = alignof(EnumTypeDef); break;
        case 2:     size = alignof(EnumTypeAlias); break;
        case 3:     size = alignof(idx); break;
    }

    outputBuffer[idx] = size;
}
