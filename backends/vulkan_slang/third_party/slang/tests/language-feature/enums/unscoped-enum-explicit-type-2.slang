//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint32_t> outputBuffer;

[UnscopedEnum]
enum unscopedEnum1 : uint32_t
{
    ENUM1_A = 1,
    ENUM1_B = 2
};

[UnscopedEnum]
enum unscopedEnum2 : uint32_t
{
    ENUM2_A = 3,
    ENUM2_B = 4
};

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 1
    // CHECK: 2
    // CHECK: 3
    // CHECK: 4
    outputBuffer[0] = ENUM1_A;
    outputBuffer[1] = ENUM1_B;
    outputBuffer[2] = ENUM2_A;
    outputBuffer[3] = ENUM2_B;
}

