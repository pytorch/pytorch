//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj

//TEST_INPUT:ubuffer(data=[0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint32_t> outputBuffer;

[UnscopedEnum]
enum unscopedEnum : uint32_t
{
    kEnumVal1 = 5,
    kEnumVal2 = 6,
};

struct structWithEnumVal
{
    unscopedEnum val;
};

void takeInToWriteFruit(unscopedEnum inData, structWithEnumVal inData2)
{
    outputBuffer[0] = inData;
    outputBuffer[1] = inData2.val;
}

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 5
    // CHECK: 6
    structWithEnumVal enumVal;
    enumVal.val = kEnumVal2;
    takeInToWriteFruit(kEnumVal1, enumVal);
}