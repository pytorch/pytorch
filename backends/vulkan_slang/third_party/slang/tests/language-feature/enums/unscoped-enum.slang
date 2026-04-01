//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj

[UnscopedEnum]
enum class Fruit
{
    Orange, Apple
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 1
    outputBuffer[0] = (int)Apple;
}