//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Test here is to try and associate a value with a type

This works.
*/

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

enum class Enum
{
    A,
    B,
};

struct A
{
    static const Enum kE = Enum::A;
};

struct B
{
    static const Enum kE = Enum::B;
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;
    
    outputBuffer[dispatchThreadID.x] = int(B::kE);
}