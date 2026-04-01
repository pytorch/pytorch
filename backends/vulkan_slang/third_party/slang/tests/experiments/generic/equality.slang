//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test for equality around interface types

Doesn't work because

.slang(8): error 38100: type 'A' does not provide required interface member 'isEqual'
struct A : IEquality
           ^~~~~~~~~
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IEquality
{
    bool isEqual(IEquality rhs);
};

struct A : IEquality
{
    bool isEqual(IEquality rhs)
    {
        // Hmm. How can I cant query rhs, and then cast, so not clear how to implement
        return true;
    }
    int value;
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;

    A a = { 1 };
    A b = { 2 };
    
    bool isEqual = a.isEqual(b);

	outputBuffer[index] = 1 + int(isEqual);
}


