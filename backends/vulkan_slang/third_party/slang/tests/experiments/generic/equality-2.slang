//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test for equality around interface types

This does work. 

It is here, because it seems like an awkward to do things, and only applys to struct like types.
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IEquality
{
    associatedtype Type;
    bool isEqual(Type rhs);
};

struct A : IEquality
{
    typedef A Type;
    bool isEqual(IEquality rhs)
    {
        return value == rhs.value;
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


