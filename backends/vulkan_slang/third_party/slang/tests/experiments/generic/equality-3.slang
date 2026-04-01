//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test for equality around interface types

This style is discussed in the documentation here (on This type):

https://github.com/shader-slang/slang/blob/master/docs/user-guide/04-interfaces-generics.md

Still has limitiation that it only works for an an implementation of an interface.  
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IEquality
{
    bool isEqual(This other);
}

struct A : IEquality
{
    bool isEqual(This other)
    {
        return value == other.value;
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


