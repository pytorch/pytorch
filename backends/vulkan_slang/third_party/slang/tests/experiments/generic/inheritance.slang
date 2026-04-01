//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Tests around inheritance.

Works.

Docs state inheritance isn't available (although states it might in the future).

https://github.com/shader-slang/slang/blob/master/docs/language-reference/07-declarations.md

> Currently only interface types may be named in the inheritance clause of a structure type. When a structure type declares that it inherits from an interface, the programmer asserts that the structure type implements the required members of the interface.

*/

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct A<T>
{
    T a;
};

struct B : A<float>
{
    int b;
};


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;

    B b = { 1, 2 + index };
    
	outputBuffer[index] = b.b * b.a;
}