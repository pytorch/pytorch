//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test for equality around interface types

This is an attempt to get the *outside* impl of equality to work. The simple case does, but just to throw a spanner in the works, lets mix in some inheritance. 

An issue here (perhaps) is that this will compile - the isEqual implementation will just slice and probably not do what the implementer expected.
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct MyStruct
{
    int a = 10;
};

struct MyStruct2 : MyStruct
{
    int b;
};

bool isEqual(MyStruct a, MyStruct b)
{
    return a.a == b.a;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;

    // I *suppose* it could be argued that this is problematic - that the b field is uninitialized 
    // but there is no warning or an error.
   
    MyStruct2 a = { 1 };
    MyStruct2 b = { 2 };
    
    bool res = isEqual(a, b);

	outputBuffer[index] = 1 + int(res);
}