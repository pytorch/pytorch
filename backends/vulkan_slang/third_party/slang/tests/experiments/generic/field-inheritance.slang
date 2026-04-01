//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* The test here uses struct inheritance but has a field with the same name in the derived type. 

This error is for the MyStruct2 version. 

.slang(20): error 39999: no overload for '==' applicable to arguments of type (overload group, overload group)
    return a.a == b.a;

No error/warning is given for the shadowing of MyStruct.a.
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct MyStruct
{
    int a = 10;
};

struct MyStruct2 : MyStruct
{
    int a = 10;
};

bool isEqual(MyStruct a, MyStruct b)
{
    return a.a == b.a;
}

bool isEqual(MyStruct2 a, MyStruct2 b)
{
    return a.a == b.a;
}
             
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;

    MyStruct a = { 1 };
    MyStruct b = { 2 };
    
    bool res = isEqual(a, b);

	outputBuffer[index] = 1 + int(res);
}

