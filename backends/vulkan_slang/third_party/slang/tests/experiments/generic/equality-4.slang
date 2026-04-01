//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test for equality around interface types

Here again trying to apply equality *outside* of the types (MyStruct) definition.  

Doesn't work:

.slang(24): error 30019: expected an expression of type 'Type', got 'T'
    return T::isEqual(a, b);
    
Note! This may be somewhat of a silly example for equality. We could get what we want here by just
implementing 'isEqual(MyStruct a, MyStruct b)` as a free function and use overloading.     
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct MyStruct
{
    int a = 10;
};

interface IEquality
{
    associatedtype Type;
    static bool isEqual(Type a, Type b);
}

extension MyStruct : IEquality
{
    // Do I need this? Is the type This?
    typedef MyStruct Type;
    static bool isEqual(Type a, Type b) { return a.a == b.a; }
};

__generic<T : IEquality>
bool isEqual(T a, T b)
{
    return T::isEqual(a, b);
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


