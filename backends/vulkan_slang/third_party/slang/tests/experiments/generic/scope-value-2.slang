//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Test here is to try and associate a value with a type

Doesn't work 
D:\local\Temp\a3393a41-6d61-4049-816b-4edcc04f47fb.slang(24): error 30027: 'kE' is not a member of 'overload group'.
    let e = A::Type::kE;
    
The only change here was to make A implement IHasType    
*/

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

enum class Enum
{
    A, B
};

interface IHasType
{
    associatedtype Type;
};

struct A : IHasType
{
    typedef A Type;
    static const Enum kE = Enum::A;
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;
    let e = A::Type::kE;    
    outputBuffer[index] = int(e);
}
