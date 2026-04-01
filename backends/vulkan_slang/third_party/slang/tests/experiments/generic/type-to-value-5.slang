//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Test here is to try and associate a value with a type

Here we try to associate by having a value defined on the type, and then set the type
on the associated type.

Doesn't work because ...

.slang(29): error 30027: 'Type' is not a member of '.This'.
    return e::Type::kE;
    
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

// This is a little perverse, because I'm defining as an associated type that is
// the same as itself
struct A : IHasType
{
    typedef A Type;
    static const Enum kE = Enum::A;
};

struct B : IHasType
{
    typedef B Type;
    static const Enum kE = Enum::B;
};

Enum getType(IHasType e)
{
    return e::Type::kE;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;
   
    B b;
    let e = getType(b);
    
    outputBuffer[dispatchThreadID.x] = int(e);
}