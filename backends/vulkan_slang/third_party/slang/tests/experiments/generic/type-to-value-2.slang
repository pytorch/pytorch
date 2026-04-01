//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Test here is to try and associate a value with a type

Lets see if we can do something with a built in type

Doesn't work because 

error 30052: invalid swizzle pattern 'getType' on type 'int'
    let type = i.getType();

*/

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

enum class Type
{
    Unknown,
    Float,
    Int,
};

interface IHasType
{
    Type getType()
};

extension int : IHasType
{
    Type getType() { return Type::Int; }
};

extension float : IHasType
{
    Type getType() { return Type::Float; }
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;

    int i;
    let type = i.getType();
    
    outputBuffer[dispatchThreadID.x] = int(type);
}