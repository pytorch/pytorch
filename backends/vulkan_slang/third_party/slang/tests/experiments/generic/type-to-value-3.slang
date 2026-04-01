//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Test here is to try and associate a value with a type

This does work. 

Note that it *requires* the getType<T> function as if we try to call say int.getType(), we get an error around
swizzling.
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
    static Type getType();
};

extension int : IHasType
{
    static Type getType() { return Type::Int; }
};

extension float : IHasType
{
    static Type getType() { return Type::Float; }
};

Type getType<T : IHasType>() { return T::getType(); }

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;
    
    let type = getType<float>();
    
    outputBuffer[dispatchThreadID.x] = int(type);
}