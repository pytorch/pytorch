//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test to use generics to treat an enum as a set. 
Here we simplify previous example, by not allowing the backing type be 
a specialization parameter (and just use int).

Moreover here E is constrained to __EnumType.

Works
*/

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

enum Enum
{
    A = 0x1,
    B = 0x2, 
    C = 0x4,
};

__generic<E : __EnumType>
struct Flags
{
    [mutating] void set(E e) { value |= (int)e; }
    bool isSet(E e) { return (((int)e) & value) != 0; }
    int value = 0;
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;
    
    Flags<Enum> flags;

    if (index & 1)
    {
        flags.set(Enum::A);
    }
    bool isASet = flags.isSet(Enum::A);
   
	outputBuffer[index] = isASet ? 2 : 1;
}

