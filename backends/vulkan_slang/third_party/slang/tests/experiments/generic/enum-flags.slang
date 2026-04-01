//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test to use generics to treat an enum as a set.

.slang(15): error 30060: expected a type, got a '__BuiltinIntegerType'
    B value = 0;
    
.slang(23): error 30019: expected an expression of type 'int', got 'typeof(uint8_t)'
    Flags<Enum, uint8_t> flags;
                ^~~~~~~    
*/

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

enum Enum
{
    A = 0x1,
    B = 0x2, 
    C = 0x4,
};

__generic<E, let B : __BuiltinIntegerType>
struct Flags
{
    [mutating] void set(E e) { value |= (B)e; }
    void isSet(E e) { return (((B)e) & value) != 0; }
    B value = 0;
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;
    
    Flags<Enum, uint8_t> flags;

    if (index & 1)
    {
        flags.set(Enum::A);
    }
    bool isASet = flags.isSet(Enum::A);
   
	outputBuffer[index] = isASet ? 2 : 1;
}

