//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -vk
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj 
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -cpu

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<uint> outputBuffer;


void silly<T : __BuiltinIntegerType>(T a, inout T b)
{
    b *= a;
}

void silly2<T : __BuiltinIntegerType>(T a, out T b)
{
    b = a;
}


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int v = 0;
    uint u = 0;

    int idx = int(dispatchThreadID.x);
    uint uidx = uint(idx);

    v += uidx;
    u += idx;

    silly(u, v);
    silly(v, u);

    // Check that detects issues with undefined variables.

    int undef1, undef2;          // undefined
    
    // Downstream compilers detect this... 
    //silly(u, undef1);
    silly2(u, undef2);
    
    outputBuffer[idx] = u + v + undef2;
}
