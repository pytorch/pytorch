//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -vk
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj 
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -cpu


//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<uint> outputBuffer;


void silly<T : __BuiltinIntegerType, let N : int>(vector<T, N> a, inout vector<T, N> b)
{
    b *= a;
}

void silly2<T : __BuiltinIntegerType, let N : int>(vector<T, N> a, out vector<T, N> b)
{
    b = a;
}


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 v = int2(0, 0);
    uint2 u = uint2(0, 0);

    int idx = int(dispatchThreadID.x);
    
    uint2 uidx = uint2(idx, idx * idx);

    v += uidx;
    u += idx;

    silly(u, v);
    silly(v, u);

    // Check that detects issues with undefined variables.

    int2 undef1, undef2;          // undefined
    // Downstream compilers detect this
    //silly(u, undef1);
    silly2(u, undef2);
    
    uint2 added = (u + v + undef2);

    outputBuffer[idx] = added.x + added.y;
}
