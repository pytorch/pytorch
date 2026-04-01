// We can't test on VK or metal, as currently we don't support integer matrix types
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -vk
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj 
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -cpu
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl
// Not supported in WGSL: Integer matrices
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-wgpu

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<uint> outputBuffer;

void silly<T : __BuiltinIntegerType, let ROWS : int, let COLS : int>(matrix<T, ROWS, COLS> a, inout matrix<T, ROWS, COLS> b)
{
    b *= a;
}

void silly2<T : __BuiltinIntegerType, let ROWS : int, let COLS : int>(matrix<T, ROWS, COLS> a, out matrix<T, ROWS, COLS> b)
{
    b = a;
}


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    matrix<int, 2, 2> v = { 1, 2, 3, 4};
    matrix<uint, 2, 2> u = { 5, 6, 7, 8};

    int idx = int(dispatchThreadID.x);
    
    matrix<uint, 2, 2> uidx = { idx, idx * idx, idx + idx, idx + 2 };

    v += uidx;
    u += idx;

    silly(u, v);
    silly(v, u);

    // Check that detects issues with undefined variables.

    matrix<int, 2, 2> undef1, undef2;          // undefined
    // Downstream compilers detect this
    //silly(u, undef1);
    silly2(u, undef2);
    
    matrix<uint, 2, 2> added = (u + v + undef2);
    uint2 added2 = added[0] + added[1];

    outputBuffer[idx] = added2.x + added2.y;
}
