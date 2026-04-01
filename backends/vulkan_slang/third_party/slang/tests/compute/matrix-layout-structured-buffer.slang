// matrix-layout-structured-buffer.slang

// This test confirms that we apply the matrix layout
// mode requested by the user, even in the case of structured
// buffers of matrices, where fxc/dxc do *not* respect
// the matrix layout mode by default.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -xslang -matrix-layout-row-major -shaderobj -dx11
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -xslang -matrix-layout-column-major -shaderobj -dx11
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -xslang -matrix-layout-row-major -shaderobj -dx12
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -xslang -matrix-layout-column-major -shaderobj -dx12


//TEST_INPUT:ubuffer(data=[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23], stride=48):name=gMatrices
RWStructuredBuffer<int3x4> gMatrices;

int test(int val)
{
    int N = 256;

    int tmp = 0;

    tmp = tmp*N + gMatrices[val%2][(val  )%3][(val  )%4];
    tmp = tmp*N + gMatrices[val%2][(val+1)%3][(val  )%4];
    tmp = tmp*N + gMatrices[val%2][(val  )%3][(val+1)%4];
    tmp = tmp*N + val;
    tmp = tmp + 0x80000000;

    return tmp;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=buffer
RWStructuredBuffer<int> buffer;

[numthreads(12, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;

    int val = tid;
    val = test(val);

    buffer[tid] = val;
}
