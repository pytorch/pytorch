// gl-33.slang
//TEST(compute):COMPARE_COMPUTE: -shaderobj

// Test for GitLab issue # 33

import gl_33_ext;

typedef A B;

int test(int val)
{
    B b = { val };
    return b.next();
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name gBuffer
RWStructuredBuffer<int> gBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;

    int val = int(tid);
    val = test(val);

    gBuffer[tid] = val;
}
