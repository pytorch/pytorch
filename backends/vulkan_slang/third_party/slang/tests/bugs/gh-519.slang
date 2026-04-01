//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj

// This bug involved incorrect computation of the successor
// blocks for a `switch`, which led to incorrect SSA form.

int test(int val)
{
    int tmp = 0;
    switch(val)
    {
    case 0:
        tmp = 1;
        break;

    case 1:
        tmp = 2;
        break;

    case 2:
        tmp = 3;
        break;

    default:
        tmp = val + 1;
        break;
    }
    return tmp;
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