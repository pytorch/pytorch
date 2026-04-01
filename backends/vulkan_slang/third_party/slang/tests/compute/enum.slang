// enum.slang
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj

// Confirm that basic `enum` declarations are supported.

enum Color
{
    Red,
    Green = (1 << 1),
    Blue,
}


int test(int val)
{
    Color c = Color.Red;

    if(val > 1)
    {
        c = Color.Green;
    }

    if(c == Color.Red)
    {
        if((val & 1) != 0)
        {
            c = Color.Blue;
        }
    }

    switch(c)
    {
    case Color.Red:
        val = 1;
        break;

    case Color.Green:
        val = 2;
        break;

    case Color.Blue:
        val = 3;
        break;

    default:
        val = -1;
        break;
    }

    return (val << 4) + int(c);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;

    int val = int(tid);
    val = test(val);

    outputBuffer[tid] = val;
}
