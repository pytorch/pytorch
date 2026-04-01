// scope.slang
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj

// Confirm that scoping on enums and types works 

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

enum Color
{
    Red,
    Green = 2,
    Blue,
}

struct Thing 
{
    int a; 
    struct Another
    {
        int b;
    }
};

int test(int val)
{
    Color c = Color.Red;

    Thing::Another another;    
    another.b = 20;

    if(val > 1)
    {
        c = Color.Green;
    }

    if(c == Color::Red)
    {
        if((val & 1) != 0)
        {
            c = Color::Blue;
        }
    }

    switch(c)
    {
    case Color::Red:
        val = 1;
        break;

    case Color::Green:
        val = 2;
        break;

    case Color::Blue:
        val = 3;
        break;

    default:
        val = -1;
        break;
    }

    return (val << 4) + int(c) + another.b - 20;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;

    int val = int(tid);
    val = test(val);

    outputBuffer[tid] = val;
}