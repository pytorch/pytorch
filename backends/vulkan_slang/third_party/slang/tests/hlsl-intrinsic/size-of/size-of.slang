
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer

RWStructuredBuffer<int> outputBuffer;

enum Enum : int8_t
{
};

struct Hey
{
    double b;
    Enum e;
};

struct Thing : Hey
{
    int a;
    float b;
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int idx = asint(dispatchThreadID.x);

    int a = sizeof(idx);
    int b = sizeof(int);
    int c = sizeof(1 + 1);
    int d = sizeof(Thing);

    float g[sizeof(a)];

    int f[sizeof(Thing) + sizeof(float2) + sizeof(g)]; 

    int size;

    switch (idx)
    {   
        default:
        case 0:     size = sizeof(float); break;
        case 1:     size = sizeof(float3); break;
        case 2:     size = sizeof(matrix<uint, 3, 2>); break;
        case 3:     size = sizeof(idx); break;
    }

    outputBuffer[idx] = size + a + b + c + d + sizeof(f);
}
