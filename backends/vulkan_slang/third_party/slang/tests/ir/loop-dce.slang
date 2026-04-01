//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -g0
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -g0

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

__target_intrinsic(hlsl, "@")
__target_intrinsic(glsl, "@")
__target_intrinsic(cpp, "@")
__target_intrinsic(cuda, "@")
[__readNone]
int produceSyntaxError() { return 0; }

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    int sum = 0;
    int array[100];
    // Next, this loop will be removed because there is no use of `array`.
    for (int i = 0; i < 100; i++)
    {
        // This loop must be removed, or we will fail downstream compilation.
        array[i] = i + produceSyntaxError();
    }

    // First, this loop will be removed because there is no use of `sum`.
    for (int i = 0; i < 100; i++)
    {
        // This loop must be removed, or we will fail downstream compilation.
        if (i < 50)
        {
            sum += array[i] + produceSyntaxError();
        }
        else
        {
            sum += i * 2 + produceSyntaxError();
        }
    }
    outputBuffer[0] = 1;
}
