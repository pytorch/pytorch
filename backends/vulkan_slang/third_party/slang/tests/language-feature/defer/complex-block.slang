//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int i = 0;
    int j = 0;
    defer {
        if (j == 2)
        {
            outputBuffer[i++] = j;
        }
        else
        {
            outputBuffer[i++] = 2048;
        }
    }
    defer {
        if (j == 1)
        {
            outputBuffer[i++] = j;
        }
        else
        {
            outputBuffer[i++] = 1024;
        }
        j++;
    }
    outputBuffer[i++] = j;
    j++;
}
