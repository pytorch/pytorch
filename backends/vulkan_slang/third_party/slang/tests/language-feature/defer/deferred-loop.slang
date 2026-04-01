//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int i = 0;
    int j = 0;
    defer {
        defer outputBuffer[i++] = j*3+3;
        outputBuffer[i++] = 1;
        for (int k = 0; k < 6; ++k)
        {
            defer outputBuffer[i++] = k*3+2;

            if(k == j-4)
                continue;

            outputBuffer[i++] = k*3;

            if(k == j)
                break;

            outputBuffer[i++] = k*3+1;
        }
    }
    outputBuffer[i++] = j;
    j += 4;
}
