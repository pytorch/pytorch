//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct NonDiff
{
    float a;
}

void getCode(uint2 x, out Array<float, 2> v)
{
    for (int i = 0; i < 8; i++)
        v[i] = (float)i;
}

[BackwardDerivativeOf(getCode)]
void getCode_bwd(uint2 x, Array<float, 2> dout)
{
    outputBuffer[0] = dout[0];
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    float a = 10.0;
    float inArray[2] = { 1, 2 };
    __bwd_diff(getCode)(uint2(1,2), inArray);
}