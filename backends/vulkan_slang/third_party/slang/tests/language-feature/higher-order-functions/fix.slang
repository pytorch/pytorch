//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX():-vk -compute -shaderobj -output-using-type
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX():-slang -compute -shaderobj -output-using-type
// Disabled due to https://github.com/shader-slang/slang/issues/2198

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

func fix(float x) -> float
{
    let epsilon = 0.01;
    let x_ = go(x);
    if(abs(x_ - x) < epsilon)
        return x_;
    return fix(x_);
}

func go(x : float) -> float
{
    return (x + 121 / x) / 2;
}

[numthreads(4, 1, 1)]
void computeMain(uint tig : SV_GroupIndex)
{
    outputBuffer[tig] = uint(fix(11));
}
