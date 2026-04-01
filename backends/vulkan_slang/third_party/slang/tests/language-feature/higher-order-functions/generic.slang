//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;
[numthreads(1, 1, 1)]
void computeMain(uint tig : SV_GroupIndex)
{
    outputBuffer[tig] = zap();
}

func foo(f : functype (float) -> int) -> int
{
    return f(0);
}

int bap<T>(float)
{
    return 1;
}

int zap()
{
    // We should be able to specify which foo we want
    return foo(bap<bool>);
}
