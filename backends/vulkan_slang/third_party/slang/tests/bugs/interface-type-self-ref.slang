//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface IFoo
{
    int compute(out IFoo param);
}

struct Impl : IFoo
{
    int compute(out IFoo param)
    {
        param = this;
        return 0;
    }
}

int f(IFoo p)
{
    IFoo r;
    return p.compute(r);
}

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;
    Impl impl;
    outputBuffer[index] = f(impl);
}
