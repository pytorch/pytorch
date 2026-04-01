// static-method.slang

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

struct S
{
    static void doThing(in out int x, int y)
    {
        x += y;
    }
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

int test(int t)
{
    S::doThing(t, 0x10);

    (S::doThing(t, 0x200));

    (S::doThing)(t, 0x4000);

    return t;
}

[numthreads(4)]
void computeMain(int3 tid : SV_DispatchThreadID)
{
    int val = tid.x;
    val = test(val);
    outputBuffer[tid.x] = val;
}
