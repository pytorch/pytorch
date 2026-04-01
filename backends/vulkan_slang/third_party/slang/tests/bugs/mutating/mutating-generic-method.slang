//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute,vulkan):COMPARE_COMPUTE_EX:-vk -slang -compute -shaderobj

// Confirm that a generic method marked `[mutating]`
// produces an `inout` parameter for `this`.

interface IGenerator
{
    [mutating] int generate();
}

struct SimpleGenerator : IGenerator
{
    int state = 0;

    [mutating] int generate() { return state++; }
}


struct Reducer
{
    int state = 0;
    [mutating] void update<S : IGenerator>(inout S g)
    {
        state = (state << 8) | state;
        state = state + g.generate();
    }
}

int test(int val)
{
    SimpleGenerator g = { 0 };
    Reducer r = { val+1 };

    r.update(g);
    r.update(g);

    return r.state;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out, name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int val = test(tid);
    outputBuffer[tid] = val;
}
