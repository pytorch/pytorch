//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute

interface IGetter
{
    int get(int x);
}

//TEST_INPUT:set gOutputBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<int> gOutputBuffer;

//TEST_INPUT: globalSpecializationArg MyGetter
//TEST_INPUT: set gGetter = new MyGetter{}
uniform IGetter gGetter;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    gOutputBuffer[dispatchThreadID.x] = gGetter.get(dispatchThreadID.x);
}

struct MyGetter : IGetter
{
    int get(int x)
    {
        return x + 1;
    }
}
