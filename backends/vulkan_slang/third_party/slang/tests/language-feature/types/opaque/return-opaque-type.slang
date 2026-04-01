// return-opaque-type.slang

// Test that a function/method can return a value of an opaque type.

//TEST(compute):COMPARE_COMPUTE:

struct Stuff
{
    RWStructuredBuffer<int> things;

    RWStructuredBuffer<int> getThings() { return things; }
}

//TEST_INPUT:set gStuff = new Stuff { ubuffer(data=[16 17 18 19], stride=4) }
ConstantBuffer<Stuff> gStuff;

int test(int val)
{
    return gStuff.getThings()[val];
}

//TEST_INPUT:set gOutput = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<int> gOutput;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal);
    gOutput[tid] = outVal;
}
