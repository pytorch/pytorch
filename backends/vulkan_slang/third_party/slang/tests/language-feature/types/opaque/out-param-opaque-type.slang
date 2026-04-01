// out-opaque-type.slang

// Test that a function/method can have an `out` parameter of opaque type

//TEST(compute):COMPARE_COMPUTE:
//TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl

//TEST_INPUT:set gThings = ubuffer(data=[16 17 18 19], stride=4)
RWStructuredBuffer<int> gThings;


void getThings(out RWStructuredBuffer<int> things)
{
    things = gThings;
}

int test(int val)
{
    RWStructuredBuffer<int> t;
    getThings(t);
    return t[val];
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
