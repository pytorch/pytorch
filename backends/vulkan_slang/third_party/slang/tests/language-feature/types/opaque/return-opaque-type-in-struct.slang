// return-opaque-type-in-struct.slang

// Test that a function/method can return a value of
// aggregate type that includes an opaque type

//TEST(compute):COMPARE_COMPUTE:
//TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl

struct Things
{
    int first;
    RWStructuredBuffer<int> rest;
}

//TEST_INPUT:set gThings = new Things { 1, ubuffer(data=[2 3 4 5], stride=4) }
ConstantBuffer<Things> gThings;

Things getThings()
{
    return gThings;
}

int test(int val)
{
    let things = getThings();
    return things.first * (16 << val) + things.rest[val];
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
