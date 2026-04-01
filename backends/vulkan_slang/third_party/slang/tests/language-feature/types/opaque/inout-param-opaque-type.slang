// inout-param-opaque-type.slang

// Test that a function/method can have an `out` parameter of opaque type

//TEST(compute):COMPARE_COMPUTE:
//TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl

//TEST_INPUT:set gX = ubuffer(data=[16 17 18 19], stride=4)
RWStructuredBuffer<int> gX;

//TEST_INPUT:set gY = ubuffer(data=[3 6 9 12], stride=4)
RWStructuredBuffer<int> gY;

void swap(
    inout RWStructuredBuffer<int> a,
    inout RWStructuredBuffer<int> b)
{
    RWStructuredBuffer<int> t = a;
    a = b;
    b = t;
}

int test(int val)
{
    RWStructuredBuffer<int> f = gX;
    RWStructuredBuffer<int> g = gY;

    swap(f, g);

    return f[val] * 256 + g[val];
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
