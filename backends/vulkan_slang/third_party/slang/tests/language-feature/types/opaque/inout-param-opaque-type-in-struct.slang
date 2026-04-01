// inout-param-opaque-type-in-struct.slang

// Test that a function/method can have an `out` parameter of
// aggregate type that includes an opaque type

//TEST(compute):COMPARE_COMPUTE:
//TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl

struct Things
{
    int first;
    RWStructuredBuffer<int> rest;
}

//TEST_INPUT:set C = new { {1, ubuffer(data=[2 3 4 5], stride=4)}, {6, ubuffer(data=[7 8 9 10], stride=4)} }
cbuffer C
{
    Things gX;
    Things gY;
}

void swap(
    inout Things a,
    inout Things b)
{
    Things t = a;
    a = b;
    b = t;
}

int eval(Things t, int val)
{
    return t.first*256 + t.rest[val];
}

int test(int val)
{
    Things f = gX;
    Things g = gY;

    swap(f, g);

    return (eval(f,val) << 16) + eval(g,val);
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
