// interface-conjunction.slang

// Test that we can compose interfaces with `&`

//TEST(compute):COMPARE_COMPUTE: -shaderobj

interface IFirst
{
    int getFirst();
}

interface ISecond
{
    int getSecond();
}

struct Pair : IFirst, ISecond
{
    int first;
    int second;

    int getFirst() { return first; }
    int getSecond() { return second; }
}

typealias IBoth = IFirst & ISecond;

int add<T : IBoth>(T input)
{
    return 2*input.getFirst() + input.getSecond();
}

int test(int value)
{
    Pair p = { value, (value+1) * 256 };
    return add(p);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
