// generic-func-param-default-arg.slang

// Test that generic functions can have default argument values on their parameters.

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST_DISABLED:SIMPLE:-target hlsl -entry computeMain -dump-ir


interface IValue
{
    __init();

    This plusA(This other);
    This plusB(int other);
}

T sum<T : IValue>(T value, T other = T(), int extra = 0)
{
    return value.plusA(other).plusB(extra);
}

struct Simple : IValue
{
    int val;

    __init() { val = 0; }
    __init(int val) { this.val = val; }

    Simple plusA(Simple other) { return Simple(val + other.val); }
    Simple plusB(int other) { return Simple(val + other); }
}

int test(int val)
{
    let s = Simple(val);
    return sum<Simple>(s).val + 16*sum(s).val;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=gBuffer
RWStructuredBuffer<int> gBuffer;

[shader("compute")]
[numthreads(4)]
void computeMain(int tid : SV_DispatchThreadID)
{
    gBuffer[tid] = test(tid);
}
