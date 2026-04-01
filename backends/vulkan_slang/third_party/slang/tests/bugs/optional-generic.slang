//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-slang -compute
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-slang -compute -vk


Optional<T> genFunc<T : IArithmetic>(T v)
{
    if (v is int)
        return v;
    return none;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name buffer

RWStructuredBuffer<int> buffer;

[numthreads(1,1,1)]
void computeMain()
{
    // BUF: 2
    buffer[0] = genFunc(2).value;
}

