//TEST(compute):COMPARE_COMPUTE: -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj -output-using-type
T simpleTest<T : IArithmetic>(T v0, T v1)
{
    if (v0 > T(0))
    {
        return v0 + v1;
    }
    else
    {
        return -v0 * v1;
    }
}

interface IMyInterface : IArithmetic
{
    int myMethod();
}

extension float : IMyInterface
{
    int myMethod() { return 4; }
}

extension double : IMyInterface
{
    int myMethod() { return 8; }
}

int genTest<T : IMyInterface>(T v0, T v1)
{
    vector<T, 2> v = vector<T, 2>(v0, v1);
    return (v.x + v.y).myMethod();
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    outputBuffer[tid] = int(simpleTest<float>(1.0f, 2.0f)) + genTest<float>(1.0f, 2.0f);
}
