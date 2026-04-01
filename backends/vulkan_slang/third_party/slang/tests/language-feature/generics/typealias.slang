//TEST(compute):COMPARE_COMPUTE: -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj -output-using-type
struct MyGenType<T : IArithmetic, let n : int>
{
    T value;
    T compute() { return value * T(n); }
}

typealias IntMyGenType<let n : int> = MyGenType<int, n>;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    IntMyGenType<2> obj;
    obj.value = 2;
    outputBuffer[tid] = obj.compute();
}
