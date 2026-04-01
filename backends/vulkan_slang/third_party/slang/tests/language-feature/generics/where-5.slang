//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

// Test that we can use `where` clause to constrain generic type dependant on generic parameters.

T process<T, int N>(vector<T, N> v) 
    where T : IFloat
    where vector<T, N> : IArithmetic
{
    return (v.mul(T(2)))[0];
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    float a = 1.0;
    outputBuffer[0] = process(a);
    // CHECK: 2.0
}
