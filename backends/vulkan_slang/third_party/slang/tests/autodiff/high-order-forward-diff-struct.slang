//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct MyStruct : IDifferentiable
{
    float x;
    uint i;

    [ForwardDifferentiable]
    __init(float x, uint i)
    {
        this.x = x;
        this.i = i;
    }
};

[ForwardDifferentiable]
float mySqr(MyStruct s)
{
    if (s.i >= 1)
        return s.x * s.x;
    else
        return 0.f;
}

[ForwardDifferentiable]
float f(float x)
{
    MyStruct ms = MyStruct(x * x, 1);
    return mySqr(ms);
}

[ForwardDifferentiable]
float df(float x)
{
    return __fwd_diff(f)(DifferentialPair<float>(x, 1.0)).d;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Given f(x) = x^4,
    // f''(x) = 12 * x^2
    // Expect f''(4) = 192
    outputBuffer[0] = __fwd_diff(df)(DifferentialPair<float>(4.0, 1.0)).d;
}

