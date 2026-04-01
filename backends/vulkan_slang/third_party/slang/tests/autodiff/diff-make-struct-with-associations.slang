//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;


struct Foo : IDifferentiable
{
    typealias Differential = Foo;
    float x[3];
};

[Differentiable]
Foo getFoo(float x)
{
    return { { x, x, x } };
}

[Differentiable]
float foobar(float x)
{
    int i = 3 * int(floor(x));
    Foo foo = getFoo(x);
    return foo.x[i] * foo.x[i];
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        float a = 0.5;
        var d = fwd_diff(foobar)(diffPair(a, 1.0)).d;
        outputBuffer[0] = d;
    }
}