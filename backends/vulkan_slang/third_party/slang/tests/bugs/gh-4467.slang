//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK): -d3d12 -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHK): -vk -compute -shaderobj -output-using-type


// Test that we can synthesize a [mutating] interface requirement from a nonmutating implementation,
// and the interface requirement signature contains an output interface-typed parameter.

//TEST_INPUT: ubuffer(data=[0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface IFoo
{
    int getVal();
};

interface IBar
{
    [mutating] void method(out IFoo o);
};

struct FooImpl : IFoo
{
    int x;
    int getVal() { return x; }
}

struct BarImpl : IBar
{
    void method(out IFoo o)
    {
        o = FooImpl(1);
    }
};

[numthreads(1,1,1)]
void computeMain()
{
    BarImpl bar;
    IFoo foo;
    bar.method(foo);
    // CHK: 1
    outputBuffer[0] = foo.getVal();
}