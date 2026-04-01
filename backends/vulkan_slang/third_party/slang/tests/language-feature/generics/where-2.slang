//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

// Test that we can use `where` clause to constrain associatedtype of generic type parameters.

interface IFoo
{
    associatedtype TA;
}

struct FooImpl : IFoo
{
    typealias TA = int;
}

__generic<typename T>
T.TA process(T v)
    where T : IFoo
    where T.TA == int
{
    return 1;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    FooImpl fooImpl;
    outputBuffer[0] = process(fooImpl);
    // CHECK: 1.0
}
