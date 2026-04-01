//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

// Test that we can use `where` clause to constrain the type of a type pack.

interface IFoo
{
    associatedtype TA;
}

struct FooImpl : IFoo
{
    typealias TA = int;
}

void add(inout int a, int b)
{
    a += b;
}

int process<each T>(T v) where T == int
{
    int result = 0;
    expand add(result, each v);
    return result;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    outputBuffer[0] = process(1,2,3);
    // CHECK: 6.0
}
