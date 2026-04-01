//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

// Test that we can use `where` clause to constrain an associatedtype.

interface IBar
{
    associatedtype TB;
    TB get();
}

interface IFoo
{
    associatedtype TA : IBar where TA.TB == int;

    TA getVal();
}

struct BarImpl : IBar
{
    typealias TB = int;
    int x;
    int get() { return x; }
}

struct FooImpl : IFoo
{
    typealias TA = BarImpl;
    TA getVal() { TA a; a.x = 1; return a; }
}

int helper<T : IFoo>(T foo)
{
    // foo.getVal().get() has type `T.TA.TB`,
    // because there is a type equality constraint defined on
    // `IFoo.TA` such that `IFoo::TA.TB == int`, we should be able
    // to conclude that `T.TA.TB` is `int` and the `return` here
    // should type check.
    return foo.getVal().get();
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    FooImpl foo;

    outputBuffer[0] = helper(foo);
    // CHECK: 1.0
}
