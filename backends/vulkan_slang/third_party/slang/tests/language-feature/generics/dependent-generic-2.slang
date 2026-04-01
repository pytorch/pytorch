//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

// Test that we can infer a generic type parameter from the base type of a dependent generic argument.

interface IFoo : IDefaultInitializable
{
    int get();
}

struct Foo : IFoo
{
    int get()
    {
        return 1;
    }
}

interface IBar<T : IFoo>
{
    int getVal();
}

struct Bar<T : IFoo> : IBar<T>
{
    int getVal()
    {
        T t = T();
        return t.get();
    }
}

int test<T:IFoo, B : IBar<T>>(B b)
{
    return b.getVal();
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    Bar<Foo> obj2;
    let result = test(obj2);

    // CHECK: 1
    outputBuffer[0] = result;
}
