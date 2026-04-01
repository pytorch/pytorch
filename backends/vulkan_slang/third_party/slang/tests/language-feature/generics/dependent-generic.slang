//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

// Test that we can define a generic where one of the type parameter conforms to an generic interface parameterized on another
// type parameter.

interface IFoo
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
    int getVal(T t);
}

struct Bar<T : IFoo> : IBar<T>
{
    int getVal(T t)
    {
        return t.get();
    }
}

int test<T:IFoo, B : IBar<T>>(B b, T t)
{
    return b.getVal(t);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    Foo obj1;
    Bar<Foo> obj2;
    let result = test(obj2, obj1);

    // CHECK: 1
    outputBuffer[0] = result;
}
