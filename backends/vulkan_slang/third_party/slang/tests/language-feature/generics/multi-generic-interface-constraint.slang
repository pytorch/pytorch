//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

interface IFoo<T : __BuiltinFloatingPointType>
{
    bool requirement(T v1);
}

struct Bar<T : __BuiltinFloatingPointType, A : IFoo<T>, B : IFoo<T>>
{
    [Differentiable]
    T doThing(T a)
    {
        return a * T(2.0);
    }
}

struct Foo : IFoo<float>
{
    bool requirement(float v)
    {
        return v > 0.5;
    }
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    Bar<float, Foo, Foo> obj;
    outputBuffer[tid] = obj.doThing(2.0f);
    // CHECK: 4
}
