//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IFoo : IDifferentiable
{
    [Differentiable]
    __init();

    float compute(float y);
}

struct Impl : IFoo
{
    float x;
    [Differentiable]
    __init()
    {
        x = 4.0;
    }

    float compute(int id) { return outputBuffer[id] = detach(x);}
}

float test<T : IFoo>(int id)
{
    T t;
    return t.compute(id);
}

[numthreads(1, 1, 1)]
void computeMain(uint id : SV_DispatchThreadID)
{
    // CHECK: 4.0
    test<Impl>(id); 
}
