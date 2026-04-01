//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IFoo : IDifferentiable
{
    [Differentiable]
    __init<T:IFoo>(T d); 
    
    [Differentiable]
    static This create(Differential d);

    [Differentiable]
    float getVal();
}

struct Impl : IFoo
{
    float x;
    [Differentiable]
    __init<T : IFoo>(T d)
    {
        this = (Impl)d;
    }
    
    [Differentiable]
    static This create(Differential d)
    {
        This v;
        v.x = d.x * d.x;
        return v;
    }

    [Differentiable]
    float getVal() { return x; }
}

[Differentiable]
float test<T:IFoo>(T.Differential d)
{
    return T.create(d).getVal();
}

[numthreads(1,1,1)]
void computeMain(uint threadId: SV_DispatchThreadID)
{
    Impl.Differential d;
    d.x = 3.0;
    Impl.Differential dd;
    dd.x = 1.0;
    outputBuffer[0] = fwd_diff(test<Impl>)(diffPair(d, dd)).d;
    // CHECK: 6.0
}
