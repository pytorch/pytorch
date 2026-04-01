//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IFoo : IDifferentiable
{
    [Differentiable]
    __init(Differential v);
}

struct Impl : IFoo
{
    float x;

    [Differentiable]
    __init(Differential v)
    {
        x = v.x;
    }

    // We have to add this __init so that the following code can still work:
    // Impl.Differential v0 = { (float)x };
    // because when there is a explicit constructor defined, we will not fall back
    // to legacy constructor. So this construction will fail.
    [Differentiable]
    __init(float v)
    {
        x = v;
    }
}

[Differentiable]
float test(float x)
{
    Impl.Differential v0 = { x };
    var v1 = Impl(v0);
    return v1.x * v1.x;
}

[numthreads(1,1,1)]
void computeMain(uint tid : SV_DispatchThreadID)
{
    var p = diffPair(3.0, 0.0);
    bwd_diff(test)(p, 1.0);
    outputBuffer[tid] = p.d;
    // CHECK: 6.0
}
