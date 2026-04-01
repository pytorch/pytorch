//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -use-dxil -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj -output-using-type

struct DiffFunctor : IDifferentiableFunc<float, float>
{
    [Differentiable]
    float operator()(float p)
    {
        return p + 1;
    }
}

float apply(IMutatingFunc<float, float> f, float p)
{
    return f(p);
}

[Differentiable]
float applyDiff(IDifferentiableFunc<float, float> f, float p)
{
    return f(p);
}

[Differentiable]
TR applyDiffGen<TR : IDifferentiable, each TP : IDifferentiable>(IDifferentiableFunc<TR, TP> f, expand each TP p)
{
    return f(expand each p);
}

//TEST_INPUT:ubuffer(data=[0 3 2 2], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint tid: SV_DispatchThreadID)
{
    // CHECK: 4
    outputBuffer[0] = (uint)apply(DiffFunctor(), 3.0);
    // CHECK: 1
    outputBuffer[1] = (uint)fwd_diff(applyDiff)(DiffFunctor(), diffPair(2.0, 1.0)).d;
    // CHECK: 1
    outputBuffer[2] = (uint)fwd_diff(applyDiffGen<float, float>)(DiffFunctor(), diffPair(2.0, 1.0)).d;
}
