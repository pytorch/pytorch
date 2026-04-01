//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type -g0

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

void foo_bwd(float a, inout DifferentialPair<float> dpx)
{
    outputBuffer[2] = 2.f;
}

[Differentiable, BackwardDerivative(foo_bwd)]
void foo(no_diff float a, float x)
{ }

[Differentiable]
float outerFunc(no_diff float a, float x)
{
    foo(a, x);
    return 1.f;
}

[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    float a = 10.0;
    DifferentialPair<float> dpx = DifferentialPair<float>(4.f, 1.f);
    bwd_diff(outerFunc)(a, dpx, 1.0);
    
    // CHECK: type: float
    // CHECK: 0.0
    // CHECK: 0.0
    // CHECK: 2.0
    // CHECK: 0.0
}