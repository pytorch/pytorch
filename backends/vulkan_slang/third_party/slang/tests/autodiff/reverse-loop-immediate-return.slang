
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK): -slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;


[BackwardDerivative(set_bwd)]
void set(uint idx, float x)
{
    outputBuffer[idx] = x;
}

void set_bwd(uint idx, inout DifferentialPair<float> x)
{
    // For debugging, we'll set the derivative to 1.0
    x = DifferentialPair<float>(x.p, 1.0f);
}

[Differentiable]
void run(
    uint idx,
    float x)
{
    if (idx >= 1) return;

    if (idx == 0)
    { }

    for (int i = 0; i < 1; i++)
    {
        if (idx > 0)
        {
            return;
        }

        if (idx == 0)
        { 
            x = x * 2.0f;
        }
    }

    if (idx == 0)
    { }

    set(idx, x);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // bwd_diff
    DifferentialPair<float> dpa = DifferentialPair<float>(1.0, 0.0);
    bwd_diff(run)(dispatchThreadID.x, dpa);
    outputBuffer[dispatchThreadID.x] = dpa.d; 
    
    // CHECK: type: float
    // CHECK: 2.0
}
