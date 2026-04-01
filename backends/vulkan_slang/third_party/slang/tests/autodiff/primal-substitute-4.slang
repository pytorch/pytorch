//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type -g0

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBufferPrimal
RWStructuredBuffer<float> outputBufferPrimal;

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):name=gradBuffer
RWStructuredBuffer<float> gradBuffer;

struct BufferWithGrad
{
    RWStructuredBuffer<float> primal;
    RWStructuredBuffer<float> grad;

    [Differentiable]
    void add(float value) { primal[0] = primal[0] + detach(value); }

    [PrimalSubstituteOf(add), Differentiable]
    void add_subst(float value)
    {
    }

    [BackwardDerivativeOf(add)]
    void add_bwd(inout DifferentialPair<float> d)
    {
        d = diffPair(d.p, grad[0]);
    }
}

[Differentiable]
void diffCall(BufferWithGrad result)
{
    result.add(1.0f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    BufferWithGrad bg = {outputBufferPrimal, gradBuffer};
    diffCall(bg); 
    bwd_diff(diffCall)(bg);

    // CHECK: type: float
    // CHECK-NEXT: 1.0
    // CHECK-NEXT: 0.0
}
