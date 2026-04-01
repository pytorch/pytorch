//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type

struct GradientBuffer
{
    StructuredBuffer<float> grads;

    [Differentiable]
    void write(int idx, float value) { /* Discard write */ }

    [BackwardDerivativeOf(write)]
    void write_bwd(int idx, inout DifferentialPair<float> d)
    {
        d = diffPair(d.p, grads[idx]);
    }
}

[Differentiable]
void test(GradientBuffer dst, int idx, float v)
{
    dst.write(idx, v);
}

//TEST_INPUT: set grad_in = ubuffer(data=[101.0 102.0 103.0 104.0], stride=4)
uniform StructuredBuffer<float> grad_in;

//TEST_INPUT: set grad_out = ubuffer(data=[0 0 0 0], stride=4)
uniform RWStructuredBuffer<float> grad_out;

//TEST_INPUT: set output = out ubuffer(data=[0 0 0 0], stride=4)
uniform RWStructuredBuffer<float> output;

[shader("compute")]
[numthreads(1,1,1)]
void computeMain()
{
    GradientBuffer grads = { grad_in };
    DifferentialPair<float> result = diffPair(1.0f);
    bwd_diff(test)(grads, 0, result);
    // CHECK: 101.0
    output[0] = result.d; // Should return grad_in[0], but returns 0.0f instead
}