//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type
public struct ReadOnlyIndex
{
    private int _idx;
    __init(int i) { _idx = i; }
    public property int idx { get { return _idx; } }
}
struct GradientBuffer
{
    RWStructuredBuffer<float> primal;
    StructuredBuffer<float> grad;

    [Differentiable]
    void write(int idx, float v) { primal[idx] = detach(v); }

    [BackwardDerivativeOf(write)]
    void write_bwd(int idx, inout DifferentialPair<float> d) { d = diffPair(d.p, grad[idx]); }

    [Differentiable]
    void store(ReadOnlyIndex idx, float v) { write(idx.idx, v); }
}
[Differentiable]
void test(GradientBuffer buf, ReadOnlyIndex b, float x)
{
    buf.store(b, x);
}
public float repro(RWStructuredBuffer<float> primal, StructuredBuffer<float> grad)
{
    DifferentialPair<float> result = diffPair(1.0f);
    GradientBuffer buf = { primal, grad };
    bwd_diff(test)(buf, ReadOnlyIndex(5), result);
    return result.d;
}

//TEST_INPUT: set output = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<float> output;

//TEST_INPUT: set gPrimal = ubuffer(data=[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0], stride=4)
RWStructuredBuffer<float> gPrimal;
//TEST_INPUT: set gGrad = ubuffer(data=[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0], stride=4)
StructuredBuffer<float> gGrad;

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 5.0
    output[0] = repro(gPrimal, gGrad);
}