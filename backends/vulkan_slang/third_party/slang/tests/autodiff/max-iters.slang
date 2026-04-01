//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type -d3d12 -use-dxil
//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type -vk
//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type -metal
//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type -cuda
//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type -cpu
//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type -wgpu

// Note: there is a bug in fxc compiler errorneously reporting infinite loop for this shader.
// Skipping d3d11 test to avoid the bug.
//DISABLE_TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type -d3d11

struct GradientBuffer<let D : int>
{
    RWStructuredBuffer<float> primal;
    StructuredBuffer<float> grad;
    int strides[D];

    int toIndex(int idx[D]) {
        int result = 0;
        for (int i = 0; i < D; ++i)
            result += strides[i] * idx[i];
        return result;
    }

    [Differentiable]
    void write(int[D] idx, float v) { primal[toIndex(idx)] = detach(v); }

    [BackwardDerivativeOf(write)]
    void write_bwd(int[D] idx, inout DifferentialPair<float> d) { d = diffPair(d.p, grad[toIndex(idx)]); }

    [Differentiable]
    void store<let N : int>(int context[D - 1], in float value[N])
    {
        int idx[D];
        //[ForceUnroll] /* Using ForceUnroll instead of MaxIters makes it work */
        [MaxIters(2)]
        for (int i = 0; i < D - 1; ++i)
            idx[i] = context[i];
        [ForceUnroll]
        for (int i = 0; i < N; i++) {
            idx[D - 1] = i;
            write(idx, value[i]);
        }
    }
}

[Differentiable]
void test(GradientBuffer<2> buf, int[1] base, float[3] value)
{
    buf.store(base, value);
}

float3 repro(RWStructuredBuffer<float> primal, StructuredBuffer<float> grad)
{
    float input[3];
    input[0] = input[1] = input[2] = 1.0f;
    var result = diffPair(input);
    GradientBuffer<2> buf = { primal, grad, {3, 1} };
    bwd_diff(test)(buf, { 1 }, result);
    return float3(result.d[0], result.d[1], result.d[2]);
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
    let result = repro(grad_out, grad_in);
    // CHECK: 104.0
    output[0] = result.x;
    output[1] = result.y;
    output[2] = result.z;
}