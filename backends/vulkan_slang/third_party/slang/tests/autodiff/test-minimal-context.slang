//TEST:SIMPLE(filecheck=CHECK): -target hlsl -profile cs_5_0 -entry computeMain -line-directive-mode none
//DISABLE_TEST:SIMPLE(filecheck=CTX):-target glsl -stage compute -entry computeMain -report-checkpoint-intermediates

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[BackwardDerivative(bwd_load)]
float load(uint idx)
{
    return outputBuffer[idx];
}

void bwd_load(uint idx, float dOut)
{
    outputBuffer[idx + 2] += dOut;
}

[BackwardDerivative(bwd_store)]
void store(uint idx, float a)
{
    outputBuffer[idx] = a;
}

[ForceInline]
float inner_bwd_store(uint idx)
{
    return outputBuffer[idx + 2];
}

[ForceInline]
void bwd_store(uint idx, inout DifferentialPair<float> a)
{
    a = diffPair(a.p, inner_bwd_store(idx));
}

[BackwardDerivative(bwd_g)]
float g(float x)
{
    return load(1) * load(1);
}

void bwd_g(inout DifferentialPair<float> x, float dOut)
{
    float y = load(1);
    x = diffPair(x.p + 2 * y, x.d + 2 * y * dOut);
    store(0, x.d);
}

[BackwardDifferentiable]
float f(int p, float x)
{
    float y = g(x);

    store(0, y);

    return 0;
}

// Check that there are no calls to primal_ctx_f in bwd_f.

// CHECK: void s_bwd_f_{{[0-9]+}}
// CHECK-NOT: s_primal_ctx_f_{{[0-9]+}}
// CHECK: return

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    dpfloat dpa = dpfloat(2.0, 0.0);

    bwd_diff(f)(0, dpa, 1.0f);
    outputBuffer[0] = dpa.d; // Expect: 1
}
// CTX: note: