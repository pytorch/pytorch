//TEST(compute):COMPARE_COMPUTE_EX:-dx12 -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-wgpu -compute -shaderobj -output-using-type
//TEST:SIMPLE(filecheck=CHECK): -target hlsl -profile cs_5_0 -entry computeMain -line-directive-mode none
//DISABLE_TEST:SIMPLE(filecheck=CHK):-target glsl -stage compute -entry computeMain -report-checkpoint-intermediates

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

// Test that compute does not have a context.
// CHECK-NOT: struct {{[a-zA-Z0-9_]*}}_compute_{{[a-zA-Z0-9_]*}}

[BackwardDifferentiable]
[PreferRecompute]
float compute(float x, float y)
{
    return x * y;
}

[BackwardDifferentiable]
[ForceInline]
float infinitesimal(float x)
{
    return x - detach(x);
}

// Test that computeLoop compiles to just return 0.
// CHECK: float3 computeLoop{{[_0-9]*}}(float y{{[_0-9]*}})
// CHECK-NOT: for{{.*}}
// CHECK: return (float3)0

// Test that computeLoop's intermediates have no float sitting
// around (must not cache the outvar from 'compute()')
// CHECK: struct s_bwd_prop_computeLoop_Intermediates
// CHECK-NEXT: {
// CHECK-NOT: {{[A-Za-z0-9_]+}} {{[A-Za-z0-9_]+}}[{{.*}}]
// CHECK: }

[PreferRecompute]
[BackwardDifferentiable]
[ForceInline]
float3 infinitesimal(float3 x)
{
    return x - detach(x);
}

//CHK: note: checkpointing context of 20 bytes associated with function: 'computeLoop'
[BackwardDifferentiable]
[PreferRecompute]
float3 computeLoop(float y)
{
    //CHK: note: 4 bytes (float) used to checkpoint the following item:
    float w = 0;

    //CHK: note: 12 bytes (Vector<float, 3> ) used to checkpoint the following item:
    float3 w3 = float3(0, 0, 0);

    //CHK: note: 4 bytes (int32_t) used for a loop counter here:
    for (int i = 0; i < 8; i++)
    {
        float k = compute(i, y);
        float g = select(k > 0.0, k, 0.0);
        w += g;
        w3 += float3(g) * y;
    }

    var p = (w3 / float3(w));
    return infinitesimal(p);
}

// Since computeLoop is recomputed, test_simple_loop should have nothing to store
// therefore we check that there is no intermediate context type generated for test_simple_loop.

// CHECK-NOT: struct {{[a-zA-Z0-9_]*}}test_simple_loop{{[a-zA-Z0-9_]*}}
[BackwardDifferentiable]
float test_simple_loop(float y)
{  
    float3 x = computeLoop(y);
    return y + x.x;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(1.0, 0.0);

        __bwd_diff(test_simple_loop)(dpa, 1.0f);
        outputBuffer[0] = dpa.d; // Expect: 1.0
    }

    {
        dpfloat dpa = dpfloat(0.4, 0.0);

        __bwd_diff(test_simple_loop)(dpa, 0.5f);
        outputBuffer[1] = dpa.d; // Expect: 2.0
    }

    outputBuffer[2] = computeLoop(1.0).x;
}

//CHK-NOT: note