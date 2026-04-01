//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST:SIMPLE(filecheck=CHECK): -target hlsl -profile cs_5_0 -entry computeMain -line-directive-mode none

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct A : IDifferentiable
{
    float data[5];
};

// Check that the intermediate context of B.eval does not have any arrays.
// This will fail if the induction variable is not properly detected, or
// if the various loop restructuring passes accidentally introduce additional 
// loop state.
// 

// CHECK: struct s_bwd_prop_B_eval_Intermediates_0
// CHECK-NOT: int {{[A-Za-z0-9_]+}}[{{.*}}]
// CHECK: }

__generic<let TBsdfCount : int>
struct B
{
    [Differentiable]
    float3 eval(const A miData, const float3 wi, const float3 wo)
    {
        float3 albedo;
        for (uint i = 0; i < 3; i++) albedo[i] = miData.data[i];

        float3 result = float3(1.f);
        [ForceUnroll] for (uint i = 0; i < TBsdfCount; i++) result *= albedo;
        return result;
    }
};

[Differentiable]
float3 outerEval(const A miData, const float3 wi, const float3 wo)
{
    B<3> b;
    return b.eval(miData, wi, wo);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    float3 wi = float3(2.0, 3.0, 0);
    float3 wo = float3(1.0, 1.0, 0);
    float data[5] = { 1, 2, 3, 4, 5 };
    A dataStruct = { data };

    float3 val = outerEval(dataStruct, wi, wo);
    outputBuffer[0] = val.x;

    DifferentialPair<float3> dpwi = diffPair(wi);
    DifferentialPair<float3> dpwo = diffPair(wo);
    DifferentialPair<A> dpdata = diffPair(dataStruct);
    float3 dOut = float3(1.0, 0.0, 0.0);
    __bwd_diff(outerEval)(dpdata, dpwi, dpwo, dOut);

    // Write output
    outputBuffer[0] = dpdata.d.data[0];
}