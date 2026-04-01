//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

// outputBuffer is defined in IBSDF.slang
//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name=outputBuffer

import MaterialSystem;
import DiffuseMaterial;
import DiffuseMaterialInstance;
import GlossyMaterial;
import GlossyMaterialInstance;

//TEST_INPUT: type_conformance DiffuseMaterial:IMaterial = 0
//TEST_INPUT: type_conformance GlossyMaterial:IMaterial = 1

[BackwardDifferentiable]
float3 evalBSDF(int type)
{
    float3 wi = normalize(float3(0.5, 0.2, 0.8));
    float3 wo = normalize(float3(-0.1, -0.3, 0.9));

    IMaterial material = createMaterialClassConformance(type, float3(0.9f, 0.6f, 0.2f));
    let mi = material.setupMaterialInstance();
    float3 f = mi.eval(wi, wo);
    return f;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    __bwd_diff(evalBSDF)(0, float3(1.f));
    __bwd_diff(evalBSDF)(1, float3(1.f));
}
