//TEST_IGNORE_FILE:

__exported import IMaterial;
__exported import GlossyMaterialInstance;

public struct GlossyMaterial : IMaterial
{
    public float3 baseColor;

    public typedef GlossyMaterialInstance MaterialInstance;

    public float3 getAlbedo(float3 albedo)
    {
        return albedo;
    }

    [ForwardDerivativeOf(getAlbedo)]
    [TreatAsDifferentiable]
    public DifferentialPair<float3> __fwd_d_getAlbedo(DifferentialPair<float3> dpAlbedo)
    {
        return diffPair(dpAlbedo.p, float3(1.f));
    }

    [BackwardDerivativeOf(getAlbedo)]
    [TreatAsDifferentiable]
    public void __bwd_d_getAlbedo(inout DifferentialPair<float3> dpAlbedo, float3 dOut)
    {
        [unroll]
        for (int j = 0; j < 3; j++) outputBuffer[j + 3] += dOut[j];
    }

    [BackwardDifferentiable]
    public GlossyMaterialInstance setupMaterialInstance(out MaterialInstanceData miData)
    {
        float3 albedo = getAlbedo(baseColor);
        GlossyMaterialInstance mi = { baseColor };
        for (uint i = 0; i < 3; i++) miData.data[i] = albedo[i];

        // unused
        miData.data[3] = 0.f;
        miData.data[4] = 0.f;
        
        return mi;
    }
};
