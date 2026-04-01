//TEST_IGNORE_FILE:

__exported import IMaterialInstance;
__exported import IBSDF;

public struct GlossyBSDF : IBSDF
{
    public float3 albedo;

    public float3 getAlbedo()
    {
        return albedo;
    }

    [ForwardDerivativeOf(getAlbedo)]
    [TreatAsDifferentiable]
    public DifferentialPair<float3> __fwd_d_getAlbedo()
    {
        return diffPair(albedo, float3(1.f));
    }

    [BackwardDerivativeOf(getAlbedo)]
    [TreatAsDifferentiable]
    public void __bwd_d_getAlbedo(float3 dOut)
    {
        [unroll]
        for (int j = 0; j < 3; j++) outputBuffer[j + 3] += dOut[j];
    }

    [BackwardDifferentiable]
    public float3 eval(const float3 wiLocal, const float3 woLocal)
    {
        float3 a = getAlbedo();
        return a * a;
    }
};

public struct GlossyMaterialInstance : IMaterialInstance
{
    public GlossyBSDF bsdf;

    [BackwardDifferentiable]
    public float3 eval(const float3 wi, const float3 wo)
    {
        return bsdf.eval(wi, wo);
    }
};
