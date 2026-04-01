//TEST_IGNORE_FILE:

__exported import IMaterialInstance;
__exported import IBSDF;

public struct DiffuseBSDF : IBSDF
{
    public float3 albedo;

    [TreatAsDifferentiable]
    public float3 eval(const float3 wiLocal, const float3 woLocal)
    {
        return albedo;
    }
};

public struct DiffuseMaterialInstance : IMaterialInstance
{
    public DiffuseBSDF bsdf;

    [TreatAsDifferentiable]
    public float3 eval(const float3 wi, const float3 wo)
    {
        return bsdf.eval(wi, wo);
    }
};
