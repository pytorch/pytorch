//TEST_IGNORE_FILE:

__exported import IMaterialInstance;
__exported import IBSDF;

public struct DiffuseMaterialInstance : IMaterialInstance
{
    public float3 albedo;

    [BackwardDifferentiable]
    public float3 eval(const MaterialInstanceData miData, const float3 wi, const float3 wo)
    {
        float3 albedo;
        for (uint i = 0; i < 3; i++) albedo[i] = miData.data[i];
        return albedo;
    }
};
