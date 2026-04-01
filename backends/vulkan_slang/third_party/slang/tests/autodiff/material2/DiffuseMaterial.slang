//TEST_IGNORE_FILE:

__exported import IMaterial;
__exported import DiffuseMaterialInstance;

public struct DiffuseMaterial : IMaterial
{
    public float3 baseColor;

    public typedef DiffuseMaterialInstance MaterialInstance;

    [TreatAsDifferentiable]
    public float3 getAlbedo(float3 albedo)
    {
        return albedo;
    }

    [BackwardDifferentiable]
    public DiffuseMaterialInstance setupMaterialInstance(out MaterialInstanceData miData)
    {
        float3 albedo = getAlbedo(baseColor);
        DiffuseMaterialInstance mi = { baseColor };
        for (uint i = 0; i < 3; i++) miData.data[i] = albedo[i];

        // unused
        miData.data[3] = 0.f;
        miData.data[4] = 0.f;
        
        return mi;
    }
};
