//TEST_IGNORE_FILE:

__exported import IMaterial;
__exported import DiffuseMaterialInstance;

public struct DiffuseMaterial : IMaterial
{
    public float3 baseColor;

    public typedef DiffuseMaterialInstance MaterialInstance;

    public DiffuseMaterialInstance setupMaterialInstance()
    {
        DiffuseMaterialInstance mi = { {baseColor} };
        return mi;
    }
};
