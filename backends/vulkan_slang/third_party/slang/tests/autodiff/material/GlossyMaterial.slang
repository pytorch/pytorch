//TEST_IGNORE_FILE:

__exported import IMaterial;
__exported import GlossyMaterialInstance;

public struct GlossyMaterial : IMaterial
{
    public float3 baseColor;

    public typedef GlossyMaterialInstance MaterialInstance;

    public GlossyMaterialInstance setupMaterialInstance()
    {
        GlossyMaterialInstance mi = { { baseColor } };
        return mi;
    }
};
