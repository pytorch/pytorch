//TEST_IGNORE_FILE:

public struct MaterialInstanceData : IDifferentiable
{
    public float data[5];
}

/*
struct MaterialInstanceSetupResult : IDifferentiable
{
    IMaterialInstance mi;
    MaterialInstanceData data;
}
*/

public interface IMaterialInstance
{
    [BackwardDifferentiable]
    public float3 eval(const MaterialInstanceData miData, const float3 wi, const float3 wo);
}
