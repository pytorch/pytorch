//TEST_IGNORE_FILE:

public interface IMaterialInstance
{
    [BackwardDifferentiable]
    public float3 eval(const float3 wi, const float3 wo);
}
