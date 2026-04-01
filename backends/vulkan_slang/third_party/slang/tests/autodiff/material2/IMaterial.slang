//TEST_IGNORE_FILE:

__exported import IMaterialInstance;

public interface IMaterial
{
    public associatedtype MaterialInstance : IMaterialInstance;

    [BackwardDifferentiable]
    public MaterialInstance setupMaterialInstance(out MaterialInstanceData miData);
}
