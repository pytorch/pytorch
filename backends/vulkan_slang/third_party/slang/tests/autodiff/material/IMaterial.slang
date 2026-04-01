//TEST_IGNORE_FILE:

__exported import IMaterialInstance;

public interface IMaterial
{
    public associatedtype MaterialInstance : IMaterialInstance;

    public MaterialInstance setupMaterialInstance();
}
