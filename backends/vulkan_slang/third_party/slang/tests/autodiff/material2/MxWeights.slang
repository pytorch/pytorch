//TEST_IGNORE_FILE:

public struct MxWeights<let TBsdfCount : int>
{
    public float3 weights[TBsdfCount];

    public __init()
    {
        for (int i = 0; i < TBsdfCount; i++)
            weights[i] = float3(0.0f);
    }
}

public interface IMxLayeredMaterialData
{
    public int getBsdfCount();
    public int getLayerCount();
}

public interface IMxLayeredWeightCalculator
{
    public void calculateWeights<let TBsdfCount : int, let TLayerCount: int, TLayeredMaterialData : IMxLayeredMaterialData>(
        const TLayeredMaterialData data, out MxWeights<TBsdfCount> weights);
}
