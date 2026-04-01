//TEST_IGNORE_FILE:

__exported import IMaterialInstance;
__exported import IBSDF;

import MxWeights;

public struct MxLayeredMaterialInstance<
    let TBsdfCount : int,
    let TLayerCount : int,
    TLayeredMaterialData : IMxLayeredMaterialData,
    TWeightsCalc : IMxLayeredWeightCalculator> : IMaterialInstance
{
    public TLayeredMaterialData data;
    public TWeightsCalc calculator;
    public typealias Weights = MxWeights<TBsdfCount>;

    public Weights calculateWeights()
    {
        Weights result;
        calculator.calculateWeights<TBsdfCount, TLayerCount, TLayeredMaterialData>(data, result);
        return result;
    }

    [Differentiable]
    public float3 eval(const MaterialInstanceData miData, const float3 wi, const float3 wo)
    {
        float3 albedo;
        for (uint i = 0; i < 3; i++) albedo[i] = miData.data[i];

        float3 result = float3(1.f);
        [ForceUnroll] for (uint i = 0; i < TBsdfCount; i++) result *= albedo;
        return result;
    }
}
