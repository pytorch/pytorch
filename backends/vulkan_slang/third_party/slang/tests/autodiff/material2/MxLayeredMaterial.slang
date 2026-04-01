//TEST_IGNORE_FILE:

__exported import IMaterial;
__exported import IMaterialInstance;
import MxLayeredMaterialInstance;
import MxWeights;

public struct LayeredData_mixedLobes : IMxLayeredMaterialData
{
    public static const int bsdfCount = 3;
    public static const int layerCount = 1;

    public int getBsdfCount() { return bsdfCount; }
    public int getLayerCount() { return layerCount; }
}
#define TMxLayeredMaterialData LayeredData_mixedLobes

public struct Layered_mixedLobes_WeightsCalculator : IMxLayeredWeightCalculator
{
    public void calculateWeights<let TBsdfCount : int, let TLayerCount : int, TLayeredMaterialData : IMxLayeredMaterialData>(
        const TLayeredMaterialData data, out MxWeights<TBsdfCount> weights)
    {
        for (uint i = 0; i < TBsdfCount; i++) weights.weights[i] = float3(0.f);
    }
}

public struct MxLayeredMaterial : IMaterial
{
    public float3 baseColor;

    public typealias UsedMaterialInstance = MxLayeredMaterialInstance<3, 1, TMxLayeredMaterialData, Layered_mixedLobes_WeightsCalculator>;
    public typedef UsedMaterialInstance MaterialInstance;

    public float3 getAlbedo(float3 albedo)
    {
        return albedo;
    }

    [ForwardDerivativeOf(getAlbedo)]
    [TreatAsDifferentiable]
    public DifferentialPair<float3> __fwd_d_getAlbedo(DifferentialPair<float3> dpAlbedo)
    {
        return diffPair(dpAlbedo.p, float3(1.f));
    }

    [BackwardDerivativeOf(getAlbedo)]
    [TreatAsDifferentiable]
    public void __bwd_d_getAlbedo(inout DifferentialPair<float3> dpAlbedo, float3 dOut)
    {
        [unroll]
        for (int j = 0; j < 3; j++) outputBuffer[j + 6] += dOut[j];
    }

    [Differentiable]
    public UsedMaterialInstance setupMaterialInstance(out MaterialInstanceData miData)
    {
        float3 albedo = getAlbedo(baseColor);
        UsedMaterialInstance mi;
        for (uint i = 0; i < 3; i++) miData.data[i] = albedo[i];

        // unused
        miData.data[3] = 0.f;
        miData.data[4] = 0.f;
        
        return mi;
    }
}
