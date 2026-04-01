// This test checks that kIROp_StructuredBufferLoad and similar instructions
// are not movable.

//TEST:SIMPLE(filecheck=CHECK): -entry computeMain -stage compute -target hlsl

[[vk::binding(0, 0)]]
StructuredBuffer<float> gSomeData;

[[vk::binding(1, 0)]]
RWTexture2D<float4> gResultImage;

struct PushConstants
{
    bool bufferHasOnlyOneElement;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

float loadDataConditionTrue()
{
    return gSomeData[0];
}

float loadDataConditionFalse()
{
    return gSomeData[1];
}

[ForceInline]
float getDataDependingOnCondition(bool condition)
{
    if (condition)
    {
        return loadDataConditionTrue();
    }
    else
    {
        return loadDataConditionFalse();
    }

    return 0.0;
}

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: if
    // CHECK: loadDataConditionTrue
    // CHECK: else
    // CHECK: loadDataConditionFalse
    float v = getDataDependingOnCondition(gPushConstants.bufferHasOnlyOneElement);

    int2 pixelIndex = int2(0);
    gResultImage[pixelIndex] = float4(v, v, v, 1.0);
}
