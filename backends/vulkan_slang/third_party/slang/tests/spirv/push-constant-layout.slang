//TEST:SIMPLE(filecheck=CHECK): -target spirv -fvk-use-entrypoint-name
// CHECK-NOT: std140
struct Transform
{
    float4 Tint;
    float2x2 ScaleRot;
    float2 Translation;
};

[[vk::push_constant]]
ConstantBuffer<Transform> transform1;

RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain1()
{
    outputBuffer[0] = transform1.Translation.x;
}

[numthreads(1,1,1)]
void computeMain2(
    [vk::push_constant] ConstantBuffer<Transform> transform2)
{
    outputBuffer[0] = transform2.Translation.x;
}