//TEST:SIMPLE(filecheck=METAL): -target metal
//TEST:SIMPLE(filecheck=CHECK): -target wgsl -entry computeMain -stage compute

// CHECK-DAG: @binding(9) @group(7)
// CHECK-DAG: @binding(3) @group(4)
// CHECK-DAG: @binding(1) @group(2)

// METAL-DAG: buffer(9)
// METAL-DAG: texture(7)

[vk::binding(1, 2)]
Texture2D texA : register(t7);

[vk::binding(3, 4)]
ConstantBuffer<float> cb;

RWStructuredBuffer<float> ob : register(u9, space7);

[numthreads(1,1,1)]
void computeMain()
{
    ob[0] = cb + texA.Load(int3(0)).x;
}