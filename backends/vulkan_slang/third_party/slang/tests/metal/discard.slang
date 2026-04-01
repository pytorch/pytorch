//TEST:SIMPLE(filecheck=CHECK): -target metal
//TEST:SIMPLE(filecheck=CHECK-ASM): -target metallib

// CHECK: discard_fragment();

// CHECK-ASM: define {{.*}} @main_fragment1

struct VOut
{
    float4 position : SV_Position;
    float4 vertexColor;
    float2 vertexUV;
    float3 vertexNormal : NORMAL;
}

[shader("fragment")]
float4 main_fragment1(VOut fragmentIn)
{
    if (fragmentIn.vertexColor.x == 0.0)
        discard;
    return fragmentIn.vertexColor;
}