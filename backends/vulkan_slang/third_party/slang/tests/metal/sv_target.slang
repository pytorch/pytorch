//TEST:SIMPLE(filecheck=CHECK): -target metal
//TEST:SIMPLE(filecheck=CHECK-ASM): -target metallib

struct Output
{
  float4 Diffuse  : SV_Target0;
  float4 Normal   : SV_Target1;
  float4 Material : SV_Target2;
}

// CHECK-ASM: define {{.*}} @fragMain
// CHECK: color(0)
// CHECK: color(1)
// CHECK: color(2)

[shader("fragment")]
Output fragMain()
{
    return { float4(1), float4(2), float4(3) };
}