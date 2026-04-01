//TEST:SIMPLE(filecheck=CHECK):-target spirv-assembly -entry main -stage fragment

// CHECK: OpDecorate %{{.*}} BuiltIn BaryCoordKHR

float4 main(float3 bary : SV_Barycentrics) : SV_Target
{
    return float4(bary, 0);
}
