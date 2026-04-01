//TEST:SIMPLE(filecheck=CHECK): -target spirv-assembly -capability GL_NV_fragment_shader_barycentric -entry main -stage fragment

// CHECK: OpCapability FragmentBarycentricKHR
// CHECK: OpDecorate [[NAME:%[A-Za-z0-9_]+]] BuiltIn BaryCoordKHR
// CHECK: [[NAME]] = OpVariable {{.*}} Input
// CHECK: {{.*}} = OpLoad %v3float [[NAME]]

float4 main(float3 bary : SV_Barycentrics) : SV_Target
{
    return float4(bary, 0);
}
