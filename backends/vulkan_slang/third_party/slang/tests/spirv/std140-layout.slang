
//TEST:SIMPLE(filecheck=SPIRV): -stage fragment -entry main -target spirv -emit-spirv-directly

// SPIRV: OpMemberDecorate %{{.*}} 0 Offset 0
// SPIRV: OpMemberDecorate %{{.*}} 1 Offset 16
// SPIRV: OpMemberDecorate %{{.*}} 2 Offset 28

uniform float3 v0;
uniform float3 v1;
uniform float v2;



float4 main() : SV_Target
{
    return float4(v2);
}
