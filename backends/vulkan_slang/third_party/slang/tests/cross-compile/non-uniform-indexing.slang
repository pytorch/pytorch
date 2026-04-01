//TEST:SIMPLE(filecheck=CHECK):-target spirv-assembly -entry main -stage fragment -verify-debug-serial-ir
//TEST:SIMPLE(filecheck=CHECK):-target spirv-assembly -entry main -stage fragment -verify-debug-serial-ir -DGLSL -allow-glsl

// Confirm that `NonUniformResourceIndex` translates to SPIR-V as expected

// CHECK: NonUniform

Texture2D t[10];
SamplerState s;

float4 main(
    float3 uv : UV)
    : SV_Target
{
#if defined(GLSL)
    return t[nonuniformEXT(int(uv.z))].Sample(s, uv.xy);
#else
    return t[NonUniformResourceIndex(int(uv.z))].Sample(s, uv.xy);
#endif
}
