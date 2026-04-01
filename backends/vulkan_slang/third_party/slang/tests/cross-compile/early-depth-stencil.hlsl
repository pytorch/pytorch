//TEST:SIMPLE(filecheck=SPIRV):-target spirv-assembly -entry main -stage fragment
//TEST:SIMPLE(filecheck=SPIRV):-target spirv-assembly -entry main -stage fragment -emit-spirv-via-glsl

// SPIRV: OpExecutionMode %main EarlyFragmentTests

[earlydepthstencil]
float4 main(): SV_Target
{
    return float4(1, 0, 0, 1); 
}

