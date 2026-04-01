//TEST:SIMPLE(filecheck=HLSL): -entry main -target hlsl
//TEST:SIMPLE(filecheck=GLSL): -entry main -target glsl
//TEST:SIMPLE(filecheck=SPIRV): -entry main -target spirv

// HLSL-DAG: main
// HLSL-DAG: SV_InstanceID

// GLSL-DAG: main
// GLSL-DAG: gl_InstanceIndex

// SPIRV: OpEntryPoint
// SPIRV-NOT: BuiltIn InstanceIndex

// We require 1 `Flat` for the fragment input `uint`
// SPIRV: OpDecorate %{{[1-9][0-9]*}} Flat
// SPIRV-NOT: OpDecorate %{{[1-9][0-9]*}} Flat


struct PSInput
{
    uint vInstance : SV_InstanceID;
	float4 color : COLOR;
};

[shader("pixel")]
float4 main(PSInput input) : SV_TARGET
{
	return input.color + float(input.vInstance).xxxx;
}
