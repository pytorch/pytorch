//TEST:REFLECTION:-profile ps_5_0 -target hlsl -no-codegen

// Confirm that we register a shader as sample-rate when
// it declares (not necessarly *uses*) a `sample` qualified input

struct PSInput
{
    float4 extra : EXTRA;
	sample float4 color : COLOR;
};

float4 main(PSInput input) : SV_Target
{
	return input.extra + input.color;
}