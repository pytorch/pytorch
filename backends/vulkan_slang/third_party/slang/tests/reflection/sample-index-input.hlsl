//TEST:REFLECTION:-profile ps_5_0 -target hlsl -no-codegen

// Confirm that we register a shader as sample-rate when
// it declares `SV_SampleIndex` as an input.

struct PSInput
{
	float4 color : COLOR;
	uint sampleIndex : SV_SampleIndex;	
};

float4 main(PSInput input) : SV_Target
{
	return input.color;
}