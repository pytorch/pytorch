//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry PSMain -stage fragment
struct PSInput
{
	float4 color : COLOR;
};

// CHECK: OpEntryPoint

float4 PSMain(PSInput input) : SV_TARGET
{
    const int nTaps = 2;
    const float flOffsets[ nTaps ] = { 0.0, 0.1 };
    return input.color.rgba + flOffsets[0].xxxx;
}