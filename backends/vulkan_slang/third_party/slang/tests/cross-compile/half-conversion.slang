// half-conversion.slang

//TEST:SIMPLE(filecheck=SPIRV):-target spirv-assembly -entry main -stage fragment

// SPIRV: OpBitcast

cbuffer C
{
	uint4 u;
}

float4 main() : SV_Target
{
	return f16tof32(u);
}
