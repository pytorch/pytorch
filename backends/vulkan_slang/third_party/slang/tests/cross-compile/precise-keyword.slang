// precise-keyword.slang

//TEST:SIMPLE(filecheck=SPIRV): -target spirv-assembly -entry main -stage fragment
//TEST:CROSS_COMPILE:-target dxbc-assembly  -entry main -stage fragment
//TEST:CROSS_COMPILE:-target dxil-assembly  -entry main -stage fragment -profile sm_6_0

// Test handling of the `precise` keyword

// SPIRV: OpEntryPoint

float4 main(float2 v : V) : SV_Target
{
	precise float z;

	if(v.x > 0)
	{
		z = v.x * v.y + v.x;
	}
	else
	{
		z = v.y * v.x + v.y;
	}

	return z;
}
