//TEST_DISABLED:CROSS_COMPILE: -profile ps_5_0 -entry main -target spirv-assembly

// Note: disabled pending IR-based translation of the `Sample()`
// operation for Vulkan.

Texture2D t;
SamplerState s;

float4 main(float2 uv) : SV_Target
{
	float4 result = 0;
	$for(i in Range(0,5))
	{
		result += t.Sample(s, uv, int2(i - 2, 0));
	}
	return result;
}
