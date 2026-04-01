//DISABLE_TEST:SIMPLE: -profile ps_5_0 -entry main -target c

//Texture2D t;
//SamplerState s;

float4 main(float2 uv) : SV_Target
{
	float4 result = 1;
	
	result.x += uv.x;
	result.y += uv.y;
	result.z -= uv.x;
	result.w -= uv.y;
	
	return result;
}
