//TEST_DISABLED:CROSS_COMPILE: -profile vs_5_0 -entry main -target spirv-assembly

// Note: disabled until we add a more complete IR solution
// for translating HLSL semantics into equivalent GLSL
// builtin variables.

struct VS_OUT
{
	float4 left : SV_Position;
    float4 right : NV_X_RIGHT;
    uint4  mask : NV_VIEWPORT_MASK;
};

VS_OUT main(float4 ll, float4 rr)
{
	VS_OUT res;
	res.left = ll;
	res.right = rr;
	res.mask = 0x1;
	return res;	
}