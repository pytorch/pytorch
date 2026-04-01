//TEST_IGNORE_FILE:

struct Material
{
	float4 color;
	Texture2D t;
	SamplerState s;
};

float4 evaluateMaterial(Material m, float2 uv)
{
	return m.color + m.t.Sample(m.s, uv);
}
