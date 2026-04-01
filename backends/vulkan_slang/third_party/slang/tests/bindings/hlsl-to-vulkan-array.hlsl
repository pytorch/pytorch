//TEST:REFLECTION:-target glsl -profile ps_4_0 -entry main -fvk-t-shift 10 all  -fvk-s-shift 100 all -fvk-u-shift 100 all -fvk-b-shift 1000 all

struct Data
{
    Texture2D tex;
    RWStructuredBuffer<float> structuredBuffer;
    float a;
    int b;
};

Data g_data[2];

float4 main() : SV_TARGET
{
	return float4(1, 1, 1, 0);
}