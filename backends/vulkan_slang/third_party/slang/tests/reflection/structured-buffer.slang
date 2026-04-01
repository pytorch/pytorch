//TEST:REFLECTION:-profile ps_4_0 -target hlsl -no-codegen

// Confirm that we reflect the contents of structure-buffer types correctly.

struct S
{
	float2 a;
	float  b;
	uint   c;
};

StructuredBuffer<uint> x;
StructuredBuffer<float2> y;
StructuredBuffer<S> z;

float4 main() : SV_Target
{
    return x[0] + y[0].xyxy + z[0].a.xyxy;
}