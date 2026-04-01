// gh-666.slang

//TEST:COMPARE_HLSL: -profile ps_5_0 -entry main

static const uint foo = 1;
static const uint bar = foo;

float4 main() : SV_TARGET
{
	return bar;
}
