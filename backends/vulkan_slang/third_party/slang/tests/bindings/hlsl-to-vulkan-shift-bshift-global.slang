//TEST:SIMPLE(filecheck=CHECK):-target glsl -profile glsl_450 -entry MainVs -stage vertex -fvk-b-shift 14 0 -fvk-s-shift 58 0 -fvk-t-shift 218 0 -fvk-u-shift 482 0

// CHECK: layout(binding = 15)
// CHECK-NEXT: layout(std140) uniform
// CHECK-NEXT: {
// CHECK-NEXT: vec4 g_vCBuffer0Values

// CHECK: layout(binding = 14)
// CHECK-NEXT: layout(std140) uniform
// CHECK-NEXT: {
// CHECK-NEXT: vec4 g_vGlobalValues

float4 g_vGlobalValues; // This ends up in globals

cbuffer CBufferTest_t 
{
	float4 g_vCBuffer0Values;
};

struct VS_INPUT
{
	float3 vPositionWs : POSITION;
};

struct PS_INPUT
{
	float4 vColor;
};

PS_INPUT MainVs( VS_INPUT i )
{
	PS_INPUT o;
	o.vColor.rgba = g_vCBuffer0Values + g_vGlobalValues;
	return o;
}