//TEST:REFLECTION:-profile ps_4_0 -target hlsl -no-codegen -matrix-layout-column-major
//TEST:REFLECTION:-profile ps_4_0 -target hlsl -no-codegen -matrix-layout-row-major

// Test that we apply matrix layout rules correctly.

cbuffer A
{
					float3x4 aa;
	row_major 		float3x4 ab;
	column_major 	float3x4 ac;
}

struct SB
{
					float3x4 ba;
	row_major 		float3x4 bb;
	column_major 	float3x4 bc;
};

cbuffer B
{
	SB b;
}

float4 main() : SV_Target
{
	return 0.0;
}