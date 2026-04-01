// explicit-register-space.slang
//TEST:REFLECTION:-profile ps_5_1 -target hlsl -no-codegen

// Confirm that we handle explicit register spaces
// on global shader parameters.

Texture2D tx : register(t1, space2);

float4 main() : SV_Target
{
	return 0.0;
}