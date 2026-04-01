//TEST:REFLECTION:-profile ps_4_0  -target hlsl -no-codegen

// Confirm that shader parameters in imported modules get reflected properly.

__import reflect_imported_code;

Texture2D 		t;
SamplerState 	s;

cbuffer C
{
	float c;
}

float4 main() : SV_Target
{
	return use(t,s_i)
	     + use(c)
	     + use(t_i, s)
	     + use(c_i);
}