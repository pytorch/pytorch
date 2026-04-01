//TEST_IGNORE_FILE:
#version 420

struct VS_OUT
{
	uint drawID;
};

layout(location = 0)
in flat uint SLANG_in_vsOut_drawID;

layout(location = 0)
out vec4 SLANG_out_main_result;

vec4 main_(VS_OUT vsOut)
{
	return vec4(float(vsOut.drawID));
}

void main()
{
	VS_OUT vsOut;
	vsOut.drawID = SLANG_in_vsOut_drawID;

	vec4 main_result;
	main_result = main_(vsOut);

	SLANG_out_main_result = main_result;
}
