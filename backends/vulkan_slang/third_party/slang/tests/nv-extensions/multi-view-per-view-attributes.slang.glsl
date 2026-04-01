#version 450
//TEST_IGNORE_FILE:

#extension GL_NVX_multiview_per_view_attributes : require

struct VS_OUT
{
	vec4  left;
    vec4  right;
    uvec4 mask;
};

VS_OUT main_(vec4 ll, vec4 rr)
{
	VS_OUT res;
	res.left = ll;
	res.right = rr;
	res.mask = uvec4(0x1);
	return res;
}

layout(location = 0)
in vec4 SLANG_in_ll;

layout(location = 1)
in vec4 SLANG_in_rr;

void main()
{
	vec4 ll = SLANG_in_ll;
	vec4 rr = SLANG_in_rr;

	VS_OUT main_result = main_(ll, rr);

	uvec4 SLANG_tmp_0 = main_result.mask;

	gl_Position = main_result.left;
	gl_PositionPerViewNV[1] = main_result.right;
	gl_ViewportMaskPerViewNV[0] = int(SLANG_tmp_0.x);
	gl_ViewportMaskPerViewNV[1] = int(SLANG_tmp_0.y);
	gl_ViewportMaskPerViewNV[2] = int(SLANG_tmp_0.z);
	gl_ViewportMaskPerViewNV[3] = int(SLANG_tmp_0.w);
	gl_PositionPerViewNV[0] = gl_Position;
}
