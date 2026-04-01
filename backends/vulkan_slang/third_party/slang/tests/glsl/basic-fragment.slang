//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage fragment -entry main -allow-glsl
#version 310 es
precision highp float;
precision highp int;

bool isOk (uint a, uint b)   { return (a == b); }
layout(location = 0) out mediump vec4 dEQP_FragColor;
layout(location = 0) flat in float in0;
layout(binding = 0, std140) uniform Reference
{
	uint out0;
} ref;
uint out0;

void main()
{
	out0 = uint(in0);
	bool RES = isOk(out0, ref.out0);
    dEQP_FragColor = vec4(RES, RES, RES, 1.0);
    // CHECK: OpEntryPoint
}

