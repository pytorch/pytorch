// precompiled-glsl.slang

// A test that precompiles a slang-module using GLSL functions.

//TEST:COMPILE: tests/library/precompiled-glsl.slang -target spirv -stage fragment -entry main
//TEST:COMPILE: tests/library/precompiled-glsl.slang -target spirv -stage fragment -entry main -embed-downstream-ir

#version 310 es
precision highp float;
precision highp int;

public vec3 func(vec3 v)
{
    return v;
}

layout(location = 0) out mediump vec4 dEQP_FragColor;

layout(location = 0) flat in uint out0;
layout(binding = 0, std140) uniform Reference
{
	uint out0;
} ref;

void main()
{
    dEQP_FragColor = mix(vec4(0.0, 1.0, 0.0, 1.0),
                         vec4(0.0, 1.0, 0.0, 1.0),
                         vec4(0.0, 1.0, 0.0, 1.0));
}
