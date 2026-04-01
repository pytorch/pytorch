//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage vertex -entry main -allow-glsl -emit-spirv-directly
//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage vertex -entry main -allow-glsl

#version 450 core
layout(location = 0) in highp vec4 a_position;
layout(location = 4) in highp vec2 a_in0;
layout(location = 5) in highp float a_in1;
layout(location = 0) out highp vec2 v_texCoord;
layout(location = 1) out highp float v_lodBias;
out gl_PerVertex {
	vec4 gl_Position;
};

void main()
{
	gl_Position = a_position;
	v_texCoord = a_in0;
    v_lodBias = a_in1;
    // CHECK: OpEntryPoint
}

