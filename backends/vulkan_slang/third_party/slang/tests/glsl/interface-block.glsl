//TEST:SIMPLE(filecheck=CHECK_GLSL): -target glsl -stage fragment -entry fragmentMain -allow-glsl

//CHECK_GLSL: layout(location = 0)
//CHECK_GLSL: out vec4 entryPointParam_fragmentMain_out1_0;

//CHECK_GLSL: layout(location = 0)
//CHECK_GLSL: in vec2 vd_texcoord_0_0;

//CHECK_GLSL: layout(location = 1)
//CHECK_GLSL: in vec2 vd_texcoord_1_0;

//CHECK_GLSL: layout(location = 2)
//CHECK_GLSL: in vec4 vd_inner_texcoord_2_0;

import glsl;

#version 400

struct innerData
{
    vec4 texcoord_2;
};

in VertexData
{
    vec2 texcoord_0;
    vec2 texcoord_1;
    innerData inner;
} vd;

out vec4 out1;

void fragmentMain()
{
    out1 = vec4(vd.texcoord_0, vd.texcoord_1.x, vd.inner.texcoord_2.y);
}

