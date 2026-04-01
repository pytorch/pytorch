// rw-buffer.slang.glsl
//TEST_IGNORE_FILE:

#version 450
layout(row_major) uniform;
layout(row_major) buffer;

layout(r32f)
layout(binding = 0)
uniform imageBuffer buffer_0;

layout(location = 0)
out vec4 _S1;

layout(location = 0)
in float _S2;

flat layout(location = 1)
in int _S3;

void main()
{
	imageStore(buffer_0, int(uint(_S3)), vec4(_S2, float(0), float(0), float(0)));
    _S1 = vec4(_S2);
    return;
}
