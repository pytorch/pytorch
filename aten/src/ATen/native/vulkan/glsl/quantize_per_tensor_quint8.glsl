#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba8ui) uniform PRECISION restrict writeonly uimage3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION                    sampler3D  uInput; //input
layout(set = 0, binding = 2)          uniform PRECISION restrict           Block {
  ivec4 size;
  vec2 scale;
  ivec2 zero_point;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.size.xyz))) {
    vec4 q_res = roundEven(texelFetch(uInput, pos, 0) / uBlock.scale.x) + uBlock.zero_point.x;

    uvec4 ret = uvec4(q_res);

    imageStore(
        uOutput,
        pos,
        ret);
  }
}
