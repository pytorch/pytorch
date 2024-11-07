#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  ivec4 size;
  vec2 params;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    vec4 inval = texelFetch(uInput, pos, 0);
    vec4 mask1 = vec4(greaterThan(inval, vec4(uBlock.params.x)));
    vec4 mask2 = 1.0f - mask1;
    vec4 outval = mask1 * inval + mask2 * uBlock.params.y;
    imageStore(uOutput, pos, outval);
  }
}
