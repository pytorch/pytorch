#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION          sampler3D uInput0;
layout(set = 0, binding = 2)         uniform PRECISION restrict Block {
  ivec4 size;
  ivec3 isize0;
  float weight;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec3 input0_pos = pos % uBlock.isize0.xyz;
    imageStore(
        uOutput,
        pos,
        imageLoad(uOutput, pos)
          + uBlock.weight
          * (texelFetch(uInput0, input0_pos, 0) - imageLoad(uOutput, pos)));
  }
}
