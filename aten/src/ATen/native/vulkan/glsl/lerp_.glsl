#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION          sampler3D uInput0;
layout(set = 0, binding = 2)         uniform PRECISION          sampler3D uInput1;
layout(set = 0, binding = 3)         uniform PRECISION restrict Block {
  ivec4 size;
  ivec4 isize0;
  ivec4 isize1;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec3 input0_pos = pos % uBlock.isize0.xyz;
    const ivec3 input1_pos = pos % uBlock.isize1.xyz;
    imageStore(
        uOutput,
        pos,
        imageLoad(uOutput, pos)
          + texelFetch(uInput1, input1_pos, 0)
          * (texelFetch(uInput0, input0_pos, 0) - imageLoad(uOutput, pos)));
  }
}
