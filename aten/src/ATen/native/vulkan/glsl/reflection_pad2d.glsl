#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  ivec4 size;
  ivec4 pad;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec2 zeros = ivec2(0, 0);
    const ivec2 off_pre  = 2*max(uBlock.pad.xz - pos.xy, zeros);
    const ivec2 off_post = 2*max(pos.xy - (uBlock.size.xy - ivec2(1, 1) - uBlock.pad.yw), zeros);

    const ivec3 inpos = ivec3(pos.xy - uBlock.pad.xz + off_pre - off_post, pos.z);
    imageStore(uOutput, pos, texelFetch(uInput, inpos, 0));
  }
}
