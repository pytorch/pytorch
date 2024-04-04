#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  ivec4 size;
  /* pad: {left, right, top, bottom} */
  ivec4 pad;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec3 corresponding_input_pos = ivec3(
      clamp(pos.xy - uBlock.pad.xz,
        ivec2(0, 0),
        uBlock.size.xy - uBlock.pad.xz - uBlock.pad.yw - ivec2(1, 1)),
      pos.z);

    imageStore(uOutput, pos, texelFetch(uInput, corresponding_input_pos, 0));
  }
}
