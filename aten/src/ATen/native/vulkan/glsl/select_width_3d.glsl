#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // width_info.x: output texture x extent
  // width_info.y: output texture y extent
  // width_info.z: output texture z extent
  // width_info.w: output texture w extent
  ivec4 width_info;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // w
  const int src_x = uBlock.width_info.w;
  // h
  const int src_y = pos.x;
  // c
  const int src_z = pos.y;

  const vec4 v = texelFetch(uInput, ivec3(src_x, src_y, src_z), 0);

  for (int i = 0; i < 4; i++) {
    ivec3 new_pos = ivec3(pos.x, pos.y * 4 + i, 0);

    // When the C-channel exceeds original block size, exit early
    if (new_pos.y >= uBlock.width_info.y) {
      return;
    }

    imageStore(uOutput, new_pos, vec4(v[i], 0, 0, 0));
  }
}
