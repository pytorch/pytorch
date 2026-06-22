#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#include "indexing.h"

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;
layout(set = 0, binding = 1)         uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 sizes;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);

  int src_base_w = pos.x * 4;
  int src_h = pos.y;

  // uBlock.sizes.y is the c in nchw.
  int num_c = uBlock.sizes.y;

  int src_c = pos.z % num_c;
  int src_n = pos.z / num_c;

  // Fetch the 4 elements from the channel-packed tensor
  ivec4 src_pos0 = get_channel_packed_pos_from_index(
    ivec4(src_n, src_c, src_h, src_base_w),
    uBlock.sizes);

  ivec4 src_pos1 = get_channel_packed_pos_from_index(
    ivec4(src_n, src_c, src_h, src_base_w + 1),
    uBlock.sizes);

  ivec4 src_pos2 = get_channel_packed_pos_from_index(
    ivec4(src_n, src_c, src_h, src_base_w + 2),
    uBlock.sizes);

  ivec4 src_pos3 = get_channel_packed_pos_from_index(
    ivec4(src_n, src_c, src_h, src_base_w + 3),
    uBlock.sizes);

  vec4 t0 = texelFetch(uInput, src_pos0.xyz, 0);
  vec4 t1 = texelFetch(uInput, src_pos1.xyz, 0);
  vec4 t2 = texelFetch(uInput, src_pos2.xyz, 0);
  vec4 t3 = texelFetch(uInput, src_pos3.xyz, 0);

  vec4 out_t = vec4(
    t0[src_pos0.w],
    t1[src_pos1.w],
    t2[src_pos2.w],
    t3[src_pos3.w]);

  imageStore(uOutput, pos, out_t);
}
