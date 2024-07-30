#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/*
 * Output Image
 */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION image3D uOutput;

/*
 * Input Textures
 */
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;

/*
 * Params Buffer
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // output texture size (x=width,y=height,z=depth,w=unused)
  ivec4 out_extents;
  // input texture size (x=width,y=height,z=depth,w=unused)
  ivec4 in_extents;
  // x: size of output channel dim (values)
  // y: size of output channel dim (texels)
  uvec2 out_ch_info;
  // x: size of input channel dim
  // y: size of input channel dim up-aligned to 4
  uvec2 in_ch_info;
  // x: total number of channel values already appended
  // y: offset to first channel texel being operated on
  // z: number of channel texels being operated on
  // w: padding
  uvec4 appended_ch_info;
}
uBlock;

/*
 * Local Work Group
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, uBlock.out_extents.xyz))) {
    return;
  }

  // Determine the N, C indices that this invocation is writing to
  const uint dst_n_idx = pos.z / uBlock.appended_ch_info.z;
  const uint dst_c4_idx = (pos.z % uBlock.appended_ch_info.z) + uBlock.appended_ch_info.y;
  uint dst_c_idx = dst_c4_idx * 4;

  // Reconstruct the output write position based on the N, C indices
  const uint dst_z_idx = dst_n_idx * uBlock.out_ch_info.y + dst_c4_idx;
  const ivec3 write_pos = ivec3(pos.xy, dst_z_idx);

  vec4 out_tex = imageLoad(uOutput, write_pos);

  const uint src_n_offset = dst_n_idx * uBlock.in_ch_info.y;

  uint dst_nc_idx = dst_z_idx * 4;
  int src_c_idx = int(dst_c_idx - uBlock.appended_ch_info.x);
  int src_nc_idx = int(src_n_offset) + src_c_idx;

  // For each element of the output, extract the corresponding value from the
  // input
  for (uint i = 0; i < 4; ++i, ++dst_c_idx, ++dst_nc_idx, ++src_c_idx, ++src_nc_idx) {
    if (src_c_idx >= 0) {
      uint src_z_idx = src_nc_idx / 4;

      vec4 in_tex = texelFetch(uInput, ivec3(pos.xy, src_z_idx), 0);

      uint src_offset = src_nc_idx % 4;
      uint dst_offset = dst_nc_idx % 4;

      if (src_c_idx < uBlock.in_ch_info.x) {
        out_tex[i] = in_tex[src_offset];
      } else {
        out_tex[i] = 1.234;
      }
    }
  }

  imageStore(uOutput, write_pos, out_tex);
}
