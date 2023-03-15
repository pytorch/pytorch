#version 450 core
#define PRECISION $precision
#define FORMAT $format

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
  // input tensor's batch size
  uint batch_size;
  // input tensor's channel size
  uint ch_size;
  // channel interval (total # of channels for all tensors)
  uint ch_interval;
  // # of channels for tensor 0 to i-1 at ith tensor
  uint ch_size_allprior;
}
uBlock;

/*
 * Local Work Group
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 in_pos = ivec3(gl_GlobalInvocationID);
  const uint max_src_index = uBlock.ch_size * uBlock.batch_size;

  if (any(greaterThanEqual(in_pos, uBlock.in_extents.xyz))) {
    return;
  }

  // x and y don't change. only z and index matter
  ivec3 out_pos = in_pos;
  const vec4 in_tex = texelFetch(uInput, in_pos, 0);

  for (uint i = 0; i < 4; ++i) {
    uint src_index = in_pos.z * 4 + i;

    if (src_index >= max_src_index) {
      // out of range
      break;
    }

    uint src_n_idx = src_index / uBlock.ch_size;
    uint src_c_idx = src_index % uBlock.ch_size;

    uint dst_nc_idx =
        src_n_idx * uBlock.ch_interval + src_c_idx + uBlock.ch_size_allprior;

    out_pos.z = int(dst_nc_idx / 4);
    uint j = (dst_nc_idx % 4);

    vec4 out_tex = imageLoad(uOutput, out_pos);
    out_tex[j] = in_tex[i];
    imageStore(uOutput, out_pos, out_tex);
  }
}
