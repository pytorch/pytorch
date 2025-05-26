#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // x=width, y=height, z=channel, w=batch
  uvec4 extents;
  // x=width, y=height, z=channel, w=batch
  // 1=flip, 0=noflip
  ivec4 dims;
}
uBlock;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Returns a new tensor with values flipped along dimension dim
 */

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  int flattened_channels = int(ceil(uBlock.extents.z / 4.0));
  vec4 out_texel = vec4(0, 0, 0, 0);
  uint src_x = pos.x;
  uint src_y = pos.y;
  uint src_z = pos.z;

  // Width
  if (uBlock.dims.x == 1) {
    src_x = uBlock.extents.x - 1 - pos.x;
  }

  // Height
  if (uBlock.dims.y == 1) {
    src_y = uBlock.extents.y - 1 - pos.y;
  }

  // Batch
  if (uBlock.dims.w == 1) {
    uint n = pos.z / flattened_channels;
    uint src_n = uBlock.extents.w - 1 - n;
    uint c_div4 = pos.z - n * flattened_channels;
    src_z = src_n * flattened_channels + c_div4;
  }

  uint prev_src_z = src_z; // save this
  for (int p = 0; p < 4; p++) {
    uint src_p = p;

    // Channel
    if (uBlock.dims.z == 1) {
      // n * [C/4]
      uint nc = (pos.z / flattened_channels) * flattened_channels;
      // i / 4
      uint c_div4 = pos.z - nc;
      uint c = c_div4 * 4 + p;
      uint src_c = uBlock.extents.z - 1 - c;

      src_z = (uBlock.dims.w == 1)
          ? prev_src_z - c_div4 + src_c / 4 // Batch and Channel
          : nc + src_c / 4; // Channel only
      src_p = src_c % 4;
    }

    vec4 v = texelFetch(uInput, ivec3(src_x, src_y, src_z), 0);
    out_texel[p] = v[src_p];
    imageStore(uOutput, pos, out_texel);
  }
}
