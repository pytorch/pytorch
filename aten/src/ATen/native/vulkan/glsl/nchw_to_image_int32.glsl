#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

/*
 * Output Image
 */
layout(set = 0, binding = 0, rgba32i) uniform PRECISION restrict writeonly iimage3D uImage;

/*
 * Input Buffer
 */
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Buffer {
  int data[];
}
uBuffer;

/*
 * Params Buffer
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // xyz contain the extents of the output texture, w contains HxW to help
  // calculate buffer offsets
  ivec4 out_extents;
  // x: number of texels spanned by one channel
  // y: number of channels
  ivec2 c_info;
}
uBlock;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, uBlock.out_extents.xyz))) {
    return;
  }

  const int n_index = int(pos.z / uBlock.c_info.x);
  const int c_index = (pos.z % uBlock.c_info.x) * 4;
  int d_offset = (n_index * uBlock.c_info.y) + c_index;

  const int base_index =
      pos.x + uBlock.out_extents.x * pos.y + uBlock.out_extents.w * d_offset;
  const ivec4 buf_indices =
      base_index + ivec4(0, 1, 2, 3) * uBlock.out_extents.w;

  int val_x = uBuffer.data[buf_indices.x];
  int val_y = uBuffer.data[buf_indices.y];
  int val_z = uBuffer.data[buf_indices.z];
  int val_w = uBuffer.data[buf_indices.w];

  ivec4 texel = ivec4(val_x, val_y, val_z, val_w);

  if (c_index + 3 >= uBlock.c_info.y) {
    ivec4 c_ind = ivec4(c_index) + ivec4(0, 1, 2, 3);
    ivec4 valid_c = ivec4(lessThan(c_ind, ivec4(uBlock.c_info.y)));
    texel = texel * valid_c;
  }

  imageStore(uImage, pos, texel);
}
