#version 450 core
#define PRECISION $precision
#define FORMAT $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

/*
 * Output Image
 */
layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uImage;

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

/*
 * Extends sign of int8
 */
int extend_sign(int x) {
  if (x >> 7 == 1) {
    return x | 0xFFFFFF00;
  }
  return x;
}

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

  int shift = (1 << 8) - 1;
  ivec4 masks;
  masks.x = shift << 8 * (buf_indices.x % 4);
  masks.y = shift << 8 * (buf_indices.y % 4);
  masks.z = shift << 8 * (buf_indices.z % 4);
  masks.w = shift << 8 * (buf_indices.w % 4);

  int buf_in_1 = uBuffer.data[buf_indices.x / 4];
  int a_v = (buf_in_1 & masks.x) >> 8 * (buf_indices.x % 4);
  a_v = extend_sign(a_v);

  int buf_in_2 = uBuffer.data[buf_indices.y / 4];
  int b_v = (buf_in_2 & masks.y) >> 8 * (buf_indices.y % 4);
  b_v = extend_sign(b_v);

  int buf_in_3 = uBuffer.data[buf_indices.z / 4];
  int g_v = (buf_in_3 & masks.z) >> 8 * (buf_indices.z % 4);
  g_v = extend_sign(g_v);

  int buf_in_4 = uBuffer.data[buf_indices.w / 4];
  int r_v = (buf_in_4 & masks.w) >> 8 * (buf_indices.w % 4);
  r_v = extend_sign(r_v);

  ivec4 texel = ivec4(a_v, b_v, g_v, r_v);

  if (c_index + 3 >= uBlock.c_info.y) {
    ivec4 c_ind = ivec4(c_index) + ivec4(0, 1, 2, 3);
    ivec4 valid_c = ivec4(lessThan(c_ind, ivec4(uBlock.c_info.y)));
    texel = texel * valid_c;
  }

  imageStore(uImage, pos, texel);
}
