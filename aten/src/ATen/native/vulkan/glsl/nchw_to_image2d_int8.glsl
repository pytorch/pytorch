#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/*
 * Output Image
 */
layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage2D uImage;

/*
 * Input Buffer
 */
layout(set = 0, binding = 1) buffer  PRECISION restrict readonly Buffer {
  int data[];
}
uBuffer;

/*
 * Extends sign of int8
 */
int extend_sign(int x) {
  if (x >> 7 == 1) {
    return x | 0xFFFFFF00;
  }
  return x;
}

/*
 * Params Buffer
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // xyz contain the extents of the output texture, w contains HxW to help
  // calculate buffer offsets
  ivec4 out_extents;
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

  const int base_index =
      pos.x + uBlock.out_extents.x * pos.y + (4 * uBlock.out_extents.w) * pos.z;
  const ivec4 buf_indices =
      base_index + ivec4(0, 1, 2, 3) * uBlock.out_extents.w;

  int shift = (1 << 8) - 1;
  ivec4 masks;
  masks.x = shift << 8 * (buf_indices.x % 4);
  masks.y = shift << 8 * (buf_indices.y % 4);
  masks.z = shift << 8 * (buf_indices.z % 4);
  masks.w = shift << 8 * (buf_indices.w % 4);

  int buf_in_1 = uBuffer.data[buf_indices.x / 4];
  int val_x = (buf_in_1 & masks.x) >> 8 * (buf_indices.x % 4);
  val_x = extend_sign(val_x);

  int buf_in_2 = uBuffer.data[buf_indices.y / 4];
  int val_y = (buf_in_2 & masks.y) >> 8 * (buf_indices.y % 4);
  val_y = extend_sign(val_y);

  int buf_in_3 = uBuffer.data[buf_indices.z / 4];
  int val_z = (buf_in_3 & masks.z) >> 8 * (buf_indices.z % 4);
  val_z = extend_sign(val_z);

  int buf_in_4 = uBuffer.data[buf_indices.w / 4];
  int val_w = (buf_in_4 & masks.w) >> 8 * (buf_indices.w % 4);
  val_w = extend_sign(val_w);

  imageStore(uImage, pos.xy, ivec4(val_x, val_y, val_z, val_w));
}
