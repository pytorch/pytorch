#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/*
 * Output Image
 */
layout(set = 0, binding = 0, rgba32i) uniform PRECISION restrict writeonly iimage2D uImage;

/*
 * Input Buffer
 */
layout(set = 0, binding = 1) buffer  PRECISION restrict readonly Buffer {
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

  int val_x = uBuffer.data[buf_indices.x];
  int val_y = uBuffer.data[buf_indices.y];
  int val_z = uBuffer.data[buf_indices.z];
  int val_w = uBuffer.data[buf_indices.w];

  imageStore(uImage, pos.xy, ivec4(val_x, val_y, val_z, val_w));
}
