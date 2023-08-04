#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/*
 * Input Sampler
 */
layout(set = 0, binding = 0) uniform PRECISION sampler3D uImage;

/*
 * Output Buffer
 */
layout(set = 0, binding = 1) buffer PRECISION restrict writeonly Buffer {
  float data[];
}
uBuffer;

/*
 * Params Buffer
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // xyz contain the extents of the input texture, w contains HxW to help
  // calculate buffer offsets
  ivec4 in_extents;
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

  if (any(greaterThanEqual(pos, uBlock.in_extents.xyz))) {
    return;
  }

  const vec4 intex = texelFetch(uImage, pos, 0);

  const int n_index = int(pos.z / uBlock.c_info.x);
  const int c_index = (pos.z % uBlock.c_info.x) * 4;
  int d_offset = (n_index * uBlock.c_info.y) + c_index;

  const int base_index =
      pos.x + uBlock.in_extents.x * pos.y + uBlock.in_extents.w * d_offset;
  const ivec4 buf_indices =
      base_index + ivec4(0, 1, 2, 3) * uBlock.in_extents.w;

  if (c_index < uBlock.c_info.y) {
    uBuffer.data[buf_indices.x] = intex.x;
  }
  if (c_index + 1 < uBlock.c_info.y) {
    uBuffer.data[buf_indices.y] = intex.y;
  }
  if (c_index + 2 < uBlock.c_info.y) {
    uBuffer.data[buf_indices.z] = intex.z;
  }
  if (c_index + 3 < uBlock.c_info.y) {
    uBuffer.data[buf_indices.w] = intex.w;
  }
}
