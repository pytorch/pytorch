#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba8ui) uniform PRECISION restrict writeonly uimage3D uImage;
layout(set = 0, binding = 1)         buffer  PRECISION restrict readonly  Buffer {
  uint data[];
} uBuffer;
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  ivec4 size;
  ivec4 offset;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.size.xyz))) {
    const int base = pos.x + uBlock.size.x * pos.y + uBlock.size.w * pos.z;
    const ivec4 index = base + uBlock.offset;

    int shift = (1 << 8) - 1;
    ivec4 masks;
    masks.x = shift << 8 * (index.x % 4);
    masks.y = shift << 8 * (index.y % 4);
    masks.z = shift << 8 * (index.z % 4);
    masks.w = shift << 8 * (index.w % 4);

    uint buf_in_1 = uBuffer.data[index.x / 4];
    uint a_v = (buf_in_1 & masks.x) >> 8 * (index.x % 4);

    uint buf_in_2 = uBuffer.data[index.y / 4];
    uint b_v = (buf_in_2 & masks.y) >> 8 * (index.y % 4);

    uint buf_in_3 = uBuffer.data[index.z / 4];
    uint g_v = (buf_in_3 & masks.z) >> 8 * (index.z % 4);

    uint buf_in_4 = uBuffer.data[index.w / 4];
    uint r_v = (buf_in_4 & masks.w) >> 8 * (index.w % 4);

    uvec4 texel = uvec4(a_v, b_v, g_v, r_v);

    imageStore(
        uImage,
        pos,
        texel);
  }
}
