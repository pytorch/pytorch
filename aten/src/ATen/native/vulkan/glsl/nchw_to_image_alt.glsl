#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba8ui) uniform PRECISION restrict writeonly uimage3D uImage;
layout(set = 0, binding = 1)         buffer  PRECISION restrict readonly  Buffer {
  int data[];
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
    //const ivec4 index = base + uBlock.offset;

    vec4 unpacked_value = unpackUnorm4x8(uBuffer.data[base]);

    imageStore(
        uImage,
        pos,
        uvec4(unpacked_value));
  }
}
