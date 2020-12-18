#version 450 core
#define PRECISION $precision

layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict writeonly image2D uImage;
layout(set = 0, binding = 1)          buffer  PRECISION restrict readonly  Buffer {
  float data[];
} uBuffer;
layout(set = 0, binding = 2)          uniform PRECISION restrict           Block {
  ivec4 size;
  ivec4 offset;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

  if (all(lessThan(pos, uBlock.size.xy))) {
    const int base = 2*pos.x + uBlock.size.w * pos.y;
    const ivec4 index = base + uBlock.offset;

    imageStore(
        uImage,
        pos,
        vec4(
            uBuffer.data[index.x],
            uBuffer.data[index.y],
            uBuffer.data[index.z],
            uBuffer.data[index.w]));
  }
}
