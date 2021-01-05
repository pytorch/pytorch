#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict writeonly image2D uImage;
layout(set = 0, binding = 1)          buffer  PRECISION restrict readonly  Buffer {
  float data[];
} uBuffer;
layout(set = 0, binding = 2)          uniform PRECISION restrict           Block {
  ivec4 size;
  ivec4 offset;
  int stride;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

  if (all(lessThan(pos, uBlock.size.xy))) {
    const int base_x = uBlock.stride*pos.x;
    const int base = base_x + uBlock.size.w * pos.y;
    const ivec4 index = base + uBlock.offset;

    const vec4 mask = vec4(lessThan(vec4(base_x, base_x+1, base_x, base_x+1), vec4(uBlock.size.w/2)));
    const vec4 outvec = vec4(
        uBuffer.data[index.x],
        uBuffer.data[index.y],
        uBuffer.data[index.z],
        uBuffer.data[index.w]);
    imageStore(
        uImage,
        pos,
        mask*outvec);
  }
}
