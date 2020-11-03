#version 450 core
#define PRECISION $precision

layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict writeonly image3D uImage;
layout(set = 0, binding = 1)          buffer  PRECISION restrict readonly  Buffer {
  float data[];
} uBuffer;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec2 size = imageSize(uImage).xy;

  if (all(lessThan(pos.xy, size))) {
    const int plane = size.x * size.y;
    const int base = pos.x + size.x * pos.y + 4 * plane * pos.z;
    const ivec4 index = base + plane * ivec4(0, 1, 2, 3);

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
