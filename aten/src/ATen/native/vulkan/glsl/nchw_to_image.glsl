#version 450 core
#define PRECISION $precision

layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba32f) uniform PRECISION restrict writeonly image3D uImage;
layout(set = 0, binding = 1)          buffer  PRECISION restrict readonly  Buffer {
  float data[];
} uBuffer;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  /* Dynamically Uniform */
  const ivec3 size = imageSize(uImage);
  const int plane = size.x * size.y;
  const int block = 4 * plane;
  const ivec4 offset = plane * ivec4(0, 1, 2, 3);

  if (all(lessThan(pos, size))) {
    const int base = pos.x + size.x * pos.y + block * pos.z;
    const ivec4 index = base + offset;

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
