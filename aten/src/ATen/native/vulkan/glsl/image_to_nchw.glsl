#version 450 core
#define PRECISION $precision

layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION                    sampler3D uImage;
layout(set = 0, binding = 1) buffer  PRECISION restrict writeonly Buffer {
  float data[];
} uBuffer;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  /* Dynamically Uniform */
  const ivec3 size = textureSize(uImage, 0);
  const int plane = size.x * size.y;
  const int block = 4 * plane;
  const ivec4 offset = plane * ivec4(0, 1, 2, 3);

  if (all(lessThan(pos, size))) {
    const vec4 texel = texelFetch(uImage, pos, 0);

    const int base = pos.x + size.x * pos.y + block * pos.z;
    const ivec4 index = base + offset;

    uBuffer.data[index.x] = texel.r;
    uBuffer.data[index.y] = texel.g;
    uBuffer.data[index.z] = texel.b;
    uBuffer.data[index.w] = texel.a;
  }
}
