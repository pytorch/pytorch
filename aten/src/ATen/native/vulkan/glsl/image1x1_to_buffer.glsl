#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION                    sampler3D uImage;
layout(set = 0, binding = 1) buffer  PRECISION restrict writeonly Buffer {
  float data[];
} uBuffer;
layout(set = 0, binding = 2) uniform PRECISION restrict           Block {
  ivec3 size;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size))) {
    const vec4 texel = texelFetch(uImage, pos, 0);

    const int base = 4*pos.z;

    uBuffer.data[base+0] = texel.r;
    uBuffer.data[base+1] = texel.g;
    uBuffer.data[base+2] = texel.b;
    uBuffer.data[base+3] = texel.a;
  }
}
