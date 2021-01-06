#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION                    sampler2D uImage;
layout(set = 0, binding = 1) buffer  PRECISION restrict writeonly Buffer {
  float data[];
} uBuffer;
layout(set = 0, binding = 2) uniform PRECISION restrict           Block {
  ivec4 size;
  ivec2 orig_size;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

  if (all(lessThan(pos, uBlock.size.xy))) {
    const int base_x = 2*pos.x;
    const int base_y = 2*pos.y;
    const vec4 texel = texelFetch(uImage, pos, 0);

    const int base = base_x + uBlock.orig_size.x * base_y;
    const ivec4 index = base + ivec4(0,1,uBlock.orig_size.x, uBlock.orig_size.x+1);

    if (base_x < uBlock.orig_size.x && base_y < uBlock.orig_size.y) {
      uBuffer.data[index.x] = texel.x;
    }
    if (base_x+1 < uBlock.orig_size.x && base_y < uBlock.orig_size.y) {
      uBuffer.data[index.y] = texel.y;
    }
    if (base_x < uBlock.orig_size.x && base_y+1 < uBlock.orig_size.y) {
      uBuffer.data[index.z] = texel.z;
    }
    if (base_x+1 < uBlock.orig_size.x && base_y+1 < uBlock.orig_size.y) {
      uBuffer.data[index.w] = texel.w;
    }
  }
}
