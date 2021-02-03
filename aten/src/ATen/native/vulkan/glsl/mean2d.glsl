#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1) uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict           Block {
  ivec4 size;
  ivec2 isize;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// This implementation is suboptimal and should be revisted.

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    vec4 sum = vec4(0);

    const int z = pos.x + uBlock.size.x * pos.y;
    const int zi = z / 4;
    const int zo = z % 4;

    for (int y = 0; y < uBlock.isize.y; ++y) {
      for (int x = 0; x < uBlock.isize.x; ++x) {
        sum += texelFetch(uInput, ivec3(x, y, zi), 0);
      }
    }

    imageStore(
        uOutput,
        pos,
        vec4(sum[zo], 0, 0, 0) / uBlock.size.w);
  }
}
