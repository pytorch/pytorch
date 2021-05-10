#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1) uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict           Block {
  ivec4 size;
  vec2 kernel;
  vec2 stride;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const vec2 ipos = pos.xy * uBlock.stride;

    const ivec2 start = ivec2(ipos);
    const ivec2 end = ivec2(ceil(ipos + uBlock.kernel));
    const ivec2 range = end - start;

    vec4 sum = vec4(0);

    for (int y = start.y; y < end.y; ++y) {
      for (int x = start.x; x < end.x; ++x) {
        sum += texelFetch(uInput, ivec3(x, y, pos.z), 0);
      }
    }

    imageStore(
        uOutput,
        pos,
        sum / (range.x * range.y));
  }
}
