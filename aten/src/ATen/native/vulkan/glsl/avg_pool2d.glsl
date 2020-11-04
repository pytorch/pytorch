#version 450 core
#define PRECISION $precision

layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)          uniform PRECISION restrict           Block {
  ivec2 kernel;
  ivec2 stride;
  ivec2 padding;
} uBlock;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const ivec2 isize = textureSize(uInput, 0).xy;
  const ivec2 size = imageSize(uOutput).xy;
  const float range = uBlock.kernel.x * uBlock.kernel.y;

  if (all(lessThan(pos.xy, size))) {
    const ivec2 ipos = pos.xy * uBlock.stride - uBlock.padding;

    const ivec2 start = max(ivec2(0), ipos);
    const ivec2 end = min(ipos + uBlock.kernel, isize);

    vec4 acc = vec4(0);

    for (int y = start.y; y < end.y; ++y) {
      for (int x = start.x; x < end.x; ++x) {
        acc += texelFetch(uInput, ivec3(x, y, pos.z), 0);
      }
    }

    imageStore(
        uOutput,
        pos,
        acc / range);
  }
}
