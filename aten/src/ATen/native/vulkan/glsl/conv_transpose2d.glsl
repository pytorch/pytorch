#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION                    sampler3D uKernel;
layout(set = 0, binding = 3)         uniform PRECISION                    sampler3D uBias;
layout(set = 0, binding = 4)         uniform PRECISION restrict           Block {
  ivec4 size;
  ivec4 kernel;
  ivec2 ikernel;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilate;
  vec2 clamp;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const ivec2 isize = ivec2(uBlock.kernel.zw);
  const vec2 ksize = vec2(uBlock.kernel.xy);
  const vec2 stride = vec2(uBlock.stride);
  const vec2 padding = vec2(uBlock.padding);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    ivec2 ipos = pos.xy + uBlock.padding;
    vec2 ipos_f = vec2(ipos);

    const ivec2 start = max(ivec2(0), ivec2(ceil((ipos_f - ksize + 1)/stride)));
    const ivec2 end = min(isize, ivec2(floor(ipos_f/stride))+1);
    ivec2 kstart = start;

    vec4 sum = texelFetch(uBias, ivec3(pos.z, 0, 0), 0);

    int ky_start = uBlock.kernel.y - 1 - (ipos.y - uBlock.stride.y*start.y) + pos.z * uBlock.ikernel.y;
    int kx_start = (uBlock.kernel.x - 1 - (ipos.x - uBlock.stride.x*start.x)) * uBlock.size.w;
    int kx_stride = uBlock.size.w * (uBlock.stride.x - 1);
    for (int y = start.y, ky = ky_start; y < end.y; ++y, ky += uBlock.stride.y) {
      int kx = kx_start;
      for (int x = start.x, kx = kx_start; x < end.x; ++x, kx += kx_stride) {
        for (int z4 = 0; z4 < uBlock.size.w/4; ++z4, kx += 4) {
          const vec4 In = texelFetch(uInput, ivec3(x, y, z4), 0);
          const ivec4 kxs = kx + ivec4(0, 1, 2, 3);

          sum = fma(In.xxxx, texelFetch(uKernel, ivec3(kxs.x, ky, 0), 0), sum);
          sum = fma(In.yyyy, texelFetch(uKernel, ivec3(kxs.y, ky, 0), 0), sum);
          sum = fma(In.zzzz, texelFetch(uKernel, ivec3(kxs.z, ky, 0), 0), sum);
          sum = fma(In.wwww, texelFetch(uKernel, ivec3(kxs.w, ky, 0), 0), sum);
        }
      }
    }

    imageStore(
        uOutput,
        pos,
        clamp(sum, uBlock.clamp.x, uBlock.clamp.y));
  }
}
