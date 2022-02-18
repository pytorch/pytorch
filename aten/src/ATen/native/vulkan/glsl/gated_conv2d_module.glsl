#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uPadding;
layout(set = 0, binding = 2)         uniform PRECISION                    sampler3D uPrevOut;
layout(set = 0, binding = 3)         uniform PRECISION                    sampler3D uKernel;
layout(set = 0, binding = 4)         uniform PRECISION                    sampler3D uBias;
layout(set = 0, binding = 5)         uniform PRECISION restrict           Block {
  ivec4 size;
  ivec4 kernel;
  ivec2 ikernel;
  ivec2 stride;
  ivec2 padding;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const int start_x = pos.x * uBlock.stride.x;
    const int end_x = min(start_x + uBlock.kernel.x, uBlock.kernel.z);
    ivec2 kstart = ivec2(0.0, 2 * pos.z * uBlock.ikernel.y);

    vec4 sum1 = texelFetch(uBias, ivec3(pos.z, 0, 0), 0);
    vec4 sum2 = texelFetch(uBias, ivec3(pos.z, 1, 0), 0);

    for (int z4 = 0; z4 < uBlock.size.w/4; ++z4, kstart.x += uBlock.ikernel.x*4) {
      int ky = kstart.y;
      for (int x = start_x, kx = kstart.x; x < end_x; ++x, kx += 4) {
        vec4 In = texelFetch(uPadding, ivec3(x, 0, z4), 0);
        const ivec4 kxs = kx + ivec4(0, 1, 2, 3);

        sum1 = fma(In.xxxx, texelFetch(uKernel, ivec3(kxs.x, ky, 0), 0), sum1);
        sum1 = fma(In.yyyy, texelFetch(uKernel, ivec3(kxs.y, ky, 0), 0), sum1);
        sum1 = fma(In.zzzz, texelFetch(uKernel, ivec3(kxs.z, ky, 0), 0), sum1);
        sum1 = fma(In.wwww, texelFetch(uKernel, ivec3(kxs.w, ky, 0), 0), sum1);

        sum2 = fma(In.xxxx, texelFetch(uKernel, ivec3(kxs.x, ky+1, 0), 0), sum2);
        sum2 = fma(In.yyyy, texelFetch(uKernel, ivec3(kxs.y, ky+1, 0), 0), sum2);
        sum2 = fma(In.zzzz, texelFetch(uKernel, ivec3(kxs.z, ky+1, 0), 0), sum2);
        sum2 = fma(In.wwww, texelFetch(uKernel, ivec3(kxs.w, ky+1, 0), 0), sum2);
      }
      ky += 2;
      for (int x = start_x, kx = kstart.x; x < end_x; ++x, kx += 4) {
        vec4 In = texelFetch(uPrevOut, ivec3(x, 0, z4), 0);
        const ivec4 kxs = kx + ivec4(0, 1, 2, 3);

        sum1 = fma(In.xxxx, texelFetch(uKernel, ivec3(kxs.x, ky, 0), 0), sum1);
        sum1 = fma(In.yyyy, texelFetch(uKernel, ivec3(kxs.y, ky, 0), 0), sum1);
        sum1 = fma(In.zzzz, texelFetch(uKernel, ivec3(kxs.z, ky, 0), 0), sum1);
        sum1 = fma(In.wwww, texelFetch(uKernel, ivec3(kxs.w, ky, 0), 0), sum1);

        sum2 = fma(In.xxxx, texelFetch(uKernel, ivec3(kxs.x, ky+1, 0), 0), sum2);
        sum2 = fma(In.yyyy, texelFetch(uKernel, ivec3(kxs.y, ky+1, 0), 0), sum2);
        sum2 = fma(In.zzzz, texelFetch(uKernel, ivec3(kxs.z, ky+1, 0), 0), sum2);
        sum2 = fma(In.wwww, texelFetch(uKernel, ivec3(kxs.w, ky+1, 0), 0), sum2);
      }
    }

    vec4 outtex = sum1 * (1/(1+exp(-1*sum2)));

    imageStore(uOutput, pos, outtex);
  }
}
