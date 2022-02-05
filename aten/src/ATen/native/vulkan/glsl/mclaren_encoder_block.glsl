#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput1;
layout(set = 0, binding = 2)         uniform PRECISION                    sampler3D uInput2;
layout(set = 0, binding = 3)         uniform PRECISION                    sampler3D uKernel1;
layout(set = 0, binding = 4)         uniform PRECISION                    sampler3D uBias1;
layout(set = 0, binding = 5)         uniform PRECISION                    sampler3D uKernel2;
layout(set = 0, binding = 6)         uniform PRECISION                    sampler3D uBias2;
layout(set = 0, binding = 7)         uniform PRECISION restrict           Block {
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

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec2 ipos = pos.xy * uBlock.stride - uBlock.padding;

    const ivec2 start = max(ivec2(0), ipos);
    const ivec2 end = min(ipos + uBlock.kernel.xy, uBlock.kernel.zw);
    ivec2 kstart = (start - ipos) / uBlock.dilate;

    kstart.x *= 4;
    kstart.y += pos.z * uBlock.ikernel.y;

    vec4 sum1 = texelFetch(uBias1, ivec3(pos.z, 0, 0), 0);
    vec4 sum2 = texelFetch(uBias2, ivec3(pos.z, 0, 0), 0);

    for (int z4 = 0; z4 < uBlock.size.w/4; ++z4, kstart.x += uBlock.ikernel.x*4) {
      for (int y = start.y, ky = kstart.y; y < end.y; y += uBlock.dilate.y, ++ky) {
        for (int x = start.x, kx = kstart.x; x < end.x; x += uBlock.dilate.x, kx += 4) {
          vec4 In = (y == 0) ? texelFetch(uInput1, ivec3(x, 0, z4), 0)
                             : texelFetch(uInput2, ivec3(x, 0, z4), 0);
          const ivec4 kxs = kx + ivec4(0, 1, 2, 3);

          sum1 = fma(In.xxxx, texelFetch(uKernel1, ivec3(kxs.x, ky, 0), 0), sum1);
          sum1 = fma(In.yyyy, texelFetch(uKernel1, ivec3(kxs.y, ky, 0), 0), sum1);
          sum1 = fma(In.zzzz, texelFetch(uKernel1, ivec3(kxs.z, ky, 0), 0), sum1);
          sum1 = fma(In.wwww, texelFetch(uKernel1, ivec3(kxs.w, ky, 0), 0), sum1);

          sum2 = fma(In.xxxx, texelFetch(uKernel2, ivec3(kxs.x, ky, 0), 0), sum2);
          sum2 = fma(In.yyyy, texelFetch(uKernel2, ivec3(kxs.y, ky, 0), 0), sum2);
          sum2 = fma(In.zzzz, texelFetch(uKernel2, ivec3(kxs.z, ky, 0), 0), sum2);
          sum2 = fma(In.wwww, texelFetch(uKernel2, ivec3(kxs.w, ky, 0), 0), sum2);
        }
      }
    }

    vec4 outtex = sum1 * (1/(1+exp(-1*sum2)));
    //vec4 outtex = sum1 * sum2;

    imageStore(uOutput, pos, outtex);
  }
}
