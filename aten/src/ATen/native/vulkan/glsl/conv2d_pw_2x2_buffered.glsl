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

  const ivec3 pos00 = ivec3(pos.x*2  , pos.y*2  , pos.z);
  const ivec3 pos10 = ivec3(pos.x*2+1, pos.y*2  , pos.z);
  const ivec3 pos01 = ivec3(pos.x*2  , pos.y*2+1, pos.z);
  const ivec3 pos11 = ivec3(pos.x*2+1, pos.y*2+1, pos.z);

  if (all(lessThan(pos00, uBlock.size.xyz))) {
    const ivec2 ipos00 = pos00.xy * uBlock.stride - uBlock.padding;
    const ivec2 ipos10 = pos10.xy * uBlock.stride - uBlock.padding;
    const ivec2 ipos01 = pos01.xy * uBlock.stride - uBlock.padding;
    const ivec2 ipos11 = pos11.xy * uBlock.stride - uBlock.padding;

    vec4 sum00 = texelFetch(uBias, ivec3(pos.z, 0, 0), 0);
    vec4 sum10 = sum00;
    vec4 sum01 = sum00;
    vec4 sum11 = sum00;

    vec4 k1_0 = texelFetch(uKernel, ivec3(0, pos.z, 0), 0);
    vec4 k2_0 = texelFetch(uKernel, ivec3(1, pos.z, 0), 0);
    vec4 k3_0 = texelFetch(uKernel, ivec3(2, pos.z, 0), 0);
    vec4 k4_0 = texelFetch(uKernel, ivec3(3, pos.z, 0), 0);

    vec4 k1_1;
    vec4 k2_1;
    vec4 k3_1;
    vec4 k4_1;

    vec4 In00_0 = texelFetch(uInput, ivec3(ipos00, 0), 0);
    vec4 In10_0 = texelFetch(uInput, ivec3(ipos10, 0), 0);
    vec4 In01_0 = texelFetch(uInput, ivec3(ipos01, 0), 0);
    vec4 In11_0 = texelFetch(uInput, ivec3(ipos11, 0), 0);

    vec4 In00_1;
    vec4 In10_1;
    vec4 In01_1;
    vec4 In11_1;

    for (int z = 0, z4 = 0; z < uBlock.size.w; z += 8, z4+=2) {
      ivec4 kxs = z + ivec4(4, 5, 6, 7);

      k1_1 = texelFetch(uKernel, ivec3(kxs.x, pos.z, 0), 0);
      k2_1 = texelFetch(uKernel, ivec3(kxs.y, pos.z, 0), 0);
      k3_1 = texelFetch(uKernel, ivec3(kxs.z, pos.z, 0), 0);
      k4_1 = texelFetch(uKernel, ivec3(kxs.w, pos.z, 0), 0);

      In00_1 = texelFetch(uInput, ivec3(ipos00, z4+1), 0);
      In10_1 = texelFetch(uInput, ivec3(ipos10, z4+1), 0);
      In01_1 = texelFetch(uInput, ivec3(ipos01, z4+1), 0);
      In11_1 = texelFetch(uInput, ivec3(ipos11, z4+1), 0);

      sum00 = fma(In00_0.xxxx, k1_0, sum00);
      sum00 = fma(In00_0.yyyy, k2_0, sum00);
      sum00 = fma(In00_0.zzzz, k3_0, sum00);
      sum00 = fma(In00_0.wwww, k4_0, sum00);

      sum10 = fma(In10_0.xxxx, k1_0, sum10);
      sum10 = fma(In10_0.yyyy, k2_0, sum10);
      sum10 = fma(In10_0.zzzz, k3_0, sum10);
      sum10 = fma(In10_0.wwww, k4_0, sum10);

      sum01 = fma(In01_0.xxxx, k1_0, sum01);
      sum01 = fma(In01_0.yyyy, k2_0, sum01);
      sum01 = fma(In01_0.zzzz, k3_0, sum01);
      sum01 = fma(In01_0.wwww, k4_0, sum01);

      sum11 = fma(In11_0.xxxx, k1_0, sum11);
      sum11 = fma(In11_0.yyyy, k2_0, sum11);
      sum11 = fma(In11_0.zzzz, k3_0, sum11);
      sum11 = fma(In11_0.wwww, k4_0, sum11);

      // Next iteration
      kxs += 4;

      k1_0 = texelFetch(uKernel, ivec3(kxs.x, pos.z, 0), 0);
      k2_0 = texelFetch(uKernel, ivec3(kxs.y, pos.z, 0), 0);
      k3_0 = texelFetch(uKernel, ivec3(kxs.z, pos.z, 0), 0);
      k4_0 = texelFetch(uKernel, ivec3(kxs.w, pos.z, 0), 0);

      In00_0 = texelFetch(uInput, ivec3(ipos00, z4+2), 0);
      In10_0 = texelFetch(uInput, ivec3(ipos10, z4+2), 0);
      In01_0 = texelFetch(uInput, ivec3(ipos01, z4+2), 0);
      In11_0 = texelFetch(uInput, ivec3(ipos11, z4+2), 0);

      sum00 = fma(In00_1.xxxx, k1_1, sum00);
      sum00 = fma(In00_1.yyyy, k2_1, sum00);
      sum00 = fma(In00_1.zzzz, k3_1, sum00);
      sum00 = fma(In00_1.wwww, k4_1, sum00);

      sum10 = fma(In10_1.xxxx, k1_1, sum10);
      sum10 = fma(In10_1.yyyy, k2_1, sum10);
      sum10 = fma(In10_1.zzzz, k3_1, sum10);
      sum10 = fma(In10_1.wwww, k4_1, sum10);

      sum01 = fma(In01_1.xxxx, k1_1, sum01);
      sum01 = fma(In01_1.yyyy, k2_1, sum01);
      sum01 = fma(In01_1.zzzz, k3_1, sum01);
      sum01 = fma(In01_1.wwww, k4_1, sum01);

      sum11 = fma(In11_1.xxxx, k1_1, sum11);
      sum11 = fma(In11_1.yyyy, k2_1, sum11);
      sum11 = fma(In11_1.zzzz, k3_1, sum11);
      sum11 = fma(In11_1.wwww, k4_1, sum11);
    }

    imageStore(
        uOutput,
        pos00,
        clamp(sum00, uBlock.clamp.x, uBlock.clamp.y));
    if (all(lessThan(pos10, uBlock.size.xyz))) {
      imageStore(
          uOutput,
          pos10,
          clamp(sum10, uBlock.clamp.x, uBlock.clamp.y));
    }
    if (all(lessThan(pos01, uBlock.size.xyz))) {
      imageStore(
          uOutput,
          pos01,
          clamp(sum01, uBlock.clamp.x, uBlock.clamp.y));
    }
    if (all(lessThan(pos11, uBlock.size.xyz))) {
      imageStore(
          uOutput,
          pos11,
          clamp(sum11, uBlock.clamp.x, uBlock.clamp.y));
    }
  }
}
