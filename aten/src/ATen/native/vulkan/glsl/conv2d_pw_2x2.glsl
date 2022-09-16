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

    for (int z = 0, z4 = 0; z < uBlock.size.w; z += 4, ++z4) {
      const ivec4 kxs = z + ivec4(0, 1, 2, 3);
      const vec4 k1 = texelFetch(uKernel, ivec3(kxs.x, pos.z, 0), 0);
      const vec4 k2 = texelFetch(uKernel, ivec3(kxs.y, pos.z, 0), 0);
      const vec4 k3 = texelFetch(uKernel, ivec3(kxs.z, pos.z, 0), 0);
      const vec4 k4 = texelFetch(uKernel, ivec3(kxs.w, pos.z, 0), 0);

      const vec4 In00 = texelFetch(uInput, ivec3(ipos00, z4), 0);
      const vec4 In10 = texelFetch(uInput, ivec3(ipos10, z4), 0);
      const vec4 In01 = texelFetch(uInput, ivec3(ipos01, z4), 0);
      const vec4 In11 = texelFetch(uInput, ivec3(ipos11, z4), 0);

      sum00 = fma(In00.xxxx, k1, sum00);
      sum00 = fma(In00.yyyy, k2, sum00);
      sum00 = fma(In00.zzzz, k3, sum00);
      sum00 = fma(In00.wwww, k4, sum00);

      sum10 = fma(In10.xxxx, k1, sum10);
      sum10 = fma(In10.yyyy, k2, sum10);
      sum10 = fma(In10.zzzz, k3, sum10);
      sum10 = fma(In10.wwww, k4, sum10);

      sum01 = fma(In01.xxxx, k1, sum01);
      sum01 = fma(In01.yyyy, k2, sum01);
      sum01 = fma(In01.zzzz, k3, sum01);
      sum01 = fma(In01.wwww, k4, sum01);

      sum11 = fma(In11.xxxx, k1, sum11);
      sum11 = fma(In11.yyyy, k2, sum11);
      sum11 = fma(In11.zzzz, k3, sum11);
      sum11 = fma(In11.wwww, k4, sum11);
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
