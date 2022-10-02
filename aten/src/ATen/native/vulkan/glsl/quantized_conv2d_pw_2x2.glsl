#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba8ui) uniform PRECISION restrict writeonly uimage3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION                    isampler3D uInput;
layout(set = 0, binding = 2)          uniform PRECISION                    isampler3D uKernel;
layout(set = 0, binding = 3)          uniform PRECISION                    isampler3D uBias;
layout(set = 0, binding = 4)          uniform PRECISION restrict           Block {
  ivec4 size;
  ivec4 kernel;
  vec2 scale;
  ivec2 zero_point;
  vec2 other_inp_scale;
  ivec2 other_inp_zero_point;
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

    vec4 q_sum00 = texelFetch(uBias, ivec3(pos.z, 0, 0), 0);
    vec4 sum00 = uBlock.other_inp_scale.y * (q_sum00 - uBlock.other_inp_zero_point.y);
    vec4 sum10 = sum00;
    vec4 sum01 = sum00;
    vec4 sum11 = sum00;

    for (int z = 0, z4 = 0; z < uBlock.size.w; z += 4, ++z4) {
      const ivec4 kxs = z + ivec4(0, 1, 2, 3);
      const vec4 q_k1 = texelFetch(uKernel, ivec3(kxs.x, pos.z, 0), 0);
      const vec4 k1 = uBlock.other_inp_scale.x * (q_k1 - uBlock.other_inp_zero_point.x);
      const vec4 q_k2 = texelFetch(uKernel, ivec3(kxs.y, pos.z, 0), 0);
      const vec4 k2 = uBlock.other_inp_scale.x * (q_k2 - uBlock.other_inp_zero_point.x);
      const vec4 q_k3 = texelFetch(uKernel, ivec3(kxs.z, pos.z, 0), 0);
      const vec4 k3 = uBlock.other_inp_scale.x * (q_k3 - uBlock.other_inp_zero_point.x);
      const vec4 q_k4 = texelFetch(uKernel, ivec3(kxs.w, pos.z, 0), 0);
      const vec4 k4 = uBlock.other_inp_scale.x * (q_k4 - uBlock.other_inp_zero_point.x);

      const vec4 In00 = texelFetch(uInput, ivec3(ipos00, z4), 0);
      vec4 deq_In00 = uBlock.scale.y * (In00 - uBlock.zero_point.y);
      const vec4 In10 = texelFetch(uInput, ivec3(ipos10, z4), 0);
      vec4 deq_In10 = uBlock.scale.y * (In10 - uBlock.zero_point.y);
      const vec4 In01 = texelFetch(uInput, ivec3(ipos01, z4), 0);
      vec4 deq_In01 = uBlock.scale.y * (In01 - uBlock.zero_point.y);
      const vec4 In11 = texelFetch(uInput, ivec3(ipos11, z4), 0);
      vec4 deq_In11 = uBlock.scale.y * (In11 - uBlock.zero_point.y);

      if (q_k1 != vec4(0.0)) {
        sum00 = fma(deq_In00.xxxx, k1, sum00);
        sum10 = fma(deq_In10.xxxx, k1, sum10);
        sum01 = fma(deq_In01.xxxx, k1, sum01);
        sum11 = fma(deq_In11.xxxx, k1, sum11);
      }

      if (q_k2 != vec4(0.0)) {
        sum00 = fma(deq_In00.yyyy, k2, sum00);
        sum10 = fma(deq_In10.yyyy, k2, sum10);
        sum01 = fma(deq_In01.yyyy, k2, sum01);
        sum11 = fma(deq_In11.yyyy, k2, sum11);
      }

      if (q_k3 != vec4(0.0)) {
        sum00 = fma(deq_In00.zzzz, k3, sum00);
        sum10 = fma(deq_In10.zzzz, k3, sum10);
        sum01 = fma(deq_In01.zzzz, k3, sum01);
        sum11 = fma(deq_In11.zzzz, k3, sum11);
      }

      if (q_k4 != vec4(0.0)) {
        sum00 = fma(deq_In00.wwww, k4, sum00);
        sum10 = fma(deq_In10.wwww, k4, sum10);
        sum01 = fma(deq_In01.wwww, k4, sum01);
        sum11 = fma(deq_In11.wwww, k4, sum11);
      }
    }
    sum00 = clamp(sum00, uBlock.clamp.x, uBlock.clamp.y);
    vec4 q_ret00 = sum00 / uBlock.scale.x + uBlock.zero_point.x;
    uvec4 res00 = uvec4(int(q_ret00.x), int(q_ret00.y), int(q_ret00.z), int(q_ret00.w));

    sum10 = clamp(sum10, uBlock.clamp.x, uBlock.clamp.y);
    vec4 q_ret10 = sum10 / uBlock.scale.x + uBlock.zero_point.x;
    uvec4 res10 = uvec4(int(q_ret10.x), int(q_ret10.y), int(q_ret10.z), int(q_ret10.w));

    sum01 = clamp(sum01, uBlock.clamp.x, uBlock.clamp.y);
    vec4 q_ret01 = sum01 / uBlock.scale.x + uBlock.zero_point.x;
    uvec4 res01 = uvec4(int(q_ret01.x), int(q_ret01.y), int(q_ret01.z), int(q_ret01.w));

    sum11 = clamp(sum11, uBlock.clamp.x, uBlock.clamp.y);
    vec4 q_ret11 = sum11 / uBlock.scale.x + uBlock.zero_point.x;
    uvec4 res11 = uvec4(int(q_ret11.x), int(q_ret11.y), int(q_ret11.z), int(q_ret11.w));

    imageStore(
        uOutput,
        pos00,
        res00);
    if (all(lessThan(pos10, uBlock.size.xyz))) {
      imageStore(
          uOutput,
          pos10,
          res10);
    }
    if (all(lessThan(pos01, uBlock.size.xyz))) {
      imageStore(
          uOutput,
          pos01,
          res01);
    }
    if (all(lessThan(pos11, uBlock.size.xyz))) {
      imageStore(
          uOutput,
          pos11,
          res11);
    }
  }
}
