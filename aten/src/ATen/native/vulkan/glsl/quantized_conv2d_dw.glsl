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
  ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec2 ipos = pos.xy * uBlock.stride - uBlock.padding;

    const ivec2 start = max(ivec2(0), ipos);
    const ivec2 end = min(ipos + uBlock.kernel.xy, uBlock.kernel.zw);
    const ivec2 kstart = (start - ipos) / uBlock.dilate;

    vec4 q_sum = texelFetch(uBias, ivec3(pos.z, 0, 0), 0);
    vec4 sum = uBlock.other_inp_scale.y * (q_sum - uBlock.other_inp_zero_point.y);

    for (int y = start.y, ky = kstart.y; y < end.y; y += uBlock.dilate.y, ++ky) {
      for (int x = start.x, kx = kstart.x + ky * uBlock.ikernel.x; x < end.x; x += uBlock.dilate.x, ++kx) {
        const vec4 In = texelFetch(uInput, ivec3(x, y, pos.z), 0);
        vec4 deq_In = uBlock.scale.y * (In - uBlock.zero_point.y);

        const vec4 weight = texelFetch(uKernel, ivec3(kx, pos.z, 0), 0);
        if (weight != vec4(0.0)) {
          vec4 deq_weight = uBlock.other_inp_scale.x * (weight - uBlock.other_inp_zero_point.x);
          sum = fma(
            deq_In,
            deq_weight,
            sum);
        }
      }
    }

    sum = clamp(sum, uBlock.clamp.x, uBlock.clamp.y);
    vec4 q_ret = sum / uBlock.scale.x + uBlock.zero_point.x;
    uvec4 res = uvec4(int(q_ret.x), int(q_ret.y), int(q_ret.z), int(q_ret.w));

    imageStore(
        uOutput,
        pos,
        res);
  }
}
