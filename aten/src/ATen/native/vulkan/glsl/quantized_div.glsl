#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba8ui) uniform PRECISION restrict writeonly uimage3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION                    isampler3D uInput0; //quantized input
layout(set = 0, binding = 2)          uniform PRECISION                    isampler3D uInput1; //quantized input
layout(set = 0, binding = 3)          uniform PRECISION restrict           Block {
  ivec4 size;
  ivec4 isize0;
  ivec4 isize1;
  vec2 in_scale;
  ivec2 in_zero_point;
  vec2 out_scale;
  ivec2 out_zero_point;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec3 input0_pos = pos % uBlock.isize0.xyz;
    const ivec3 input1_pos = pos % uBlock.isize1.xyz;

    vec4 texel0 = texelFetch(uInput0, input0_pos, 0);
    vec4 texel1 = texelFetch(uInput1, input1_pos, 0);

    vec4 deq_in_0 = uBlock.in_scale.x * (texel0 - uBlock.in_zero_point.x);
    vec4 deq_in_1 = uBlock.in_scale.y * (texel1 - uBlock.in_zero_point.y);

    vec4 res = deq_in_0 / deq_in_1;
    vec4 q_res = res / uBlock.out_scale.x + uBlock.out_zero_point.x;

    uvec4 ret = uvec4(int(q_res.x), int(q_res.y), int(q_res.z), int(q_res.w));

    imageStore(
        uOutput,
        pos,
        ret);
  }
}
