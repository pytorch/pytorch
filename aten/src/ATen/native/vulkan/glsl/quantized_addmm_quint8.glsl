#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba8ui) uniform PRECISION restrict writeonly uimage3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    isampler3D uM1; //quantized input
layout(set = 0, binding = 2)         uniform PRECISION                    isampler3D uM2; //quantized input
layout(set = 0, binding = 3)         uniform PRECISION                    sampler3D uT;
layout(set = 0, binding = 4)         uniform PRECISION restrict           Block {
  ivec4 size;
  ivec4 um1_size;
  ivec4 um2_size;
  ivec4 ut_size;
  vec2 multiplier;
  vec2 scales;
  vec2 out_scale;
  ivec2 zero_points;
  ivec2 out_zero_point;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec3 posx = ivec3(pos.x*2, pos.y*2, pos.z);

  if (all(lessThan(posx, uBlock.size.xyz))) {
    vec4 sum = vec4(0);
    for (int k = 0; k < uBlock.size.w; ++k) {
      const ivec3 inposx = ivec3(2*k, 2*pos.y, pos.z);
      vec4 intexx = vec4(0.0);
      if (all(lessThan(inposx, uBlock.um1_size.xyz))) {
        const vec4 intexx_quant = texelFetch(uM1, inposx, 0);
        intexx = uBlock.scales.x * (intexx_quant - uBlock.zero_points.x);
      }
      const ivec3 inposy = ivec3(inposx.x + 1, inposx.y, pos.z);
      vec4 intexy = vec4(0.0);
      if (all(lessThan(inposy, uBlock.um1_size.xyz))) {
        const vec4 intexy_quant = texelFetch(uM1, inposy, 0);
        intexy = uBlock.scales.x * (intexy_quant - uBlock.zero_points.x);
      }
      const ivec3 inposz = ivec3(inposx.x, inposx.y + 1, pos.z);
      vec4 intexz = vec4(0.0);
      if (all(lessThan(inposz, uBlock.um1_size.xyz))) {
        const vec4 intexz_quant = texelFetch(uM1, inposz, 0);
        intexz = uBlock.scales.x * (intexz_quant - uBlock.zero_points.x);
      }
      const ivec3 inposw = ivec3(inposx.x + 1, inposx.y + 1, pos.z);
      vec4 intexw = vec4(0.0);
      if (all(lessThan(inposw, uBlock.um1_size.xyz))) {
        const vec4 intexw_quant = texelFetch(uM1, inposw, 0);
        intexw = uBlock.scales.x * (intexw_quant - uBlock.zero_points.x);
      }

      vec4 texel1 = vec4(intexx.x, intexy.x, intexz.x, intexw.x);
      vec4 texel2 = vec4(0.0);
      ivec3 um2_pos = ivec3(pos.x, k, pos.z);
      if (all(lessThan(um2_pos, uBlock.um2_size.xyz))) {
        vec4 texel2_quant = texelFetch(uM2, um2_pos, 0);
        texel2 = uBlock.scales.y * (texel2_quant - uBlock.zero_points.y);
      }
      sum = fma(texel1.xxzz, texel2.xyxy, sum);
      sum = fma(texel1.yyww, texel2.zwzw, sum);
    }

    vec4 outtex;
    const ivec3 bias_pos = pos % uBlock.ut_size.xyz;
    if (all(lessThan(bias_pos, uBlock.ut_size.xyz))) {
      outtex = uBlock.multiplier.x * sum + uBlock.multiplier.y * texelFetch(uT, bias_pos, 0);
    } else {
      outtex = uBlock.multiplier.x * sum;
    }

    const ivec3 posy = posx + ivec3(int((posx.x + 1) < uBlock.size.x), 0, 0);
    vec4 outy = vec4(outtex.y, 0, 0, 0);
    outy = roundEven(outy / uBlock.out_scale.x) + uBlock.out_zero_point.x;
    uvec4 storey = uvec4(outy);
    imageStore(uOutput, posy, storey);

    const ivec3 posz = posx + ivec3(0, int((posx.y + 1) < uBlock.size.y), 0);
    vec4 outz = vec4(outtex.z, 0, 0, 0);
    outz = roundEven(outz / uBlock.out_scale.x) + uBlock.out_zero_point.x;
    uvec4 storez = uvec4(outz);
    imageStore(uOutput, posz, storez);

    const int valid = int((posx.x + 1) < uBlock.size.x && (posx.y + 1) < uBlock.size.y);
    const ivec3 posw = posx + ivec3(valid, valid, 0);
    vec4 outw = vec4(outtex.w, 0, 0, 0);
    outw = roundEven(outw / uBlock.out_scale.x) + uBlock.out_zero_point.x;
    uvec4 storew = uvec4(outw);
    imageStore(uOutput, posw, storew);

    vec4 outx = vec4(outtex.x, 0, 0, 0);
    outx = roundEven(outx / uBlock.out_scale.x) + uBlock.out_zero_point.x;
    uvec4 storex = uvec4(outx);
    imageStore(uOutput, posx, storex);
  }
}
