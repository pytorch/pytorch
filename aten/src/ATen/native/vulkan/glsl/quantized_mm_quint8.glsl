#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba8ui) uniform PRECISION restrict writeonly uimage3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    isampler3D uM1; //quantized input
layout(set = 0, binding = 2)         uniform PRECISION                    isampler3D uM2; //quantized input
layout(set = 0, binding = 3)         uniform PRECISION restrict           Block {
  ivec4 size;
  ivec4 um1_size;
  ivec4 um2_size;
  vec2 scales;
  vec2 out_scale;
  ivec2 zero_points;
  ivec2 out_zero_point;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  ivec3 posx = ivec3(pos.x*2, pos.y*2, pos.z);

  if (all(lessThan(posx, uBlock.size.xyz))) {
    vec4 sum = vec4(0);

    for (int k = 0; k < uBlock.size.w; ++k) {
      ivec3 inposx = ivec3(2*k, 2*pos.y, pos.z);
      vec4 intexx = vec4(0.0);
      if (all(lessThan(inposx, uBlock.um1_size.xyz))) {
        const vec4 intexx_quant = texelFetch(uM1, inposx, 0);
        intexx = uBlock.scales.x * (intexx_quant - uBlock.zero_points.x);
      }
      ivec3 inposy = ivec3(inposx.x + 1, inposx.y, pos.z);
      vec4 intexy = vec4(0.0);
      if (all(lessThan(inposy, uBlock.um1_size.xyz))) {
        const vec4 intexy_quant = texelFetch(uM1, inposy, 0);
        intexy = uBlock.scales.x * (intexy_quant - uBlock.zero_points.x);
      }
      ivec3 inposz = ivec3(inposx.x, inposx.y + 1, pos.z);
      vec4 intexz = vec4(0.0);
      if (all(lessThan(inposz, uBlock.um1_size.xyz))) {
        const vec4 intexz_quant = texelFetch(uM1, inposz, 0);
        intexz = uBlock.scales.x * (intexz_quant - uBlock.zero_points.x);
      }
      ivec3 inposw = ivec3(inposx.x + 1, inposx.y + 1, pos.z);
      vec4 intexw = vec4(0.0);
      if (all(lessThan(inposw, uBlock.um1_size.xyz))) {
        const vec4 intexw_quant = texelFetch(uM1, inposw, 0);
        intexw = uBlock.scales.x * (intexw_quant - uBlock.zero_points.x);
      }

      vec4 texel1 = vec4(intexx.x, intexy.x, intexz.x, intexw.x);
      vec4 texel2 = vec4(0.0);
      ivec3 texel2_loc = ivec3(pos.x, k, pos.z);
      if (all(lessThan(texel2_loc, uBlock.um2_size.xyz))) {
        const vec4 texel2_quant = texelFetch(uM2, texel2_loc, 0);
        texel2 = uBlock.scales.y * (texel2_quant - uBlock.zero_points.y);
      }
      sum = fma(texel1.xxzz, texel2.xyxy, sum);
      sum = fma(texel1.yyww, texel2.zwzw, sum);
    }

    vec4 outx = vec4(sum.x, 0, 0, 0);

    ivec3 posy = ivec3(posx.x+1, posx.y, pos.z);
    vec4 outy = vec4(sum.y, 0, 0, 0);

    ivec3 posz = ivec3(posx.x, posx.y+1, pos.z);
    vec4 outz = vec4(sum.z, 0, 0, 0);

    ivec3 posw = ivec3(posx.x+1, posx.y+1, pos.z);
    vec4 outw = vec4(sum.w, 0, 0, 0);

    outx = roundEven(outx / uBlock.out_scale.x) + uBlock.out_zero_point.x;
    uvec4 storex = uvec4(outx);
    imageStore(uOutput, posx, storex);
    if (all(lessThan(posy, uBlock.size.xyz))) {
      outy = roundEven(outy / uBlock.out_scale.x) + uBlock.out_zero_point.x;
      uvec4 storey = uvec4(outy);
      imageStore(uOutput, posy, storey);
    }
    if (all(lessThan(posz, uBlock.size.xyz))) {
      outz = roundEven(outz / uBlock.out_scale.x) + uBlock.out_zero_point.x;
      uvec4 storez = uvec4(outz);
      imageStore(uOutput, posz, storez);
    }
    if (all(lessThan(posw, uBlock.size.xyz))) {
      outw = roundEven(outw / uBlock.out_scale.x) + uBlock.out_zero_point.x;
      uvec4 storew = uvec4(outw);
      imageStore(uOutput, posw, storew);
    }
  }
}
