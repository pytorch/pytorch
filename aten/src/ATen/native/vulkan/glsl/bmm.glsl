#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uM1;
layout(set = 0, binding = 2)         uniform PRECISION                    sampler3D uM2;
layout(set = 0, binding = 3)         uniform PRECISION restrict           Block {
  ivec4 size;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  ivec3 posx = ivec3(pos.x * 2, pos.y * 2, pos.z);

  if (all(lessThan(posx, uBlock.size.xyz))) {
    vec4 sum_b1 = vec4(0);
    vec4 sum_b2 = vec4(0);
    vec4 sum_b3 = vec4(0);
    vec4 sum_b4 = vec4(0);

    for (int k = 0; k < uBlock.size.w; ++k) {
      ivec3 inposx = ivec3(2 * k, 2 * pos.y, pos.z);
      vec4 intexx = texelFetch(uM1, inposx, 0);
      ivec3 inposy = ivec3(inposx.x + 1, inposx.y, pos.z);
      vec4 intexy = texelFetch(uM1, inposy, 0);
      ivec3 inposz = ivec3(inposx.x, inposx.y + 1, pos.z);
      vec4 intexz = texelFetch(uM1, inposz, 0);
      ivec3 inposw = ivec3(inposx.x + 1, inposx.y + 1, pos.z);
      vec4 intexw = texelFetch(uM1, inposw, 0);

      vec4 texel1_b1 = vec4(intexx.x, intexy.x, intexz.x, intexw.x);
      vec4 texel1_b2 = vec4(intexx.y, intexy.y, intexz.y, intexw.y);
      vec4 texel1_b3 = vec4(intexx.z, intexy.z, intexz.z, intexw.z);
      vec4 texel1_b4 = vec4(intexx.w, intexy.w, intexz.w, intexw.w);

      vec4 texel2_b1 = texelFetch(uM2, ivec3(pos.x, k, 4 * pos.z), 0);
      vec4 texel2_b2 = texelFetch(uM2, ivec3(pos.x, k, 4 * pos.z + 1), 0);
      vec4 texel2_b3 = texelFetch(uM2, ivec3(pos.x, k, 4 * pos.z + 2), 0);
      vec4 texel2_b4 = texelFetch(uM2, ivec3(pos.x, k, 4 * pos.z + 3), 0);

      sum_b1 = fma(texel1_b1.xxzz, texel2_b1.xyxy, sum_b1);
      sum_b1 = fma(texel1_b1.yyww, texel2_b1.zwzw, sum_b1);
      sum_b2 = fma(texel1_b2.xxzz, texel2_b2.xyxy, sum_b2);
      sum_b2 = fma(texel1_b2.yyww, texel2_b2.zwzw, sum_b2);
      sum_b3 = fma(texel1_b3.xxzz, texel2_b3.xyxy, sum_b3);
      sum_b3 = fma(texel1_b3.yyww, texel2_b3.zwzw, sum_b3);
      sum_b4 = fma(texel1_b4.xxzz, texel2_b4.xyxy, sum_b4);
      sum_b4 = fma(texel1_b4.yyww, texel2_b4.zwzw, sum_b4);
    }

    const vec4 outx = vec4(sum_b1.x, sum_b2.x, sum_b3.x, sum_b4.x);

    const ivec3 posy = ivec3(posx.x + 1, posx.y, pos.z);
    const vec4 outy = vec4(sum_b1.y, sum_b2.y, sum_b3.y, sum_b4.y);

    const ivec3 posz = ivec3(posx.x, posx.y + 1, pos.z);
    const vec4 outz = vec4(sum_b1.z, sum_b2.z, sum_b3.z, sum_b4.z);

    const ivec3 posw = ivec3(posx.x + 1, posx.y + 1, pos.z);
    const vec4 outw = vec4(sum_b1.w, sum_b2.w, sum_b3.w, sum_b4.w);

    imageStore(uOutput, posx, outx);
    if (all(lessThan(posy, uBlock.size.xyz))) {
      imageStore(uOutput, posy, outy);
    }
    if (all(lessThan(posz, uBlock.size.xyz))) {
      imageStore(uOutput, posz, outz);
    }
    if (all(lessThan(posw, uBlock.size.xyz))) {
      imageStore(uOutput, posw, outw);
    }
  }
}
