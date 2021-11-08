#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1) uniform PRECISION                    sampler3D uM1;
layout(set = 0, binding = 2) uniform PRECISION                    sampler3D uM2;
layout(set = 0, binding = 3) uniform PRECISION restrict           Block {
  ivec4 size;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  ivec3 posx = ivec3(pos.x*2, pos.y*2, pos.z);

  if (all(lessThan(posx, uBlock.size.xyz))) {
    vec4 sum = vec4(0);

    for (int k = 0; k < uBlock.size.w; ++k) {
      ivec3 inposx = ivec3(2*k, 2*pos.y, pos.z);
      vec4 intexx = texelFetch(uM1, inposx, 0);
      ivec3 inposy = ivec3(inposx.x + 1, inposx.y, pos.z);
      vec4 intexy = texelFetch(uM1, inposy, 0);
      ivec3 inposz = ivec3(inposx.x, inposx.y + 1, pos.z);
      vec4 intexz = texelFetch(uM1, inposz, 0);
      ivec3 inposw = ivec3(inposx.x + 1, inposx.y + 1, pos.z);
      vec4 intexw = texelFetch(uM1, inposw, 0);

      vec4 texel1 = vec4(intexx.x, intexy.x, intexz.x, intexw.x);
      vec4 texel2 = texelFetch(uM2, ivec3(pos.x, k, pos.z), 0);
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
