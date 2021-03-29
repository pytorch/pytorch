#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1) uniform PRECISION                    sampler3D uM1;
layout(set = 0, binding = 2) uniform PRECISION                    sampler3D uM2;
layout(set = 0, binding = 3) uniform PRECISION                    sampler3D uT;
layout(set = 0, binding = 4) uniform PRECISION restrict           Block {
  ivec4 size;
  vec2 multiplier;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec3 posx = ivec3(pos.x*2, pos.y*2, pos.z);

  if (all(lessThan(posx, uBlock.size.xyz))) {
    vec4 sum = vec4(0);

    for (int k = 0; k < uBlock.size.w; ++k) {
      const ivec3 inposx = ivec3(2*k, 2*pos.y, pos.z);
      const vec4 intexx = texelFetch(uM1, inposx, 0);
      const ivec3 inposy = ivec3(inposx.x + 1, inposx.y, pos.z);
      const vec4 intexy = texelFetch(uM1, inposy, 0);
      const ivec3 inposz = ivec3(inposx.x, inposx.y + 1, pos.z);
      const vec4 intexz = texelFetch(uM1, inposz, 0);
      const ivec3 inposw = ivec3(inposx.x + 1, inposx.y + 1, pos.z);
      const vec4 intexw = texelFetch(uM1, inposw, 0);

      vec4 texel1 = vec4(intexx.x, intexy.x, intexz.x, intexw.x);
      vec4 texel2 = texelFetch(uM2, ivec3(pos.x, k, pos.z), 0);
      sum = fma(texel1.xxzz, texel2.xyxy, sum);
      sum = fma(texel1.yyww, texel2.zwzw, sum);
    }

    const vec4 outtex = uBlock.multiplier.x * sum + uBlock.multiplier.y * texelFetch(uT, pos, 0);

    const ivec3 posy = posx + ivec3(int((posx.x + 1) < uBlock.size.x), 0, 0);
    const vec4 outy = vec4(outtex.y, 0, 0, 0);
    imageStore(uOutput, posy, outy);

    const ivec3 posz = posx + ivec3(0, int((posx.y + 1) < uBlock.size.y), 0);
    const vec4 outz = vec4(outtex.z, 0, 0, 0);
    imageStore(uOutput, posz, outz);

    const int valid = int((posx.x + 1) < uBlock.size.x && (posx.y + 1) < uBlock.size.y);
    const ivec3 posw = posx + ivec3(valid, valid, 0);
    const vec4 outw = vec4(outtex.w, 0, 0, 0);
    imageStore(uOutput, posw, outw);

    const vec4 outx = vec4(outtex.x, 0, 0, 0);
    imageStore(uOutput, posx, outx);
  }
}
