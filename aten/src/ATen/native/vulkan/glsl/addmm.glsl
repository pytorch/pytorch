#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict writeonly image2D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION                    sampler2D uM1;
layout(set = 0, binding = 2)          uniform PRECISION                    sampler2D uM2;
layout(set = 0, binding = 3)          uniform PRECISION                    sampler2D uT;
layout(set = 0, binding = 4)          uniform PRECISION restrict           Block {
  ivec4 size;
  vec2 multiplier;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

  if (all(lessThan(pos, uBlock.size.xy))) {
    vec4 sum = vec4(0);

    for (int k = 0; k < uBlock.size.w; ++k) {
      vec4 texel1 = texelFetch(uM1, ivec2(k, pos.y), 0);
      vec4 texel2 = texelFetch(uM2, ivec2(pos.x, k), 0);
      sum = fma(texel1.xxzz, texel2.xyxy, sum);
      sum = fma(texel1.yyww, texel2.zwzw, sum);
    }

    imageStore(
        uOutput,
        pos,
        uBlock.multiplier.x * sum + uBlock.multiplier.y * texelFetch(uT, pos, 0));
  }
}
