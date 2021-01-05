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

shared vec4 sh_mem_1[64];
shared vec4 sh_mem_2[64];

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
  const ivec2 tid = ivec2(gl_LocalInvocationID.xy);
  const ivec2 group_size = ivec2(gl_WorkGroupSize.xy);

  const int num_tiles = (uBlock.size.w + group_size.x - 1) / group_size.x;

  vec4 sum = vec4(0);

  for (int tile = 0; tile < num_tiles; ++tile) {
    sh_mem_1[tid.y*group_size.x + tid.x] = texelFetch(uM1, ivec2(tile*group_size.x + tid.x, pos.y), 0);
    sh_mem_2[tid.y*group_size.x + tid.x] = texelFetch(uM2, ivec2(pos.x, tile*group_size.y + tid.y), 0);
    memoryBarrierShared();
    barrier();
    const int limit = min(uBlock.size.w - tile*group_size.x, group_size.x);
    for (int k = 0; k < limit; ++k) {
      vec4 texel1 = sh_mem_1[tid.y*group_size.x + k];
      vec4 texel2 = sh_mem_2[k*group_size.x + tid.x];
      sum = fma(texel1.xxzz, texel2.xyxy, sum);
      sum = fma(texel1.yyww, texel2.zwzw, sum);
    }
  }


  if (all(lessThan(pos, uBlock.size.xy))) {
    imageStore(
        uOutput,
        pos,
        uBlock.multiplier.x * sum + uBlock.multiplier.y * texelFetch(uT, pos, 0));
  }
}
