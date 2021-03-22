#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1) uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION                    sampler3D uKernel;
layout(set = 0, binding = 3) buffer  PRECISION restrict readonly  Bias {
  vec4 data[];
} uBias;
layout(set = 0, binding = 4) uniform PRECISION restrict           Block {
  ivec4 size;
  ivec2 stride;
  ivec2 padding;
  vec2 clamp;
} uBlock;

shared vec4 sh_mem[128];

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec3 tid = ivec3(gl_LocalInvocationID);
  const ivec3 group_size = ivec3(gl_WorkGroupSize);

  const int ic_start = tid.y*group_size.x + tid.x;
  const int ic_inc = group_size.x*group_size.y;
  for (int ic = ic_start; ic < uBlock.size.w; ic += ic_inc) {
    sh_mem[uBlock.size.w*tid.z + ic] = texelFetch(uKernel, ivec3(ic, pos.z, 0), 0);
  }
  memoryBarrierShared();
  barrier();

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec2 ipos = pos.xy * uBlock.stride - uBlock.padding;

    vec4 sum = uBias.data[pos.z];

    for (int z = 0, z4 = 0; z < uBlock.size.w; z += 4, ++z4) {
      const vec4 In = texelFetch(uInput, ivec3(ipos, z4), 0);
      const ivec4 kxs = z + ivec4(0, 1, 2, 3);

      sum = fma(In.xxxx, sh_mem[uBlock.size.w*tid.z + kxs.x], sum);
      sum = fma(In.yyyy, sh_mem[uBlock.size.w*tid.z + kxs.y], sum);
      sum = fma(In.zzzz, sh_mem[uBlock.size.w*tid.z + kxs.z], sum);
      sum = fma(In.wwww, sh_mem[uBlock.size.w*tid.z + kxs.w], sum);
    }

    imageStore(
        uOutput,
        pos,
        clamp(sum, uBlock.clamp.x, uBlock.clamp.y));
  }
}
