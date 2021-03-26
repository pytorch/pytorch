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
  vec2 clamp;
} uBlock;

shared vec4 dg[4][4][4];

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec3 tid = ivec3(gl_LocalInvocationID);
  const ivec3 outpos = ivec3(2*(pos.xy/4) + tid.xy, pos.z);

  dg[tid.z][tid.y][tid.x] += vec4(0);
  for (int z4 = 0; z4 < uBlock.size.w; ++z4) {
    const ivec2 wpos00 = ivec2(16*z4 + 4*tid.x, 4*pos.z + tid.y);
    const vec4 intex = texelFetch(uInput, ivec3(pos.xy, z4), 0);

    dg[tid.z][tid.y][tid.x] += vec4(
      dot(intex, texelFetch(uKernel, ivec3(wpos00.x  , wpos00.y, 0), 0)),
      dot(intex, texelFetch(uKernel, ivec3(wpos00.x+1, wpos00.y, 0), 0)),
      dot(intex, texelFetch(uKernel, ivec3(wpos00.x+2, wpos00.y, 0), 0)),
      dot(intex, texelFetch(uKernel, ivec3(wpos00.x+3, wpos00.y, 0), 0)));
  }

  memoryBarrierShared();
  barrier();

  if (all(lessThan(outpos, uBlock.size.xyz)) && all(lessThan(tid.xy, ivec2(2,2)))) {
    imageStore(uOutput, outpos, dg[tid.z][tid.y][tid.x]);
  }
}
