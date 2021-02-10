#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1) uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict           Block {
  ivec4 size;
  ivec3 isize;
} uBlock;

shared vec4 sh_mem[64];

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec3 tid = ivec3(gl_LocalInvocationID);
  const ivec3 group_size = ivec3(gl_WorkGroupSize);

  if (pos.z < uBlock.isize.z) {
    vec4 sum = vec4(0);

    for (int y = tid.y; y < uBlock.isize.y; y+=group_size.y) {
      for (int x = tid.x; x < uBlock.isize.x; x+=group_size.x) {
        sum += texelFetch(uInput, ivec3(x, y, pos.z), 0);
      }
    }

    sh_mem[tid.z * group_size.y * group_size.x + tid.y * group_size.x + tid.x] = sum;
  }
  memoryBarrierShared();
  barrier();

  if (tid.y > 0 || tid.x > 0 || pos.z >= uBlock.isize.z) {
    return;
  }

  vec4 total = vec4(0);
  for (int y = 0; y < group_size.y; ++y) {
    for (int x = 0; x < group_size.x; ++x) {
      total += sh_mem[tid.z * group_size.y * group_size.x + y * group_size.x + x];
    }
  }

  vec4 outtex = total / uBlock.size.w;
  int zout = 4*pos.z;
  int width = uBlock.size.x;

  ivec3 posx = ivec3(zout%width, zout/width, 0);
  vec4 outx = vec4(outtex.x, 0, 0, 0);

  ivec3 posy = ivec3((zout+1)%width, (zout+1)/width, 0);
  vec4 outy = vec4(outtex.y, 0, 0, 0);

  ivec3 posz = ivec3((zout+2)%width, (zout+2)/width, 0);
  vec4 outz = vec4(outtex.z, 0, 0, 0);

  ivec3 posw = ivec3((zout+3)%width, (zout+3)/width, 0);
  vec4 outw = vec4(outtex.w, 0, 0, 0);

  if (all(lessThan(posx, uBlock.size.xyz))) {
    imageStore(uOutput, posx, outx);
  }
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
