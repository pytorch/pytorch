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

  const vec4 outtex = total / uBlock.size.w;
  const int zoutx = 4*pos.z;
  const int width = uBlock.size.x;
  const int maxlen = uBlock.size.x * uBlock.size.y;

  const int zouty = min(zoutx + 1, maxlen);
  ivec3 posy = ivec3((zouty)%width, (zouty)/width, 0);
  vec4 outy = vec4(outtex.y, 0, 0, 0);
  imageStore(uOutput, posy, outy);

  const int zoutz = min(zoutx + 2, maxlen);
  ivec3 posz = ivec3((zoutz)%width, (zoutz)/width, 0);
  vec4 outz = vec4(outtex.z, 0, 0, 0);
  imageStore(uOutput, posz, outz);

  const int zoutw = min(zoutx + 3, maxlen);
  ivec3 posw = ivec3((zoutw)%width, (zoutw)/width, 0);
  vec4 outw = vec4(outtex.w, 0, 0, 0);
  imageStore(uOutput, posw, outw);

  ivec3 posx = ivec3(zoutx%width, zoutx/width, 0);
  vec4 outx = vec4(outtex.x, 0, 0, 0);
  imageStore(uOutput, posx, outx);
}
