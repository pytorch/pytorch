#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1) uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict           Block {
  ivec4 size;
  ivec2 limits;
  ivec2 padding;
} uBlock;

shared vec4 i[4][4][4];

ivec2 off[4] = {
  ivec2(0, 2),
  ivec2(0, 1),
  ivec2(0, -1),
  ivec2(-2, 0)
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec3 tid = ivec3(gl_LocalInvocationID);

  const ivec2 ipos = (pos.xy/4) * 2 - uBlock.padding + tid.xy;

  const int shz = tid.z*16;
  const int shy = tid.y*4;
  const int shzy = shz + shy;

  i[tid.z][tid.y][tid.x] = texelFetch(uInput, ivec3(ipos.x, ipos.y, pos.z), 0) *
                           int(all(greaterThanEqual(ipos, ivec2(0,0)))) *
                           int(all(lessThan(ipos, uBlock.limits)));

  memoryBarrierShared();
  barrier();

  const ivec2 ys = off[tid.y] + tid.y;
  const ivec2 xs = off[tid.x] + tid.x;

  const vec4 c0 = tid.y != 1 ? i[tid.z][ys.x][xs.x] - i[tid.z][ys.y][xs.x] :
                               i[tid.z][ys.x][xs.x] + i[tid.z][ys.y][xs.x];
  const vec4 c1 = tid.y != 1 ? i[tid.z][ys.x][xs.y] - i[tid.z][ys.y][xs.y] :
                               i[tid.z][ys.x][xs.y] + i[tid.z][ys.y][xs.y];

  vec4 outvec;
  if (tid.x == 1)
    outvec = c0 + c1;
  else
    outvec = c0 - c1;

  if (all(lessThan(pos, uBlock.size.xyz))) {
    imageStore(uOutput, pos, outvec);
  }
}
