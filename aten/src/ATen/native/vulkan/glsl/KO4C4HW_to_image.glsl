#version 450 core
#define PRECISION $precision
layout(std430) buffer;
layout(set = 0, rgba16f, binding = 0) writeonly PRECISION uniform image3D uOutput;
layout(set = 0, binding = 1) readonly buffer kernel {
  vec4 data[];
}
uKernel;
layout(set = 0, binding = 2) uniform constBlock {
  int KWxKH;
  int C_4;
}
uConstBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID) * ivec3(4, 1, 1);
  int KWxKH = uConstBlock.KWxKH;
  int C_4 = uConstBlock.C_4;
  int bufferIdx = pos.x * KWxKH + 4 * pos.y * C_4 * KWxKH + 4 * pos.z;
  vec4 v0 = uKernel.data[bufferIdx + 0];
  vec4 v1 = uKernel.data[bufferIdx + 1];
  vec4 v2 = uKernel.data[bufferIdx + 2];
  vec4 v3 = uKernel.data[bufferIdx + 3];

  imageStore(uOutput, ivec3(pos.x + 0, pos.y, pos.z), v0);
  imageStore(uOutput, ivec3(pos.x + 1, pos.y, pos.z), v1);
  imageStore(uOutput, ivec3(pos.x + 2, pos.y, pos.z), v2);
  imageStore(uOutput, ivec3(pos.x + 3, pos.y, pos.z), v3);
}
