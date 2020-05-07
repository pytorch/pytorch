#version 450 core
layout(std430) buffer;
layout(std430) uniform;

layout(set = 0, rgba16f, binding = 0) writeonly mediump uniform image3D uOutput;
layout(set = 0, binding = 1) uniform mediump sampler3D uInput0;
layout(set = 0, binding = 2) uniform mediump sampler3D uInput1;
layout(set = 0, binding = 3) uniform constBlock {
  int W;
  int H;
  int C;
  float alpha;
}
uConstBlock;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  ivec3 WHC = ivec3(uConstBlock.W, uConstBlock.H, uConstBlock.C);
  if (all(lessThan(pos, WHC))) {
    vec4 v = texelFetch(uInput0, pos, 0) +
        uConstBlock.alpha * texelFetch(uInput1, pos, 0);
    imageStore(uOutput, pos, v);
  }
}
