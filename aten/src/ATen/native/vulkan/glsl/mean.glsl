#version 450 core
layout(std430) buffer;
layout(std430) uniform;
layout(set = 0, rgba16f, binding = 0) writeonly mediump uniform image3D uOutput;
layout(set = 0, binding = 1) uniform mediump sampler3D uInput;
layout(set = 0, binding = 2) uniform constBlock {
  int W;
  int H;
  int OW;
  int OH;
}
uConstBlock;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  int W = uConstBlock.W;
  int H = uConstBlock.H;
  int OW = uConstBlock.OW;
  int OH = uConstBlock.OH;
  vec4 r = vec4(1.0) / float(W) / float(H);
  vec4 acc = vec4(0);
  int xi, yi;
  for (xi = 0; xi < W; ++xi) {
    for (yi = 0; yi < H; ++yi) {
      acc += texelFetch(uInput, ivec3(xi, yi, pos.z), 0);
    }
  }
  vec4 outValue = r * acc;
  for (int vi = 0; vi < 4; ++vi) {
    int oy = (4 * pos.z + vi) / OW;
    int ox = (4 * pos.z + vi) % OW;
    imageStore(uOutput, ivec3(ox, oy, 0), vec4(outValue[vi], 0, 0, 0));
  }
}
