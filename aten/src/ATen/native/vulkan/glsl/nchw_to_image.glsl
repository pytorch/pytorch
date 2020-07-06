#version 450 core
layout(std430) buffer;
layout(std430) uniform;
layout(set = 0, rgba16f, binding = 0) writeonly highp uniform image3D uImage;
layout(set = 0, binding = 1) readonly buffer destBuffer {
  float data[];
}
uInBuffer;
layout(set = 0, binding = 2) uniform sizeBlock {
  int width;
  int height;
}
uSizeBlock;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  int W = uSizeBlock.width;
  int H = uSizeBlock.height;
  if (pos.x < W && pos.y < H) {
    vec4 color;
    int z = pos.z * 4;
    int WH = W * H;
    color.r = uInBuffer.data[W * pos.y + pos.x + (z + 0) * WH];
    color.g = uInBuffer.data[W * pos.y + pos.x + (z + 1) * WH];
    color.b = uInBuffer.data[W * pos.y + pos.x + (z + 2) * WH];
    color.a = uInBuffer.data[W * pos.y + pos.x + (z + 3) * WH];
    imageStore(uImage, pos, color);
  }
}
