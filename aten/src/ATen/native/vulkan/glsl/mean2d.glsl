#version 450 core
#define PRECISION $precision
layout(std430) buffer;
layout(std430) uniform;
layout(set = 0, binding = 0, rgba32f) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)          uniform PRECISION restrict                     Block {
  int W;
  int H;
} uBlock;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  vec4 r = vec4(1.0) / (float(uBlock.W) * float(uBlock.H));
  vec4 acc = vec4(0);
  int xi, yi;
  int zi = (imageSize(uOutput).x*pos.y + pos.x)/4;
  int zo = (imageSize(uOutput).x*pos.y + pos.x)%4;
  for (yi = 0; yi < uBlock.H; ++yi) {
    for (xi = 0; xi < uBlock.W; ++xi) {
      acc += texelFetch(uInput, ivec3(xi, yi, zi), 0);
    }
  }
  vec4 outValue = r * acc;

  imageStore(uOutput, pos, vec4(outValue[zo], 0,0,0));
}
