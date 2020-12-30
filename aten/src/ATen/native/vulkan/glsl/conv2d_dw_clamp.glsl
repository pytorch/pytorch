#version 450 core
#define PRECISION $precision
layout(std430) buffer;
layout(set = 0, rgba16f, binding = 0) writeonly PRECISION uniform image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uKernel;
layout(set = 0, binding = 3) readonly buffer bias {
  vec4 data[];
}
uBias;
layout(set = 0, binding = 4) uniform constBlock {
  ivec2 padding;
  ivec2 kernelSize;
  ivec2 stride;
  ivec2 dilate;
  ivec4 outputSize;
  ivec4 inputSize;
  float outputMin;
  float outputMax;
}
uConstBlock;

#define UP_DIV(x, y) (((x) + (y)-1) / (y))

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  ivec4 outputSize = uConstBlock.outputSize;
  if (all(lessThan(ivec3(gl_GlobalInvocationID), outputSize.xyz))) {
    int KW = uConstBlock.kernelSize.x;
    int KH = uConstBlock.kernelSize.y;
    ivec4 inputSize = uConstBlock.inputSize;
    ivec2 dilate = uConstBlock.dilate;
    ivec2 padding = uConstBlock.padding;
    ivec2 stride = uConstBlock.stride;

    ivec2 s0 = pos.xy * stride - padding;
    ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, dilate)));
    ivec2 efxy = min(uConstBlock.kernelSize, UP_DIV(inputSize.xy - s0, dilate));

    vec4 acc = uBias.data[pos.z];
    int sx, kxi, kyi;
    for (kyi = sfxy.y; kyi < efxy.y; ++kyi) {
      int sy = kyi * dilate.y + s0.y;
      for (kxi = 0; kxi < KW; ++kxi) {
        sx = kxi * dilate.x + s0.x;
        vec4 iv = texelFetch(uInput, ivec3(sx, sy, pos.z), 0);
        vec4 kv = texelFetch(uKernel, ivec3(kxi, kyi, pos.z), 0);
        acc += kv * iv;
      }
    }
    vec4 outputMin = vec4(uConstBlock.outputMin);
    vec4 outputMax = vec4(uConstBlock.outputMax);
    imageStore(uOutput, pos, clamp(acc, outputMin, outputMax));
  }
}
