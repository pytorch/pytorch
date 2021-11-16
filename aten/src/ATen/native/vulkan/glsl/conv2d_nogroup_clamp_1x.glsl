#version 450 core
#define PRECISION $precision
layout(std430) buffer;
layout(set = 0, rgba32f, binding = 0) writeonly PRECISION uniform image3D uOutput;
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

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uConstBlock.outputSize.xyz))) {
    int kernelX = uConstBlock.kernelSize.x;
    int kernelY = uConstBlock.kernelSize.y;
    ivec3 inputSize = uConstBlock.inputSize.xyz;
    ivec2 s0 = pos.xy * uConstBlock.stride - uConstBlock.padding;
    int fx, fy, fz;
    ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uConstBlock.dilate)));
    ivec2 efxy =
        min(uConstBlock.kernelSize,
            UP_DIV(uConstBlock.inputSize.xy - s0, uConstBlock.dilate));
    vec4 color = uBias.data[pos.z];
    int kY = pos.z;
    int strideX = uConstBlock.stride.x;
    for (fy = sfxy.y; fy < efxy.y; ++fy) {
      int sy = fy * uConstBlock.dilate.y + s0.y;
      for (fx = 0; fx < kernelX; ++fx) {
        int kZ = fx + fy * kernelX;
        int sx = fx * uConstBlock.dilate.x + s0.x;
        fz = 0;
        for (; fz < inputSize.z; ++fz) {
          int kX = 4 * fz;
          vec4 k0 = texelFetch(uKernel, ivec3(kX + 0, kY, kZ), 0);
          vec4 k1 = texelFetch(uKernel, ivec3(kX + 1, kY, kZ), 0);
          vec4 k2 = texelFetch(uKernel, ivec3(kX + 2, kY, kZ), 0);
          vec4 k3 = texelFetch(uKernel, ivec3(kX + 3, kY, kZ), 0);

          mat4 k = mat4(k0, k1, k2, k3);

          color += k * texelFetch(uInput, ivec3(sx, sy, fz), 0);
        }
      }
    }
    vec4 outputMin = vec4(uConstBlock.outputMin);
    vec4 outputMax = vec4(uConstBlock.outputMax);
    imageStore(uOutput, ivec3(pos.x, pos.y, pos.z), clamp(color, outputMin, outputMax));
  }
}
