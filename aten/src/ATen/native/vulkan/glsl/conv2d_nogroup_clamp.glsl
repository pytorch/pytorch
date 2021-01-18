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
  ivec3 gpos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(gpos, uConstBlock.outputSize.xyz))) {
    ivec3 pos = gpos * ivec3(4, 1, 1);
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
    vec4 color2 = color;
    vec4 color3 = color;
    vec4 color4 = color;
    int kY = pos.z;
    int strideX = uConstBlock.stride.x;
    for (fy = sfxy.y; fy < efxy.y; ++fy) {
      int sy = fy * uConstBlock.dilate.y + s0.y;
      for (fx = 0; fx < kernelX; ++fx) {
        int kZ = fx + fy * kernelX;
        int sx1 = fx * uConstBlock.dilate.x + s0.x;
        int sx2 = sx1 + strideX;
        int sx3 = sx1 + strideX * 2;
        int sx4 = sx1 + strideX * 3;
        float m1 = sx1 >= 0 && sx1 < inputSize.x ? 1.0 : 0.0;
        float m2 = sx2 >= 0 && sx2 < inputSize.x ? 1.0 : 0.0;
        float m3 = sx3 >= 0 && sx3 < inputSize.x ? 1.0 : 0.0;
        float m4 = sx4 >= 0 && sx4 < inputSize.x ? 1.0 : 0.0;
        fz = 0;
        for (; fz < inputSize.z; ++fz) {
          int kX = 4 * fz;
          vec4 k0 = texelFetch(uKernel, ivec3(kX + 0, kY, kZ), 0);
          vec4 k1 = texelFetch(uKernel, ivec3(kX + 1, kY, kZ), 0);
          vec4 k2 = texelFetch(uKernel, ivec3(kX + 2, kY, kZ), 0);
          vec4 k3 = texelFetch(uKernel, ivec3(kX + 3, kY, kZ), 0);

          mat4 k = mat4(k0, k1, k2, k3);

          color += k * texelFetch(uInput, ivec3(sx1, sy, fz), 0) * m1;
          color2 += k * texelFetch(uInput, ivec3(sx2, sy, fz), 0) * m2;
          color3 += k * texelFetch(uInput, ivec3(sx3, sy, fz), 0) * m3;
          color4 += k * texelFetch(uInput, ivec3(sx4, sy, fz), 0) * m4;
        }
      }
    }
    vec4 outputMin = vec4(uConstBlock.outputMin);
    vec4 outputMax = vec4(uConstBlock.outputMax);
    imageStore(uOutput, ivec3(pos.x + 0, pos.y, pos.z), clamp(color, outputMin, outputMax));
    imageStore(uOutput, ivec3(pos.x + 1, pos.y, pos.z), clamp(color2, outputMin, outputMax));
    imageStore(uOutput, ivec3(pos.x + 2, pos.y, pos.z), clamp(color3, outputMin, outputMax));
    imageStore(uOutput, ivec3(pos.x + 3, pos.y, pos.z), clamp(color4, outputMin, outputMax));
  }
}
