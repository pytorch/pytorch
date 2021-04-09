#version 450 core
#define PRECISION $precision
layout(std430) buffer;
layout(set = 0, rgba16f, binding = 0) writeonly PRECISION uniform image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform constBlock {
  ivec4 inputSize;
  ivec4 outputSize;
  ivec2 kernelSize;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilate;
}
uConstBlock;

#define UP_DIV(x, y) (((x) + (y)-1) / (y))
#define FLT_MAX 3.402823466e+38

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  ivec3 outputSize = uConstBlock.outputSize.xyz;
  if (all(lessThan(pos, outputSize))) {
    ivec2 s0 = pos.xy * uConstBlock.stride - uConstBlock.padding;
    ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uConstBlock.dilate)));
    ivec2 efxy =
        min(uConstBlock.kernelSize,
            UP_DIV(uConstBlock.inputSize.xy - s0, uConstBlock.dilate));

    vec4 v = vec4(-FLT_MAX);
    for (int kyi = sfxy.y; kyi < efxy.y; ++kyi) {
      for (int kxi = sfxy.x; kxi < efxy.x; ++kxi) {
        ivec2 ixy = s0 + ivec2(kxi, kyi);
        v = max(texelFetch(uInput, ivec3(ixy.x, ixy.y, pos.z), 0), v);
      }
    }

    imageStore(uOutput, pos, v);
  }
}
