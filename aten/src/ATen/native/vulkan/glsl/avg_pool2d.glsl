#version 450 core
#define PRECISION $precision

layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)          uniform           restrict           Block {
  ivec4 inputSize;
  ivec4 outputSize;
  ivec2 kernelSize;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilate;
} uBlock;

#define UP_DIV(x, y) (((x) + (y)-1) / (y))

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.outputSize.xyz))) {
    ivec2 s0 = pos.xy * uBlock.stride - uBlock.padding;
    ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uBlock.dilate)));
    ivec2 efxy = min(uBlock.kernelSize, UP_DIV(uBlock.inputSize.xy - s0, uBlock.dilate));

    vec4 r = vec4(1.0) / float(efxy.x - sfxy.x) / float(efxy.x - sfxy.x);
    vec4 acc = vec4(0);

    for (int kyi = sfxy.y; kyi < efxy.y; ++kyi) {
      for (int kxi = sfxy.x; kxi < efxy.x; ++kxi) {
        ivec2 ixy = s0 + ivec2(kxi, kyi);
        acc += texelFetch(uInput, ivec3(ixy.x, ixy.y, pos.z), 0);
      }
    }

    imageStore(uOutput, pos, r * acc);
  }
}
