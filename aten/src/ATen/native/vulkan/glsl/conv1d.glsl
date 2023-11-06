#version 450 core
#define PRECISION $precision
#define FORMAT $format

layout(std430) buffer;

/*
 * Output Image
 */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;

/*
 * Input Textures
 */
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uKernel;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;

layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  int out_channels;
  int in_lengths;
  int kernel_size;
}
uBlock;

// This implementation optimize for simplicity (and partially performance) for a
// (1, C, L) where C == groups. Hence we only focus on calculating the rolling
// kernel of the L dimension.
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const int out_channels = uBlock.out_channels;
  const int in_lengths = uBlock.in_lengths;
  const int kernel_size = uBlock.kernel_size;

  // The global workgroup should have taken care of it. We only perform one
  // work item for each 1d tensor on lengths
  if (pos.x >= 1) {
    return;
  }

  int c = pos.y;
  if (c >= out_channels) {
    return;
  }

  // Assume n = 1, do not handle n > 1 case for now.
  int n = pos.z;
  if (n >= 1) {
    return;
  }

  vec4 bias = texelFetch(uBias, ivec3(c, 0, 0), 0);

  for (int i = 0; i < in_lengths - kernel_size + 1; i++) {
    vec4 v = vec4(0,0,0,0);
    for (int k = 0; k < kernel_size; k++) {
      const ivec3 in_pos = ivec3(i+k, c, 0);
      const vec4 input_value = texelFetch(uInput, in_pos, 0);

      // Note that we are reading weight in the inner loop, this could be
      // improved by moving it before the outer loop. Since the weight vector is
      // contant for the entire call.

      // weight in input-space: (c, 0, k);
      // notice that c is 4-packed. We need to mod 4 to get the actual weight.
      const ivec3 w_pos = ivec3(k, 0, c / 4);
      const vec4 weight = texelFetch(uKernel, w_pos, 0);

      float w = weight.x;
      if (c % 4 == 1) {
        w = weight.y;
      } else if (c % 4 == 2) {
        w = weight.z;
      } else if (c % 4 == 3) {
        w = weight.w;
      }

      v += w * input_value.x;
    }

    ivec3 out_pos = ivec3(i, c, 0);
    imageStore(uOutput, out_pos, vec4(v.x + bias.x, 0, 0, 0));
  }
}
