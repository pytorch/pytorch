#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

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
  int in_length;
  int kernel_size;
  int strides;
  int padding;
  int dilation;
  int in_group_size;
  int out_group_size;
  int batch_size;
}
uBlock;

// In our shader's usage, both the numerator and denominator are int, so the
// result of their division is already truncated to int. GLSL's ceil() expects
// one float input, so instead we introduce our own helper.
int ceil(int a, int b) {
  return (a + b - 1) / b;
}

// Let us define
//
// input = (n, in_C, in_L),
// output = (n, out_C, out_L),
// groups = G,
// kernel = K,
//
// which results in shapes
//
// weight = (out_C, in_C / G, K),
// bias = (out_C,).
//
// This implementation performs out_C number of shader invocations, where each
// invocation calculates the rolling kernel of the length dimension, i.e.,
// computes the out_L results.
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const int in_length = uBlock.in_length;
  const int kernel_size = uBlock.kernel_size;
  const int strides = uBlock.strides;
  const int padding = uBlock.padding;
  const int dilation = uBlock.dilation;
  const int in_group_size = uBlock.in_group_size;
  const int out_group_size = uBlock.out_group_size;
  const int batch_size = uBlock.batch_size;

  // The global workgroup should have taken care of it. We perform one shader
  // invocation, per 1D length array of the output tensor.
  if (pos.x >= 1 || pos.z >= 1) {
    return;
  }

  // "out_c" is the output's channel index where we write our result.
  int out_c = pos.y;
  vec4 bias = texelFetch(uBias, ivec3(out_c, 0, 0), 0);

  // "in_c" tracks the input's channel start index.
  // We iterate over the input group that corresponds to the output group.
  int c_start = (out_c / out_group_size) * in_group_size;
  int c_end = c_start + in_group_size;

  // "in_l" tracks the input's length start index for our input-kernel overlay
  // region.
  int l_start = -padding;
  int l_end = in_length + padding - dilation * (kernel_size - 1);

  // "out_l" tracks the output's length index where we write our result.
  int out_l = 0;

  for (int in_l = l_start; in_l < l_end; in_l += strides, ++out_l) {

    // "k" tracks the kernel's index for our input-kernel computation.
    // The kstart/kend borders detect when the corresponding input index is out
    // of bounds.
    int k_start = max(0, ceil(-in_l, dilation));
    int k_end = min(kernel_size, ceil(in_length-in_l, dilation));

    // Since the input/output tensors are channel-packed, which is along the
    // batch dimension, we can batch-read/write four elements at a time.
    for (int n = 0; n < batch_size; n += 4) {
      vec4 v = vec4(0,0,0,0);

      for (int in_c = c_start; in_c < c_end; ++in_c) {
        for (int k = k_start; k < k_end; ++k) {
          int in_pos_x = in_l + k * dilation;
          const ivec3 in_pos = ivec3(in_pos_x, in_c, n / 4);
          const vec4 input_value = texelFetch(uInput, in_pos, 0);

          // Note that we are reading weight in the inner loop, this could be
          // improved by moving it before the outer loop. Since the weight vector is
          // contant for the entire call.

          // weight in input-space: (out_c, in_c % in_group_size, k);
          // notice that c is 4-packed. We need to mod 4 to get the actual weight.
          const ivec3 w_pos = ivec3(k, in_c % in_group_size, out_c / 4);
          const vec4 weight = texelFetch(uKernel, w_pos, 0);

          v += weight[out_c % 4] * input_value;
        }
      }

      ivec3 out_pos = ivec3(out_l, out_c, n / 4);
      imageStore(uOutput, out_pos, v + bias.x);
    }
  }
}
