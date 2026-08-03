#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

/*
 * Output Image
 */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;

/*
 * Input Buffer
 */
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;

/*
 * Params Buffer
 * input_shader_extents is the dimensions of the Vulkan 3D texture XYZ
 * with a zero pad at W.
 * input_tensor_dims is the dimensions of the NCHW PyTorch Tensor.
 * input_dim_stride is the stride to include elements along the softmax
 * dimension calculation. early_exit is the global workgroup position-based
 * condition for unnecessary invocations to exit.
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 input_shader_extents;
  ivec4 input_tensor_dims;
  ivec4 input_dim_stride;
  ivec4 early_exit;
}
uBlock;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  // how "wide" a batch is in terms of z. Only have one invocation per batch,
  // as one batch width has elements from every channel in-memory.
  if (!all(lessThan(pos, uBlock.early_exit.xyz))) {
    return;
  }
  const int b_stride = int(ceil(uBlock.input_tensor_dims.y / 4.0));
  const ivec3 src_pos = ivec3(pos.x, pos.y, pos.z * b_stride);
  // tail case, padded zeros in memory if tensor's channel dim % 4 != 0
  uint tail_case_size = uBlock.input_tensor_dims.y % 4;
  if (tail_case_size == 0) {
    tail_case_size = 4;
  }
  // Calculate the denominator for the whole dimension.
  // For numerical stability to avoid floating point overflow,
  // we leverage the translation invariance of the softmax function,
  // subtracting every element along channel by the maximum element along
  // channel. find the maximum element
  float max_element = texelFetch(uInput, src_pos, 0)[0];
  for (int c = 0; c < b_stride - 1; c++) {
    const vec4 c_texel =
        texelFetch(uInput, ivec3(src_pos.x, src_pos.y, src_pos.z + c), 0);
    for (int t = 0; t < 4; t++) {
      if (c_texel[t] > max_element) {
        max_element = c_texel[t];
      }
    }
  }
  vec4 c_texel = texelFetch(
      uInput, ivec3(src_pos.x, src_pos.y, src_pos.z + b_stride - 1), 0);
  for (int t = 0; t < tail_case_size; t++) {
    if (c_texel[t] > max_element) {
      max_element = c_texel[t];
    }
  }
  // Calculate the denominator.
  float denominator = 0;
  for (int c = 0; c < b_stride - 1; c++) {
    const vec4 c_texel =
        texelFetch(uInput, ivec3(src_pos.x, src_pos.y, src_pos.z + c), 0);
    for (int t = 0; t < 4; t++) {
      denominator += exp(c_texel[t] - max_element);
    }
  }
  c_texel = texelFetch(
      uInput, ivec3(src_pos.x, src_pos.y, src_pos.z + b_stride - 1), 0);
  for (int t = 0; t < tail_case_size; t++) {
    denominator += exp(c_texel[t] - max_element);
  }
  // Calculate every final channel element.
  for (int c = 0; c < b_stride; c++) {
    const ivec3 dst_pos = ivec3(src_pos.x, src_pos.y, src_pos.z + c);
    const vec4 numerator = exp(texelFetch(uInput, dst_pos, 0) - max_element);
    imageStore(uOutput, dst_pos, numerator / denominator);
  }
}
