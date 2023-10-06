#version 450 core
#define PRECISION $precision
#define FORMAT $format

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

/*
 * This shader can compute softmax along batch, height, and width.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (!all(lessThan(pos, uBlock.early_exit.xyz))) {
    return;
  }
  // Calculate the denominator for the whole dimension.
  // For numerical stability to avoid floating point overflow,
  // we leverage the translation invariance of the softmax function,
  // subtracting every element along input_dim_stride by
  // the maximum element along input_dim_stride.
  // find the maximum element
  vec4 max_element = texelFetch(uInput, pos, 0);
  ivec3 cand_pos = pos + uBlock.input_dim_stride.xyz;
  while (all(lessThan(cand_pos, uBlock.input_shader_extents.xyz))) {
    max_element = max(texelFetch(uInput, cand_pos, 0), max_element);
    cand_pos += uBlock.input_dim_stride.xyz;
  }
  // Calculate the denominator along the direction of input_dim_stride.
  cand_pos = pos;
  vec4 denominator = vec4(0, 0, 0, 0);
  while (all(lessThan(cand_pos, uBlock.input_shader_extents.xyz))) {
    denominator += exp(texelFetch(uInput, cand_pos, 0) - max_element);
    cand_pos += uBlock.input_dim_stride.xyz;
  }
  // Calculate every final element along the direction of input_dim_stride.
  cand_pos = pos;
  while (all(lessThan(cand_pos, uBlock.input_shader_extents.xyz))) {
    const vec4 numerator = exp(texelFetch(uInput, cand_pos, 0) - max_element);
    imageStore(uOutput, cand_pos, numerator / denominator);
    cand_pos += uBlock.input_dim_stride.xyz;
  }
}
