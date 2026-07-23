#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;

/*
 * Params Buffer
 * input_shader_extents is the dimensions of the Vulkan 3D texture XYZ
 * with a zero pad at W.
 * input_tensor_dims is the dimensions of the NCHW PyTorch Tensor.
 * input_dim_stride is the stride to include elements along the scan
 * dimension calculation. early_exit is the global workgroup position-based
 * condition for unnecessary invocations to exit.
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 input_shader_extents;
  ivec4 input_tensor_dims;
  ivec4 input_dim_stride;
  ivec4 early_exit;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * This shader can compute cumsum along batch, height, and width.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (!all(lessThan(pos, uBlock.early_exit.xyz))) {
    return;
  }
  ivec3 cand_pos = pos;
  vec4 sum = vec4(0, 0, 0, 0);
  while (all(lessThan(cand_pos, uBlock.input_shader_extents.xyz))) {
    sum += texelFetch(uInput, cand_pos, 0);
    imageStore(uOutput, cand_pos, sum);
    cand_pos += uBlock.input_dim_stride.xyz;
  }
}
