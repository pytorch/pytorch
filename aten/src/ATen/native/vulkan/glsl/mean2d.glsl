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

/*
 * Params Buffer
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // extents of the output texture
  // w contains pre-computed H*W of the input texture for convenience
  ivec4 out_extents;
  // extents of the input texture
  // w contains size of input channels aligned to 4
  ivec4 in_extents;
}
uBlock;

/*
 * Shared memory buffer
 */
shared vec4 sh_mem[64];

/*
 * Local Work Group
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Computes the mean of an input tensor along the width and height axes.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec3 tid = ivec3(gl_LocalInvocationID);
  const ivec3 group_size = ivec3(gl_WorkGroupSize);

  if (pos.z < uBlock.in_extents.z) {
    vec4 sum = vec4(0);

    for (int y = tid.y; y < uBlock.in_extents.y; y += group_size.y) {
      for (int x = tid.x; x < uBlock.in_extents.x; x += group_size.x) {
        sum += texelFetch(uInput, ivec3(x, y, pos.z), 0);
      }
    }

    sh_mem[tid.z * group_size.y * group_size.x + tid.y * group_size.x + tid.x] =
        sum;
  }
  memoryBarrierShared();
  barrier();

  if (tid.y > 0 || tid.x > 0 || pos.z >= uBlock.in_extents.z) {
    return;
  }

  vec4 total = vec4(0);
  for (int y = 0; y < group_size.y; ++y) {
    for (int x = 0; x < group_size.x; ++x) {
      total +=
          sh_mem[tid.z * group_size.y * group_size.x + y * group_size.x + x];
    }
  }

  const vec4 outtex = total / uBlock.out_extents.w;

  const int nc_idx = pos.z * 4;
  const int out_width = uBlock.out_extents.x;
  const int out_height = uBlock.out_extents.y;

  for (int i = 0; i < 4; ++i) {
    const int n_idx = (nc_idx + i) / uBlock.in_extents.w;
    const int c_idx = (nc_idx + i) % uBlock.in_extents.w;

    ivec3 pos = ivec3(c_idx, n_idx, 0);
    if (c_idx < out_width && n_idx < out_height) {
      imageStore(uOutput, pos, vec4(outtex[i], 0, 0, 0));
    }
  }
}
