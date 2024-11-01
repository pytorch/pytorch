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
layout(set = 0, binding = 2) uniform PRECISION sampler3D uGamma;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uBeta;
layout(set = 0, binding = 4) uniform PRECISION sampler3D uMean;
layout(set = 0, binding = 5) uniform PRECISION sampler3D uVar;

/*
 * Params Buffer
 */
layout(set = 0, binding = 6) uniform PRECISION restrict Block {
  // xyz contains extents of the output texture, w contains the number of
  // channels divided by 4, rounded up.
  ivec4 out_extents;
  float eps;
}
uBlock;

/*
 * Local Work Group
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Computes a Batch normalization. Each shader invocation calculates the output
 * at a single output location.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // Return if this global position is outside output texture bounds
  if (any(greaterThanEqual(pos, uBlock.out_extents.xyz))) {
    return;
  }

  const ivec3 ch_pos = ivec3(0, 0, pos.z % uBlock.out_extents.w);

  const vec4 in_tex = texelFetch(uInput, pos, 0);
  const vec4 gamma_tex = texelFetch(uGamma, ch_pos, 0);
  const vec4 beta_tex = texelFetch(uBeta, ch_pos, 0);
  const vec4 mean_tex = texelFetch(uMean, ch_pos, 0);
  const vec4 var_tex = texelFetch(uVar, ch_pos, 0);

  const vec4 out_tex =
      (in_tex - mean_tex) / sqrt(var_tex + uBlock.eps) * gamma_tex + beta_tex;

  imageStore(uOutput, pos, out_tex);
}
