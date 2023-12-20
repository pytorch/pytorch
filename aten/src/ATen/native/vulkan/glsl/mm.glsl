#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

// To convince the SPIR-V compiler to unroll the loops optimally, need this
// macro
#define FOUR 4
layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uM1;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uM2;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 shader_extents;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.shader_extents.xyz))) {
    // we avoid mat4 and vec4 usage here as they compile to much less efficient
    // SPIR-V
    float results[FOUR][FOUR];
    for (int i = 0; i < FOUR; i++) {
      for (int j = 0; j < FOUR; j++) {
        results[i][j] = 0;
      }
    }

    for (int j = 0; j < uBlock.shader_extents.w; j++) {
      // we may potentially read out of bounds, but (0, 0, 0, 0) will be sampled
      // safely read and cache 4x4 tile of uM1 (4 adjacent rows)
      vec4 uM1_partial_rows[FOUR];
      vec4 uM2_partial_cols[FOUR];

      for (int k = 0; k < FOUR; k++) {
        const int pos_y_offset = (FOUR * pos.y) + k;
        const ivec3 pos_rd = ivec3(j, pos_y_offset, pos.z);
        uM1_partial_rows[k] = texelFetch(uM1, pos_rd, 0);
      }
      // read and cache 4x4 tile of uM2 (4 adjacent columns)
      for (int k = 0; k < FOUR; k++) {
        const int pos_x_offset = (FOUR * pos.x) + k;
        const ivec3 pos_rd = ivec3(pos_x_offset, j, pos.z);
        uM2_partial_cols[k] = texelFetch(uM2, pos_rd, 0);
      }
      // perform partial dot products and add partial result to results
      for (int idx_r = 0; idx_r < FOUR; idx_r++) {
        for (int idx_c = 0; idx_c < FOUR; idx_c++) {
          results[idx_r][idx_c] +=
              dot(uM1_partial_rows[idx_r], uM2_partial_cols[idx_c]);
        }
      }
    }
    // results is in transposed order w.r.t. the desired output
    for (int idx_c = 0; idx_c < FOUR; idx_c++) {
      for (int idx_r = 0; idx_r < FOUR; idx_r++) {
        const ivec3 out_pos =
            ivec3(idx_r + FOUR * pos.x, idx_c + FOUR * pos.y, pos.z);
        imageStore(
            uOutput, out_pos, vec4(results[idx_c][idx_r], 0.0, 0.0, 0.0));
      }
    }
  }
}
