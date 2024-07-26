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
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // depth_info.x: number of batches
  // depth_info.y: number of texels per batch
  // depth_info.z: index along channel dim to select
  // depth_info.w: zero pad for alignment
  ivec4 depth_info;
}
uBlock;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  // read in the same channel from 4 separate batches
  vec4 out_texel = vec4(0, 0, 0, 0);
  for (int k = 0; k < 4; k++) {
    if ((k + pos.z * 4) >=
        uBlock.depth_info.x) { // < 4 batches for this texel, exit early
      break;
    }
    const uint src_pos_z = (4 * uBlock.depth_info.y * pos.z) +
        (k * uBlock.depth_info.y) + (uBlock.depth_info.z / 4);
    const uint src_pos_t = uBlock.depth_info.z % 4;
    out_texel[k] =
        texelFetch(uInput, ivec3(pos.x, pos.y, src_pos_z), 0)[src_pos_t];
  }

  imageStore(uOutput, pos, out_texel);
}
