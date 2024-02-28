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
  // width_info.x: number of batches
  // width_info.y: number of texels per batch
  // width_info.z: index along width dim to select
  // width_info.w: zero pad for alignment
  ivec4 width_info;
}
uBlock;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  vec4 out_texel = vec4(0, 0, 0, 0);
  // read in the same channel from 4 separate batches
  for (int k = 0; k < 4; k++) {
    if ((k + pos.z * 4) >=
        uBlock.width_info.x) { // < 4 batches for this texel, exit early
      break;
    }
    const uint src_pos_z = (pos.z * uBlock.width_info.y * 4) +
        k * uBlock.width_info.y + (pos.y / 4);
    out_texel[k] = texelFetch(
        uInput, ivec3(uBlock.width_info.z, pos.x, src_pos_z), 0)[pos.y % 4];
  }
  imageStore(uOutput, pos, out_texel);
}
