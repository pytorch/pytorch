#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  ivec4 size;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// This implementation is suboptimal and should be revisited.

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.z == 0 && all(lessThan(pos, uBlock.size.xyz))) {
    float sum = 0;
    for (int z = 0; z < uBlock.size.z - 1; ++z) {
      const vec4 input_exp = exp(
        texelFetch(uInput, ivec3(pos.x, pos.y, z), 0)
      );
      sum += (input_exp.x + input_exp.y + input_exp.z + input_exp.w);
    }

    const vec4 last_input_exp = exp(
      texelFetch(uInput, ivec3(pos.x, pos.y, uBlock.size.z - 1), 0)
    );
    sum += (
      last_input_exp.x +
      (uBlock.size.w >= 1 ? last_input_exp.y : 0) +
      (uBlock.size.w >= 2 ? last_input_exp.z : 0) +
      (uBlock.size.w == 3 ? last_input_exp.w : 0)
    );

    for (int z = 0; z < uBlock.size.z; ++z) {
      const ivec3 curr_pos = ivec3(pos.x, pos.y, z);
      const vec4 input_exp = exp(texelFetch(uInput, curr_pos, 0));
      imageStore(uOutput, curr_pos, log(input_exp / sum));
    }
  }
}
