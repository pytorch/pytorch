#version 450 core
#define PRECISION $precision

layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba32f) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION                    sampler3D uInput0;
layout(set = 0, binding = 2)          uniform PRECISION                    sampler3D uInput1;
layout(set = 0, binding = 3)          uniform PRECISION restrict           Block {
  float alpha;
} uBlock;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  /* Dynamically Uniform */
  const ivec3 size = imageSize(uOutput);

  if (all(lessThan(pos, size))) {
    imageStore(
        uOutput,
        pos,
        texelFetch(uInput0, pos, 0) + uBlock.alpha * texelFetch(uInput1, pos, 0));
  }
}
