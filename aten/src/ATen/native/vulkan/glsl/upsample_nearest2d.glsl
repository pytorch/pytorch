#version 450 core
#define PRECISION $precision

layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba32f) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)          uniform PRECISION restrict           Block {
  vec2 scale;
} uBlock;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  /* Dynamically Uniform */
  const ivec3 size = imageSize(uOutput);
  const ivec3 isize = textureSize(uInput, 0);

  if (all(lessThan(pos, size))) {
    const ivec2 ipos = clamp(
        ivec2(pos.xy * uBlock.scale),
        ivec2(0),
        isize.xy - 1);

    imageStore(
        uOutput,
        pos,
        texelFetch(uInput, ivec3(ipos, pos.z), 0));
  }
}
