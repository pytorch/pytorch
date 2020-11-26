#version 450 core
#define PRECISION $precision

layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba32f) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION                    sampler3D uInput;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  /* Dynamically Uniform */
  const ivec3 size = imageSize(uOutput);
  const ivec3 isize = textureSize(uInput, 0);
  const float range = isize.x * isize.y;

  if (all(lessThan(pos, size))) {
    vec4 sum = vec4(0);

    for (int y = 0; y < isize.y; ++y) {
      for (int x = 0; x < isize.x; ++x) {
        sum += texelFetch(uInput, ivec3(x, y, pos.z), 0);
      }
    }

    imageStore(
        uOutput,
        pos,
        sum / range);
  }
}
