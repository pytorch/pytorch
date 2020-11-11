#version 450 core
#define PRECISION $precision

layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba32f) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION                    sampler3D uM1;
layout(set = 0, binding = 2)          uniform PRECISION                    sampler3D uM2;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  /* Dynamically Uniform */
  const ivec3 size = imageSize(uOutput);
  const int dim = textureSize(uM1, 0).x;

  if (all(lessThan(pos, size))) {
    vec4 sum = vec4(0);

    for (int k = 0; k < dim; ++k) {
      sum = fma(
          texelFetch(uM1, ivec3(k, pos.y, pos.z), 0),
          texelFetch(uM2, ivec3(pos.x, k, pos.z), 0),
          sum);
    }

    imageStore(uOutput, pos, sum);
  }
}
