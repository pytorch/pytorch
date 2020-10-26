#version 450 core
#define PRECISION $precision

layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - precision - memory - invariance - precise */

layout(set = 0, binding = 0, rgba16f) PRECISION writeonly uniform image3D   uOutput;
layout(set = 0, binding = 1)          PRECISION           uniform sampler3D uInput0;
layout(set = 0, binding = 2)          PRECISION           uniform sampler3D uInput1;
layout(set = 0, binding = 3)                              uniform Block {
  ivec3 WHC;
  float alpha;
} block;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, block.WHC))) {
    imageStore(
        uOutput,
        pos,
        texelFetch(uInput0, pos, 0) + block.alpha * texelFetch(uInput1, pos, 0));
  }
}
