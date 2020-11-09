#version 450 core
#define PRECISION $precision
layout(std430) buffer;
layout(std430) uniform;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, rgba16f) uniform PRECISION writeonly image3D   uOutput;
layout(set = 0, binding = 1)          uniform PRECISION           sampler3D uM1;
layout(set = 0, binding = 2)          uniform PRECISION           sampler3D uM2;
layout(set = 0, binding = 3)          uniform           restrict  Block {
  ivec3 WHC;
  float beta;
  float alpha;
  int K;
} uBlock;
layout(set = 0, binding = 4)          uniform PRECISION           sampler3D uT;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.WHC))) {
    const int K = uBlock.K;
    vec4 mmv = vec4(0);
    int ki = 0;
    for (; ki < K; ++ki) {
      vec4 m1ki = texelFetch(uM1, ivec3(ki, pos.y, pos.z), 0);
      vec4 m2ki = texelFetch(uM2, ivec3(pos.x, ki, pos.z), 0);
      mmv += m1ki * m2ki;
    }
    vec4 tv = texelFetch(uT, pos, 0);
    imageStore(uOutput, pos, uBlock.beta * tv + uBlock.alpha * mmv);
  }
}
