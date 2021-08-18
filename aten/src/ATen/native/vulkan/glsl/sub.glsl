#version 450 core
#define PRECISION $precision

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1) uniform PRECISION                    sampler3D uInput0;
layout(set = 0, binding = 2) uniform PRECISION                    sampler3D uInput1;
layout(set = 0, binding = 3) uniform PRECISION restrict           Block {
  ivec4 size;
  ivec4 isize0;
  ivec3 isize1;
  float alpha;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec3 input0_pos = pos;
    const ivec3 input1_pos = ivec3(pos.x, 0, 0);

    const vec4 in0tex = texelFetch(uInput0, input0_pos, 0);
    const vec4 in1tex = texelFetch(uInput1, input1_pos, 0);
    vec4 outtex = in0tex - uBlock.alpha * in1tex;

    imageStore(uOutput, pos, outtex);
  }
}
