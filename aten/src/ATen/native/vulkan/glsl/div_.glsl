#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION          sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION restrict Block {
  ivec4 size;
  ivec4 isize;
  float alpha;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const ivec3 input_pos = pos % uBlock.isize.xyz;
    vec4 v = uBlock.isize.w == 1
                ? texelFetch(uInput, input_pos, 0).xxxx
                : texelFetch(uInput, input_pos, 0);

    const int c_index = (pos.z % ((uBlock.size.w + 3) / 4)) * 4;
    if (uBlock.isize.w != 1 && c_index + 3 >= uBlock.size.w) {
      ivec4 c_ind = ivec4(c_index) + ivec4(0, 1, 2, 3);
      vec4 mask = vec4(lessThan(c_ind, ivec4(uBlock.size.w)));
      v = v * mask + vec4(1, 1, 1, 1) - mask;
    }

    imageStore(
        uOutput,
        pos,
        imageLoad(uOutput, pos) / v);
  }
}
