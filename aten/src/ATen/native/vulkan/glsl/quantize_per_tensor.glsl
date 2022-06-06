#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput; //input
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  ivec4 size;
  vec2 scale;
  ivec2 zero_point;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);


  //vec4 vec_scale = vec4(uBlock.scale);
  //vec4 vec_zero_point = vec4(uBlock.zero_point.x);

  vec4 ret =  texelFetch(uInput, pos, 0) / uBlock.scale.x + uBlock.zero_point.x;
  ret.x = int(ret.x);
  ret.y = int(ret.y);
  ret.w = int(ret.w);
  ret.z = int(ret.z);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    imageStore(
        uOutput,
        pos,
        ret);
  }
}
