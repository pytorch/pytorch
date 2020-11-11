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
  const vec3 isize = textureSize(uInput, 0);
  const vec2 stride = isize.xy / size.xy;
  const vec2 kernel = isize.xy - (size.xy - 1) * stride;

  if (all(lessThan(pos, size))) {
    const vec2 ipos = pos.xy * stride;

    const ivec2 start = ivec2(ipos);
    const ivec2 end = ivec2(ceil(ipos + kernel));
    const ivec2 range = end - start;

    vec4 sum = vec4(0);

    for (int y = start.y; y < end.y; ++y) {
      for (int x = start.x; x < end.x; ++x) {
        sum += texelFetch(uInput, ivec3(x, y, pos.z), 0);
      }
    }

    imageStore(
        uOutput,
        pos,
        sum / (range.x * range.y));
  }
}
