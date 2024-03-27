#version 450 core

#define PRECISION $precision
#define FORMAT $format

#include "indexing.h"

layout(std430) buffer;

/*
 * Output Buffer
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;

/*
 * Output Buffer Metadata
 */
layout(set = 0, binding = 1) uniform PRECISION restrict OutMeta {
  uvec4 sizes;
  uvec4 strides;
  uint ndim;
  uint buf_length;
}
uOutMeta;

/*
 * Input Buffer
 */
layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer {
  float data[];
}
uInput;

/*
 * Input Buffer Metadata
 */
layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 sizes;
  uvec4 strides;
  uint ndim;
  uint buf_length;
}
uInMeta;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Copies data from the tensor at uInput to the tensor at uOutput based on 4D
 * coordinate. Each element at (x,y,c,n) in uInput will be copied to uOutput at
 * (x,y,c,n). If (x,y,c,n) is outside the bounds of uInput then 0 will be
 * written.
 *
 * Each shader invocation is responsible for one element of the output buffer.
 */
void main() {
  const uint write_idx = ivec3(gl_GlobalInvocationID).x;

  if (write_idx >= uOutMeta.buf_length) {
    return;
  }

  uvec4 write_coord =
      idx_to_coord(write_idx, uOutMeta.strides, uOutMeta.sizes);

  float outval = 0u;
  if (all(lessThan(write_coord, uInMeta.sizes))) {
    uint read_idx = coord_to_idx(write_coord, uInMeta.strides);
    outval = uInput.data[read_idx];
  }

  uOutput.data[write_idx] = outval;
}
