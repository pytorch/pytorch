#version 450 core
//#define PRECISION $precision
#define PRECISION highp

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) buffer  PRECISION restrict writeonly OutputBuffer {
  float data[];
} uOutput;
layout(set = 0, binding = 1) buffer  PRECISION restrict readonly  DetsBuffer   {
  vec4 data[];
} uDets;
layout(set = 0, binding = 2) buffer  PRECISION restrict readonly  IndicesBuffer   {
  float data[];
} uIndices;
layout(set = 0, binding = 3) uniform PRECISION restrict           Block {
  float iou_threshold;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (pos.x < pos.y) {
    return;
  }

  int ai = int(uIndices.data[pos.y]);
  int bi = int(uIndices.data[pos.x]);

  vec4 a = uDets.data[ai];
  vec4 b = uDets.data[bi];

  float left = max(a[0], b[0]);
  float top = max(a[1], b[1]);
  float right = min(a[2], b[2]);
  float bottom = min(a[3], b[3]);

  float width = max(right - left, 0.0);
  float height = max(bottom - top, 0.0);
  float interS = width * height;
  float Sa = (a[2] - a[0]) * (a[3] - a[1]);
  float Sb = (b[2] - b[0]) * (b[3] - b[1]);
  float iou = interS / (Sa + Sb - interS);
  if (iou > uBlock.iou_threshold) {
    uOutput.data[pos.x] = 1.;
  }
}
