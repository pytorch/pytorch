#version 450 core
//#define PRECISION $precision
#define PRECISION highp

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) buffer  PRECISION restrict writeonly OutputBuffer {
  float data[];
} uOutput;
layout(set = 0, binding = 1) buffer  PRECISION restrict readonly  RoisBuffer   {
  float data[];
} uRois;
layout(set = 0, binding = 2) buffer  PRECISION restrict readonly  DeltasBuffer {
  float data[];
} uDeltas;
layout(set = 0, binding = 3) buffer  PRECISION restrict readonly  ImInfosBuffer {
  float data[];
} uImInfos;
layout(set = 0, binding = 4) uniform PRECISION restrict           Block {
  float legacy_plus_one;
  vec4 weights;
  int num_classes;
  float bbox_xform_clip;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const int im_i = pos.x;
  const int class_i = pos.y;
  const int batch_i = pos.z;

  // assume box_dim == 4
  //TODO: support box_dim == 5
  float x0 = uRois.data[4 * im_i + 0];
  float y0 = uRois.data[4 * im_i + 1];
  float x1 = uRois.data[4 * im_i + 2];
  float y1 = uRois.data[4 * im_i + 3];

  float w = x1 - x0 + uBlock.legacy_plus_one;
  float h = y1 - y0 + uBlock.legacy_plus_one;

  float center_x = x0 + 0.5 * w;
  float center_y = y0 + 0.5 * h;

  float dx = uDeltas.data[im_i * 4 * uBlock.num_classes + 0] / uBlock.weights[0];
  float dy = uDeltas.data[im_i * 4 * uBlock.num_classes + 1] / uBlock.weights[1];
  float dw = min(
      uDeltas.data[im_i * 4 * uBlock.num_classes + 2] / uBlock.weights[2],
      uBlock.bbox_xform_clip);
  float dh = min(
      uDeltas.data[im_i * 4 * uBlock.num_classes + 3] / uBlock.weights[3],
      uBlock.bbox_xform_clip);

  float pred_center_x = dx * w + center_x;
  float pred_center_y = dy * h + center_y;
  float pred_w = exp(dw) * w;
  float pred_h = exp(dh) * h;

  const int outputIdx = 4 * (uBlock.num_classes * (batch_i + im_i) + class_i);
  uOutput.data[outputIdx + 0] = pred_center_x - 0.5 * w;
  uOutput.data[outputIdx + 1] = pred_center_y - 0.5 * h;
  uOutput.data[outputIdx + 2] = pred_center_x + 0.5 * w + uBlock.legacy_plus_one;
  uOutput.data[outputIdx + 3] = pred_center_y + 0.5 * h + uBlock.legacy_plus_one;
}
