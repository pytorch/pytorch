#version 450 core
//#define PRECISION $precision
#define PRECISION highp

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) buffer  PRECISION restrict writeonly OutputBuffer  {
  float data[];
} uOutput;
layout(set = 0, binding = 1) buffer  PRECISION restrict writeonly OutputKeepBuffer  {
  float data[];
} uOutputKeep;
layout(set = 0, binding = 2) buffer  PRECISION restrict readonly  ImInfosBuffer {
  float data[];
} uImInfos;
layout(set = 0, binding = 3) buffer  PRECISION restrict readonly  AnchorsBuffer {
  float data[];
} uAnchors;
layout(set = 0, binding = 4) buffer  PRECISION restrict readonly  DeltasBuffer  {
  float data[];
} uDeltas;
layout(set = 0, binding = 5) uniform PRECISION restrict           Block {
  int height;
  int width;
  float feat_stride;
  float legacy_plus_one;
  float bbox_xform_clip;
  float min_size;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  // pos.x - H*W
  // pos.y - A
  // pos.z - N

  int w_i = pos.x % uBlock.width;
  int h_i = pos.y / uBlock.width;
  int n_i = pos.z;
  int a_i = pos.y;
  
  //TODO: check ordering
  float im_height = uImInfos.data[3 * n_i + 0];
  float im_width = uImInfos.data[3 * n_i + 1];

  // support box_dim == 5
  int box_dim = 4;
  float x0 = uAnchors.data[a_i * box_dim + 0];
  float y0 = uAnchors.data[a_i * box_dim + 1];
  float x1 = uAnchors.data[a_i * box_dim + 2];
  float y1 = uAnchors.data[a_i * box_dim + 3];
  
  // anchor coords
  // h_i w_i shift
  // bbox_transform

  // clip by im_info.w im_info.h, clip_angle, legacy_plus_one
  // filter by min_size

  float w = x1 - x0 + uBlock.legacy_plus_one;
  float h = y1 - y0 + uBlock.legacy_plus_one;

  float center_x = x0 + 0.5 * w;
  float center_y = y0 + 0.5 * h;

  int deltasIdx = 0; //TODO:
  float dx = uDeltas.data[deltasIdx + 0];
  float dy = uDeltas.data[deltasIdx + 0];
  float dw = min(uDeltas.data[deltasIdx + 2], uBlock.bbox_xform_clip);
  float dh = min(uDeltas.data[deltasIdx + 3], uBlock.bbox_xform_clip);

  float pred_center_x = dx * w + center_x;
  float pred_center_y = dy * h + center_y;
  float pred_w = exp(dw) * w;
  float pred_h = exp(dh) * h;

  float res_x0 = pred_center_x - 0.5 * w;
  float res_y0 = pred_center_y - 0.5 * h;
  float res_x1 = pred_center_x + 0.5 * w + uBlock.legacy_plus_one;
  float res_y1 = pred_center_y + 0.5 * h + uBlock.legacy_plus_one;

  // clip TODO: support clip by angle threshold
  res_x0 = min(max(0, res_x0), im_width);
  res_y0 = min(max(0, res_y0), im_height);
  res_x1 = min(max(0, res_x1), im_width);
  res_y1 = min(max(0, res_y1), im_height);

  bool keep = ((res_x1 - res_x0) > uBlock.min_size) && ((res_y1 - res_y0) > uBlock.min_size);

}
