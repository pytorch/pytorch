#version 450 core
//#define PRECISION $precision
#define PRECISION highp

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0) buffer  PRECISION restrict writeonly OutputBuffer {
  float data[];
} uOutput;
layout(set = 0, binding = 1) buffer  PRECISION restrict readonly  InputBuffer  {
  float data[];
} uInput;
layout(set = 0, binding = 2) buffer  PRECISION restrict readonly  RoisBuffer   {
  float data[];
} uRois;
layout(set = 0, binding = 3) uniform PRECISION restrict           Block {
  ivec2 input_size;
  ivec2 pooled_size;
  int num_rois;
  int roi_cols;
  int channels;
  int sampling_ratio;
  float spatial_scale;
  float offset;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const int roi_i = pos.z / uBlock.channels;
  const int c_i = pos.z % uBlock.channels;
  const int roi_idx = uBlock.roi_cols * roi_i;

  vec2 roi_start, roi_end, roi;

  roi_start.x = uRois.data[roi_idx + 1] * uBlock.spatial_scale - uBlock.offset;
  roi_start.y = uRois.data[roi_idx + 2] * uBlock.spatial_scale - uBlock.offset;
  roi_end.x = uRois.data[roi_idx + 3] * uBlock.spatial_scale - uBlock.offset;
  roi_end.y = uRois.data[roi_idx + 4] * uBlock.spatial_scale - uBlock.offset;
  roi = max(roi_end - roi_start, vec2(1.0f));

  // sampling_ratio branch in separate shader?
  ivec2 bin_grid = ivec2(ceil(roi / vec2(uBlock.pooled_size)));
  vec2 bin_size = roi / vec2(uBlock.pooled_size);
  
  const float scale = 1.0f / (bin_grid.x * bin_grid.y);
  
  const int W = uBlock.input_size[0];
  const int H = uBlock.input_size[1];
  const int PW = uBlock.pooled_size[0];
  const int PH = uBlock.pooled_size[1];

  vec2 bin_grid_size = bin_size / bin_grid;

  int inputBaseIdx = c_i * H * W;

  float sum = 0;
  vec2 p, pw, qw;
  ivec2 pl, ph;
  for (int iy = 0; iy < bin_grid.y; ++iy) {
    p.y = roi_start.y + pos.y * bin_size.y + (iy + 0.5f) * bin_grid_size.y;
    p.y = min(max(p.y, 0), H - 1);

    for (int ix = 0; ix < bin_grid.x; ++ix) {
      p.x = roi_start.x + pos.x * bin_size.x + (ix + 0.5f) * bin_grid_size.x;
      p.x = min(max(p.x, 0), W - 1);

      pl = ivec2(floor(p));
      ph = min(pl + ivec2(1), ivec2(W - 1, H - 1));

      pw = p - pl;
      qw = vec2(1.f) - pw;

      float w1 = qw.y * qw.x;
      float w2 = qw.y * pw.x;
      float w3 = pw.y * qw.x;
      float w4 = pw.y * pw.x;
      
      sum += w1 * uInput.data[inputBaseIdx + pl.y * W + pl.x];
      sum += w2 * uInput.data[inputBaseIdx + pl.y * W + ph.x];
      sum += w3 * uInput.data[inputBaseIdx + ph.y * W + pl.x];
      sum += w4 * uInput.data[inputBaseIdx + ph.y * W + ph.x];
    }
  }

  const int outputIdx =
      roi_i * uBlock.channels * PH * PW +
      c_i * PH * PW +
      pos.y * PW +
      pos.x;

  uOutput.data[outputIdx] = sum * scale; 
}
