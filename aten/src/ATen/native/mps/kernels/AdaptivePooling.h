#pragma once

struct AdaptiveAvgPool2DParams {
  long B;
  long C;
  long input_height;
  long input_width;
  long output_height;
  long output_width;
  long input_strides[4];
  long output_strides[4];
};
