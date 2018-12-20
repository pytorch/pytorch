#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/SpatialUpSamplingBicubic.c"
#else

#include <THNN/generic/upsampling.h>

static inline void THNN_(SpatialUpSamplingBicubic_shapeCheck)
     (THTensor *input, THTensor *gradOutput,
      int nBatch, int nChannels,
      int input_height, int input_width,
      int output_height, int output_width) {
  THArgCheck(input_height > 0 && input_width > 0
	     && output_height > 0 && output_width > 0, 2,
	     "input and output sizes should be greater than 0,"
	     " but got input (H: %d, W: %d) output (H: %d, W: %d)",
	     input_height, input_width, output_height, output_width);
  if (input != NULL) {
    THNN_ARGCHECK(!input->is_empty() && input->dim() == 4, 2, input,
		  "non-empty 4D input tensor expected but got: %s");
  }

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 0, nBatch);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 1, nChannels);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 2, output_height);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 3, output_width);
  }
}

void THNN_(SpatialUpSamplingBicubic_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int output_height,
    int output_width,
    bool align_corners) {

  const int nbatch = THTensor_(size)(input, 0);
  const int channels = THTensor_(size)(input, 1);
  const int input_height = THTensor_(size)(input, 2);
  const int input_width = THTensor_(size)(input, 3);

  THNN_(SpatialUpSamplingBicubic_shapeCheck)
    (input, NULL,
     nbatch, channels,
     input_height, input_width,
     output_height, output_width);

  input = THTensor_(newContiguous)(input);
  THTensor_(resize4d)(output,
          THTensor_(size)(input, 0),
          THTensor_(size)(input, 1),
          output_height, output_width);
  THTensor_(zero)(output);
  scalar_t *idata = input->data<scalar_t>();
  scalar_t *odata = output->data<scalar_t>();

  // Special case: input/output same size, just copy
  if (input_height == output_height && input_width == output_width) {
    for (int output_y = 0; output_y < output_height; output_y++) {
      for (int output_x = 0; output_x < output_width; output_x++) {
        const scalar_t* in = &idata[output_y * input_width + output_x];
        scalar_t* out = &odata[output_y * output_width + output_x];
        for (int c = 0; c < channels; ++c) {
          out[0] = in[0];
          in += input_width * input_height;
          out += output_width * output_height;
        }
      }
    }
    c10::raw::intrusive_ptr::decref(input);
    return;
  }

  // Bicubic interpolation
  const accreal height_scale = linear_upsampling_compute_scale<accreal>(
    input_height,
    output_height,
    align_corners);
  const accreal width_scale = linear_upsampling_compute_scale<accreal>(
    input_width,
    output_width,
    align_corners);

  for (int output_y = 0; output_y < output_height; output_y++) {
    for (int output_x = 0; output_x < output_width; output_x++) {
      scalar_t* in = idata;
      scalar_t* out = odata;

      const scalar_t real_x = width_scale * output_x;
      int input_x = real_x;
      const scalar_t t_x = real_x - input_x;

      const scalar_t real_y = height_scale * output_y;
      int input_y = real_y;
      const scalar_t t_y = real_y - input_y;

      for (int c = 0; c < channels * nbatch; c++) {
        scalar_t coefficients[4];

        // Interpolate 4 times in the x direction
        for (int i = 0; i < 4; i++) {
          coefficients[i] = cubic_interp1d<scalar_t>(
            upsampling_get_value_bounded<scalar_t>(
              in, input_width, input_height, input_x - 1, input_y - 1 + i),
            upsampling_get_value_bounded<scalar_t>(
              in, input_width, input_height, input_x + 0, input_y - 1 + i),
            upsampling_get_value_bounded<scalar_t>(
              in, input_width, input_height, input_x + 1, input_y - 1 + i),
            upsampling_get_value_bounded<scalar_t>(
              in, input_width, input_height, input_x + 2, input_y - 1 + i),
            t_x
          );
        }

        // Interpolate in the y direction using x interpolations
        out[output_y * output_width + output_x] = cubic_interp1d<scalar_t>(
          coefficients[0],
          coefficients[1],
          coefficients[2],
          coefficients[3],
          t_y
        );

        // Move to next channel
        in += input_width * input_height;
        out += output_width * output_height;
      }
    }
  }

  c10::raw::intrusive_ptr::decref(input);
}

void THNN_(SpatialUpSamplingBicubic_updateGradInput)(
    THNNState *state,
    THTensor *gradOutput,
    THTensor *gradInput,
    int nbatch,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    bool align_corners){

  THNN_(SpatialUpSamplingBicubic_shapeCheck)
    (NULL, gradOutput,
     nbatch, channels,
     input_height, input_width,
     output_height, output_width);

  THTensor_(resize4d)(gradInput, nbatch, channels, input_height, input_width);
  THTensor_(zero)(gradInput);

  gradOutput = THTensor_(newContiguous)(gradOutput);
  scalar_t *idata = gradInput->data<scalar_t>();
  scalar_t *odata = gradOutput->data<scalar_t>();
  channels = nbatch * channels;

  // Special case: input/output same size, just copy
  if (input_height == output_height && input_width == output_width) {
    for (int output_y = 0; output_y < output_height; output_y++) {
      for (int output_x = 0; output_x < output_width; output_x++) {
        scalar_t* in = &idata[output_y * input_width + output_x];
        scalar_t* out = &odata[output_y * output_width + output_x];
        for (int c = 0; c < channels; ++c) {
          in[0] = out[0];
          in += input_width * input_height;
          out += output_width * output_height;
        }
      }
    }
    c10::raw::intrusive_ptr::decref(gradOutput);
    return;
  }

  const accreal height_scale = linear_upsampling_compute_scale<accreal>(
    input_height, output_height, align_corners);
  const accreal width_scale = linear_upsampling_compute_scale<accreal>(
    input_width, output_width, align_corners);

  for (int output_y = 0; output_y < output_height; output_y++) {
    for (int output_x = 0; output_x < output_width; output_x++) {
      scalar_t* in = idata;
      scalar_t* out = odata;

      scalar_t real_x = width_scale * output_x;
      int input_x = real_x;
      scalar_t t_x = real_x - input_x;

      scalar_t real_y = height_scale * output_y;
      int input_y = real_y;
      scalar_t t_y = real_y - input_y;

      scalar_t x_coeffs[4];
      scalar_t y_coeffs[4];

      get_cubic_upsampling_coefficients<scalar_t>(x_coeffs, t_x);
      get_cubic_upsampling_coefficients<scalar_t>(y_coeffs, t_y);


      for (int c = 0; c < channels; c++) {
        scalar_t out_value = out[output_y * output_width + output_x];

        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 4; j++) {
            upsampling_increment_value_bounded<scalar_t>(in,
              input_width,
              input_height,
              input_x - 1 + i,
              input_y - 1 + j,
              out_value * y_coeffs[j] * x_coeffs[i]);
          }
        }

        in += input_width * input_height;
        out += output_width * output_height;
      }
    }
  }

  c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif
