#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialFullConvolutionMap.c"
#else

void THNN_(SpatialFullConvolutionMap_updateOutput)(
  THNNState *state, THTensor *input, THTensor *output_, THTensor *weight, THTensor *bias,
  THTensor *connTable, int nInputPlane, int nOutputPlane,
  int dW, int dH)
{
  THArgCheck(THTensor_(isContiguous)(weight), 4, "weight must be contiguous");
  THArgCheck(!bias || THTensor_(isContiguous)(bias), 5, "bias must be contiguous");
  THArgCheck(
    weight != NULL && weight->nDimension == 3
    && connTable != NULL && connTable->size[0] == weight->size[0], 4,
    "3D weight tensor expected (connTable:size(%d) x kH x kW)", TH_INDEX_BASE
  );

  const int kH = (int)weight->size[1];
  const int kW = (int)weight->size[2];

  THArgCheck(input != NULL && input->nDimension == 3, 2, "3D tensor expected");
  THArgCheck(input->size[0] >= nInputPlane, 2, "invalid number of input planes");

  THTensor_(resize3d)(
    output_, nOutputPlane,
    (input->size[1] - 1) * dH + kH,
    (input->size[2] - 1) * dW + kW
  );

  /* contiguous */
  input = THTensor_(newContiguous)(input);
  THTensor* output = THTensor_(newContiguous)(output_);

  /* get raw pointers */
  real *input_data = THTensor_(data)(input);
  real *output_data = THTensor_(data)(output);
  real *weight_data = THTensor_(data)(weight);
  real *bias_data = THTensor_(data)(bias);
  real *connTable_data = THTensor_(data)(connTable);

  /* and dims */
  const int64_t input_h = input->size[1];
  const int64_t input_w = input->size[2];
  const int64_t output_h = output->size[1];
  const int64_t output_w = output->size[2];
  const int64_t weight_h = weight->size[1];
  const int64_t weight_w = weight->size[2];

  int64_t p;
#pragma omp parallel for private(p)
  for (p = 0; p < nOutputPlane; p++)
  {
    /* add bias */
    real *ptr_output = output_data + p*output_w*output_h;
    int64_t j;
    int nweight;
    int64_t k;

    for (j = 0; j < output_h*output_w; j++)
      ptr_output[j] = bias_data[p];

    /* convolve all maps */
    nweight = connTable->size[0];
    for (k = 0; k < nweight; k++)
    {
      /* get offsets for input/output */
      int o = (int)connTable_data[k*2+1] - TH_INDEX_BASE;
      int i = (int)connTable_data[k*2+0] - TH_INDEX_BASE;

      if (o == p)
      {
        THTensor_(fullConv2Dptr)(
          output_data + o*output_w*output_h,
          1.0,
          input_data + i*input_w*input_h, input_h, input_w,
          weight_data + k*weight_w*weight_h, weight_h, weight_w,
          dH, dW
        );
      }
    }
  }

  /* clean up */
  THTensor_(free)(input);
  THTensor_(freeCopyTo)(output, output_);
}

void THNN_(SpatialFullConvolutionMap_updateGradInput)(
  THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradInput_, THTensor *weight, THTensor *bias,
  THTensor *connTable, int nInputPlane, int nOutputPlane,
  int dW, int dH)
{
  THArgCheck(
    weight != NULL && weight->nDimension == 3
    && connTable != NULL && connTable->size[0] == weight->size[0], 5,
    "3D weight tensor expected (connTable:size(%d) x kH x kW)", TH_INDEX_BASE
  );

  /* contiguous */
  THTensor* gradInput = THTensor_(newContiguous)(gradInput_);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* Resize/Zero */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  /* get raw pointers */
  real *gradInput_data = THTensor_(data)(gradInput);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  real *weight_data = THTensor_(data)(weight);
  real *connTable_data = THTensor_(data)(connTable);

  /* and dims */
  const int64_t input_h = input->size[1];
  const int64_t input_w = input->size[2];
  const int64_t output_h = gradOutput->size[1];
  const int64_t output_w = gradOutput->size[2];
  const int64_t kH = weight->size[1];
  const int64_t kW = weight->size[2];

  int64_t p;
#pragma omp parallel for private(p)
  for (p = 0; p < nInputPlane; p++)
  {
    int64_t k;
    /* backward all */
    int nkernel = connTable->size[0];
    for (k = 0; k < nkernel; k++)
    {
      int o = (int)connTable_data[k*2+1] - TH_INDEX_BASE;
      int i = (int)connTable_data[k*2+0] - TH_INDEX_BASE;
      if (i == p)
      {
        /* gradient to input */
        THTensor_(validXCorr2Dptr)(
          gradInput_data + i*input_w*input_h,
          1.0,
          gradOutput_data + o*output_w*output_h,  output_h,  output_w,
          weight_data + k*kW*kH, kH, kW,
          dH, dW
        );
      }
    }
  }

  /* clean up */
  THTensor_(freeCopyTo)(gradInput, gradInput_);
  THTensor_(free)(gradOutput);
}

void THNN_(SpatialFullConvolutionMap_accGradParameters)(
  THNNState *state,
  THTensor *input,
  THTensor *gradOutput,
  THTensor *gradWeight,
  THTensor *gradBias,
  THTensor *connTable,
  int nInputPlane,
  int nOutputPlane,
  int dW, int dH,
  accreal scale_)
{
  real scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  THArgCheck(
    gradWeight != NULL && gradWeight->nDimension == 3
    && connTable != NULL && connTable->size[0] == gradWeight->size[0], 5,
    "3D gradWeight tensor expected (connTable:size(%d) x kH x kW)", TH_INDEX_BASE
  );

  /* contiguous */
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* get raw pointers */
  real *input_data = THTensor_(data)(input);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  real *gradWeight_data = THTensor_(data)(gradWeight);
  real *gradBias_data = THTensor_(data)(gradBias);

  /* and dims */
  const int64_t input_h  = input->size[1];
  const int64_t input_w  = input->size[2];
  const int64_t output_h = gradOutput->size[1];
  const int64_t output_w = gradOutput->size[2];
  const int64_t weight_h = gradWeight->size[1];
  const int64_t weight_w = gradWeight->size[2];

  /* gradients wrt bias */
  int64_t k;
#pragma omp parallel for private(k)
  for (k = 0; k < nOutputPlane; k++)
  {
    real *ptr_gradOutput = gradOutput_data + k*output_w*output_h;
    int64_t l;
    for (l = 0; l < output_h*output_w; l++)
      gradBias_data[k] += scale*ptr_gradOutput[l];
  }

  /* gradients wrt weight */
  int nkernel = connTable->size[0];
#pragma omp parallel for private(k)
  for (k = 0; k < nkernel; k++)
  {
    int o = (int)THTensor_(get2d)(connTable,k,1) - TH_INDEX_BASE;
    int i = (int)THTensor_(get2d)(connTable,k,0) - TH_INDEX_BASE;

    /* gradient to kernel */
    THTensor_(validXCorr2DRevptr)(
      gradWeight_data + k*weight_w*weight_h,
      scale,
      gradOutput_data + o*output_w*output_h, output_h, output_w,
      input_data + i*input_w*input_h, input_h, input_w,
      dH, dW
    );
  }

  /* clean up */
  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
}

#endif
