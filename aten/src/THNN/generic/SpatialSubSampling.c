#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialSubSampling.c"
#else

static inline void THNN_(SpatialSubSampling_shapeCheck)(
                         THTensor *input,
                         THTensor *gradOutput,
                         THTensor *weight,
                         int kW, int kH) {
  THNN_ARGCHECK(!input->is_empty() && (input->dim() == 3 || input->dim() == 4), 2, input,
                  "3D or 4D input tensor expected but got: %s");
  THArgCheck(THTensor_(isContiguous)(weight), 4, "weight must be contiguous");

  int nInputPlane = THTensor_(size)(weight, 0);

  int dimw = 2;
  int dimh = 1;

  int64_t inputWidth;
  int64_t inputHeight;

  if (input->dim() == 4) {
    dimw++;
    dimh++;
  }

  inputWidth = input->size(dimw);
  inputHeight = input->size(dimh);

  THArgCheck(input->size(dimh-1) == nInputPlane, 2, "invalid number of input planes");
  THArgCheck(inputWidth >= kW && inputHeight >= kH, 2, "input image smaller than kernel size");
}

void THNN_(SpatialSubSampling_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    THTensor *weight,
    THTensor *bias,
    int kW, int kH,
    int dW, int dH)
{
  THArgCheck(!bias || THTensor_(isContiguous)(bias), 5, "bias must be contiguous");

  scalar_t *weight_data = weight->data<scalar_t>();
  scalar_t *bias_data = bias->data<scalar_t>();
  scalar_t *output_data;
  scalar_t *input_data;

  int dimw = 2;
  int dimh = 1;
  int64_t nbatch = 1;

  int64_t inputWidth;
  int64_t inputHeight;
  int64_t outputWidth;
  int64_t outputHeight;

  int nInputPlane = THTensor_(size)(weight,0);

  int64_t k;

  THNN_(SpatialSubSampling_shapeCheck)(input, NULL, weight, kW, kH);

  if (input->dim() == 4) {
    nbatch = input->size(0);
    dimw++;
    dimh++;
  }

  inputWidth = input->size(dimw);
  inputHeight = input->size(dimh);
  outputWidth = (inputWidth - kW) / dW + 1;
  outputHeight = (inputHeight - kH) / dH + 1;

  if (input->dim() == 3)
    THTensor_(resize3d)(output, nInputPlane, outputHeight, outputWidth);
  else
    THTensor_(resize4d)(output, input->size(0), nInputPlane, outputHeight, outputWidth);

  input = THTensor_(newContiguous)(input);
  input_data = input->data<scalar_t>();
  output_data = output->data<scalar_t>();

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane; k++)
  {
    int64_t p;
    for(p = 0; p < nbatch; p++)
    {
      int64_t xx, yy;
      /* For all output pixels... */
      scalar_t *ptr_output = output_data + p*nInputPlane*outputWidth*outputHeight + k*outputWidth*outputHeight;
      /* Get the good mask for (k,i) (k out, i in) */
      scalar_t the_weight = weight_data[k];
      /* Initialize to the bias */
      scalar_t z = bias_data[k];
      int64_t i;
      for(i = 0; i < outputWidth*outputHeight; i++)
        ptr_output[i] = z;

      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          /* Compute the mean of the input image... */
          scalar_t *ptr_input = input_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight + yy*dH*inputWidth+xx*dW;
          scalar_t sum = 0;
          int64_t kx, ky;

          for(ky = 0; ky < kH; ky++)
          {
            for(kx = 0; kx < kW; kx++)
              sum += ptr_input[kx];
            ptr_input += inputWidth; /* next input line */
          }
          /* Update output */
          *ptr_output++ += the_weight*sum;
        }
      }
    }
  }
  c10::raw::intrusive_ptr::decref(input);
}

void THNN_(SpatialSubSampling_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    THTensor *weight,
    int kW, int kH,
    int dW, int dH)
{
  THNN_(SpatialSubSampling_shapeCheck)(input, gradOutput, weight, kW, kH);

  int dimw = 2;
  int dimh = 1;
  int64_t nbatch = 1;

  int64_t inputWidth;
  int64_t inputHeight;
  int64_t outputWidth;
  int64_t outputHeight;

  int nInputPlane = THTensor_(size)(weight,0);

  scalar_t *weight_data;
  scalar_t *gradOutput_data;
  scalar_t *gradInput_data;

  int64_t k;

  if (input->dim() == 4) {
    nbatch = input->size(0);
    dimw++;
    dimh++;
  }

  inputWidth = input->size(dimw);
  inputHeight = input->size(dimh);
  outputWidth = (inputWidth - kW) / dW + 1;
  outputHeight = (inputHeight - kH) / dH + 1;

  weight_data = weight->data<scalar_t>();
  gradOutput = THTensor_(newContiguous)(gradOutput);
  gradOutput_data = gradOutput->data<scalar_t>();

  THTensor_(resizeAs)(gradInput, input);
  gradInput_data = gradInput->data<scalar_t>();

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane; k++)
  {
    int64_t p;
    for(p = 0; p < nbatch; p++)
    {
      scalar_t the_weight = weight_data[k];
      scalar_t *ptr_gradOutput = gradOutput_data + p*nInputPlane*outputHeight*outputWidth + k*outputWidth*outputHeight;
      int64_t xx, yy;

      scalar_t* ptr_gi = gradInput_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;
      int64_t i;
      for(i=0; i<inputWidth*inputHeight; i++)
        ptr_gi[i] = 0.0;

      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          scalar_t *ptr_gradInput = gradInput_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight + yy*dH*inputWidth+xx*dW;
          scalar_t z = *ptr_gradOutput++ * the_weight;
          int64_t kx, ky;

          for(ky = 0; ky < kH; ky++)
          {
            for(kx = 0; kx < kW; kx++)
              ptr_gradInput[kx] += z;
            ptr_gradInput += inputWidth;
          }
        }
      }
    }
  }
  c10::raw::intrusive_ptr::decref(gradOutput);
}

void THNN_(SpatialSubSampling_accGradParameters)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradWeight,
    THTensor *gradBias,
    int kW, int kH,
    int dW, int dH,
    accreal scale_)
{
  scalar_t scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  THNN_(SpatialSubSampling_shapeCheck)(input, gradOutput, gradWeight, kW, kH);

  int64_t nbatch = 1;
  int64_t dimw = 2;
  int64_t dimh = 1;

  int64_t inputWidth;
  int64_t inputHeight;
  int64_t outputWidth;
  int64_t outputHeight;

  int nInputPlane = THTensor_(size)(gradWeight,0);

  scalar_t *gradWeight_data;
  scalar_t *gradBias_data;
  scalar_t *gradOutput_data;
  scalar_t *input_data;

  int64_t k;

  if (input->dim() == 4) {
    dimw++;
    dimh++;
    nbatch = input->size(0);
  }

  inputWidth = input->size(dimw);
  inputHeight = input->size(dimh);
  outputWidth = (inputWidth - kW) / dW + 1;
  outputHeight = (inputHeight - kH) / dH + 1;

  gradWeight_data = gradWeight->data<scalar_t>();
  gradBias_data = gradBias->data<scalar_t>();
  gradOutput = THTensor_(newContiguous)(gradOutput);
  gradOutput_data = gradOutput->data<scalar_t>();

  input = THTensor_(newContiguous)(input);
  input_data = input->data<scalar_t>();

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane; k++)
  {
    int64_t p;
    for(p = 0; p < nbatch; p++)
    {
      scalar_t *ptr_gradOutput = gradOutput_data + p*nInputPlane*outputHeight*outputWidth + k*outputWidth*outputHeight;
      scalar_t sum;
      int64_t xx, yy;
      int64_t i;

      sum = 0;
      for(i = 0; i < outputWidth*outputHeight; i++)
        sum += ptr_gradOutput[i];
      gradBias_data[k] += scale*sum;

      sum = 0;
      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          scalar_t *ptr_input = input_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight + yy*dH*inputWidth+xx*dW;
          scalar_t z = *ptr_gradOutput++;
          int64_t kx, ky;

          for(ky = 0; ky < kH; ky++)
          {
            for(kx = 0; kx < kW; kx++)
              sum += z * ptr_input[kx];
            ptr_input += inputWidth;
          }
        }
      }
      gradWeight_data[k] += scale*sum;
    }
  }

  c10::raw::intrusive_ptr::decref(input);
  c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif
