#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialSubSampling.c"
#else

static inline void THNN_(SpatialSubSampling_shapeCheck)(
                         THTensor *input,
                         THTensor *gradOutput,
                         THTensor *weight,
                         int kW, int kH) {
  int ndims = input->nDimension;
  THNN_ARGCHECK(input->nDimension == 3 || input->nDimension == 4, 2, input,
                  "3D or 4D input tensor expected but got: %s");

  int nInputPlane = THTensor_(size)(weight, 0);

  int dimw = 2;
  int dimh = 1;

  long inputWidth;
  long inputHeight;

  if (input->nDimension == 4) {
    dimw++;
    dimh++;
  }

  inputWidth = input->size[dimw];
  inputHeight = input->size[dimh];

  THArgCheck(input->size[dimh-1] == nInputPlane, 2, "invalid number of input planes");
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
  
  real *weight_data = THTensor_(data)(weight);
  real *bias_data = THTensor_(data)(bias);
  real *output_data;
  real *input_data;

  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;

  long inputWidth;
  long inputHeight;
  long outputWidth;
  long outputHeight;

  int nInputPlane = THTensor_(size)(weight,0);

  long k;

  THNN_(SpatialSubSampling_shapeCheck)(input, NULL, weight, kW, kH);

  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  inputWidth = input->size[dimw];
  inputHeight = input->size[dimh];
  outputWidth = (inputWidth - kW) / dW + 1;
  outputHeight = (inputHeight - kH) / dH + 1;

  if (input->nDimension == 3)
    THTensor_(resize3d)(output, nInputPlane, outputHeight, outputWidth);
  else
    THTensor_(resize4d)(output, input->size[0], nInputPlane, outputHeight, outputWidth);
  
  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);
  output_data = THTensor_(data)(output);
  
#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane; k++)
  {
    long p;
    for(p = 0; p < nbatch; p++)
    {
      long xx, yy;
      /* For all output pixels... */
      real *ptr_output = output_data + p*nInputPlane*outputWidth*outputHeight + k*outputWidth*outputHeight;
      /* Get the good mask for (k,i) (k out, i in) */
      real the_weight = weight_data[k];
      /* Initialize to the bias */
      real z = bias_data[k];
      long i;
      for(i = 0; i < outputWidth*outputHeight; i++)
        ptr_output[i] = z;
      
      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          /* Compute the mean of the input image... */
          real *ptr_input = input_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight + yy*dH*inputWidth+xx*dW;
          real sum = 0;
          long kx, ky;

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
  THTensor_(free)(input);
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
  long nbatch = 1;

  long inputWidth;
  long inputHeight;
  long outputWidth;
  long outputHeight;

  int nInputPlane = THTensor_(size)(weight,0);

  real *weight_data;
  real *gradOutput_data;
  real *input_data, *gradInput_data;

  long k;

  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  inputWidth = input->size[dimw];
  inputHeight = input->size[dimh];
  outputWidth = (inputWidth - kW) / dW + 1;
  outputHeight = (inputHeight - kH) / dH + 1;

  weight_data = THTensor_(data)(weight);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  gradOutput_data = THTensor_(data)(gradOutput);

  input_data = THTensor_(data)(input);

  THTensor_(resizeAs)(gradInput, input);
  gradInput_data = THTensor_(data)(gradInput);

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane; k++)
  {
    long p;
    for(p = 0; p < nbatch; p++)
    {
      real the_weight = weight_data[k];
      real *ptr_gradOutput = gradOutput_data + p*nInputPlane*outputHeight*outputWidth + k*outputWidth*outputHeight;
      long xx, yy;

      real* ptr_gi = gradInput_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;
      long i;
      for(i=0; i<inputWidth*inputHeight; i++)
        ptr_gi[i] = 0.0;

      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          real *ptr_gradInput = gradInput_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight + yy*dH*inputWidth+xx*dW;
          real z = *ptr_gradOutput++ * the_weight;
          long kx, ky;

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
  THTensor_(free)(gradOutput);
}

void THNN_(SpatialSubSampling_accGradParameters)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradWeight,
    THTensor *gradBias,
    int kW, int kH,
    int dW, int dH,
    real scale)
{
  THNN_(SpatialSubSampling_shapeCheck)(input, gradOutput, gradWeight, kW, kH);

  long nbatch = 1;
  long dimw = 2;
  long dimh = 1;

  long inputWidth;
  long inputHeight;
  long outputWidth;
  long outputHeight;

  int nInputPlane = THTensor_(size)(gradWeight,0);

  real *gradWeight_data;
  real *gradBias_data;
  real *gradOutput_data;
  real *input_data;

  long k;

  if (input->nDimension == 4) {
    dimw++;
    dimh++;
    nbatch = input->size[0];
  }

  inputWidth = input->size[dimw];
  inputHeight = input->size[dimh];
  outputWidth = (inputWidth - kW) / dW + 1;
  outputHeight = (inputHeight - kH) / dH + 1;

  gradWeight_data = THTensor_(data)(gradWeight);
  gradBias_data = THTensor_(data)(gradBias);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  gradOutput_data = THTensor_(data)(gradOutput);

  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane; k++)
  {
    long p;
    for(p = 0; p < nbatch; p++)
    {
      real *ptr_gradOutput = gradOutput_data + p*nInputPlane*outputHeight*outputWidth + k*outputWidth*outputHeight;
      real sum;
      long xx, yy;
      long i;

      sum = 0;
      for(i = 0; i < outputWidth*outputHeight; i++)
        sum += ptr_gradOutput[i];
      gradBias_data[k] += scale*sum;

      sum = 0;
      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          real *ptr_input = input_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight + yy*dH*inputWidth+xx*dW;
          real z = *ptr_gradOutput++;
          long kx, ky;

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

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
}

#endif
