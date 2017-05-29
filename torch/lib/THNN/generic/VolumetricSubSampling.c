#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricSubSampling.c"
#else

static inline void THNN_(VolumetricSubSampling_shapeCheck)(
                         THTensor *input,
                         THTensor *gradOutput,
                         THTensor *weight,
                         int kW, int kH, int kT) {
  int ndims = input->nDimension;
  THNN_ARGCHECK(input->nDimension == 4 || input->nDimension == 5, 2, input,
                  "4D or 5D input tensor expected but got: %s");

  int nInputPlane = THTensor_(size)(weight, 0);

  int dimw = 3;
  int dimh = 2;
  int dimt = 1;

  long inputWidth;
  long inputHeight;
  long inputDepth;

  if (input->nDimension == 5) {
    dimw++;
    dimh++;
    dimt++;
  }

  inputWidth = input->size[dimw];
  inputHeight = input->size[dimh];
  inputDepth = input->size[dimt];

  THArgCheck(input->size[dimt-1] == nInputPlane, 2, "invalid number of input planes");
  THArgCheck(inputWidth >= kW && inputHeight >= kH
             && inputDepth >= kT, 2, "input image smaller than kernel size");
}

void THNN_(VolumetricSubSampling_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    THTensor *weight,
    THTensor *bias,
    int kW, int kH, int kT,
    int dW, int dH, int dT)
{

  real *weight_data = THTensor_(data)(weight);
  real *bias_data = THTensor_(data)(bias);
  real *output_data;
  real *input_data;

  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  long nbatch = 1;

  long inputWidth;
  long inputHeight;
  long inputDepth;
  long outputWidth;
  long outputHeight;
  long outputDepth;

  int nInputPlane = THTensor_(size)(weight,0);

  long k;

  THNN_(VolumetricSubSampling_shapeCheck)(input, NULL, weight, kW, kH, kT);

  if (input->nDimension == 5) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
    dimt++;
  }

  inputWidth = input->size[dimw];
  inputHeight = input->size[dimh];
  inputDepth = input->size[dimt];
  outputWidth = (inputWidth - kW) / dW + 1;
  outputHeight = (inputHeight - kH) / dH + 1;
  outputDepth = (inputDepth - kT) / dT + 1;

  if (input->nDimension == 4)
    THTensor_(resize4d)(output, nInputPlane, outputDepth, outputHeight, outputWidth);
  else
    THTensor_(resize5d)(output, input->size[0], nInputPlane, outputDepth, outputHeight, outputWidth);

  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);
  output_data = THTensor_(data)(output);

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane; k++)
  {
    long p;
    for(p = 0; p < nbatch; p++)
    {
      long xx, yy, zz;
      /* For all output pixels... */
      real *ptr_output = output_data + p*nInputPlane*outputWidth*outputHeight*outputDepth
                         + k*outputWidth*outputHeight*outputDepth;
      /* Get the good mask for (k,i) (k out, i in) */
      real the_weight = weight_data[k];
      /* Initialize to the bias */
      real z = bias_data[k];
      long i;
      for(i = 0; i < outputWidth*outputHeight*outputDepth; i++)
        ptr_output[i] = z;

      for(zz = 0; zz < outputDepth; zz++)
      {
        for(yy = 0; yy < outputHeight; yy++)
        {
          for(xx = 0; xx < outputWidth; xx++)
          {
            /* Compute the mean of the input image... */
            real *ptr_input = input_data + p*nInputPlane*inputWidth*inputHeight*inputDepth + k*inputWidth*inputHeight*inputDepth + zz*dT*inputHeight*inputWidth + yy*dH*inputWidth + xx*dW;
            real sum = 0;
            long kx, ky, kz;

            for(kz = 0; kz < kT; kz++)
            {
              for(ky = 0; ky < kH; ky++)
              {
                for(kx = 0; kx < kW; kx++)
                  sum += ptr_input[ky*inputWidth+kx];
              }
              ptr_input += inputHeight * inputWidth; /* next input frame */
            }
            /* Update output */
            *ptr_output++ += the_weight*sum;
          }
        }
      }
    }
  }
  THTensor_(free)(input);
}

void THNN_(VolumetricSubSampling_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    THTensor *weight,
    int kW, int kH, int kT,
    int dW, int dH, int dT)
{
  THNN_(VolumetricSubSampling_shapeCheck)(input, gradOutput, weight, kW, kH, kT);

  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  long nbatch = 1;

  long inputWidth;
  long inputHeight;
  long inputDepth;
  long outputWidth;
  long outputHeight;
  long outputDepth;

  int nInputPlane = THTensor_(size)(weight,0);

  real *weight_data;
  real *gradOutput_data;
  real *input_data, *gradInput_data;

  long k;

  if (input->nDimension == 5) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
    dimt++;
  }

  inputWidth = input->size[dimw];
  inputHeight = input->size[dimh];
  inputDepth = input->size[dimt];
  outputWidth = (inputWidth - kW) / dW + 1;
  outputHeight = (inputHeight - kH) / dH + 1;
  outputDepth = (inputDepth - kT) / dT + 1;

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
      real *ptr_gradOutput = gradOutput_data + p*nInputPlane*outputHeight*outputWidth*outputDepth + k*outputWidth*outputHeight*outputDepth;
      long xx, yy, zz;

      real* ptr_gi = gradInput_data + p*nInputPlane*inputWidth*inputHeight*inputDepth + k*inputWidth*inputHeight*inputDepth;
      long i;
      for(i=0; i<inputWidth*inputHeight*inputDepth; i++)
        ptr_gi[i] = 0.0;

      for(zz = 0; zz < outputDepth; zz++)
      {
        for(yy = 0; yy < outputHeight; yy++)
        {
          for(xx = 0; xx < outputWidth; xx++)
          {
            real *ptr_gradInput = gradInput_data + p*nInputPlane*inputWidth*inputHeight*inputDepth + k*inputWidth*inputHeight*inputDepth + zz*dT*inputHeight*inputWidth + yy*dH*inputWidth + xx*dW;
            real z = *ptr_gradOutput++ * the_weight;
            long kx, ky, kz;

            for(kz = 0; kz < kT; kz++)
            {
              for(ky = 0; ky < kH; ky++)
              {
                for(kx = 0; kx < kW; kx++)
                  ptr_gradInput[ky*inputWidth+kx] += z;
              }
              ptr_gradInput += inputHeight * inputWidth;
            }
          }
        }
      }
    }
  }
  THTensor_(free)(gradOutput);
}

void THNN_(VolumetricSubSampling_accGradParameters)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradWeight,
    THTensor *gradBias,
    int kW, int kH, int kT,
    int dW, int dH, int dT,
    accreal scale_)
{
  real scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  THNN_(VolumetricSubSampling_shapeCheck)(input, gradOutput, gradWeight, kW, kH, kT);

  long nbatch = 1;
  long dimw = 3;
  long dimh = 2;
  long dimt = 1;

  long inputWidth;
  long inputHeight;
  long inputDepth;
  long outputWidth;
  long outputHeight;
  long outputDepth;

  int nInputPlane = THTensor_(size)(gradWeight,0);

  real *gradWeight_data;
  real *gradBias_data;
  real *gradOutput_data;
  real *input_data;

  long k;

  if (input->nDimension == 5) {
    dimw++;
    dimh++;
    dimt++;
    nbatch = input->size[0];
  }

  inputWidth = input->size[dimw];
  inputHeight = input->size[dimh];
  inputDepth = input->size[dimt];
  outputWidth = (inputWidth - kW) / dW + 1;
  outputHeight = (inputHeight - kH) / dH + 1;
  outputDepth = (inputDepth - kT) / dT + 1;

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
      real *ptr_gradOutput = gradOutput_data + p*nInputPlane*outputHeight*outputWidth*outputDepth + k*outputWidth*outputHeight*outputDepth;
      real sum;
      long xx, yy, zz;
      long i;

      sum = 0;
      for(i = 0; i < outputWidth*outputHeight*outputDepth; i++)
        sum += ptr_gradOutput[i];
      gradBias_data[k] += scale*sum;

      sum = 0;
      for(zz = 0; zz < outputDepth; zz++)
      {
        for(yy = 0; yy < outputHeight; yy++)
        {
          for(xx = 0; xx < outputWidth; xx++)
          {
            real *ptr_input = input_data + p*nInputPlane*inputWidth*inputHeight*inputDepth + k*inputWidth*inputHeight*inputDepth + zz*dT*inputHeight*inputWidth + yy*dH*inputWidth + xx*dW;
            real z = *ptr_gradOutput++;
            long kx, ky, kz;

            for(kz = 0; kz < kT; kz++)
            {
              for(ky = 0; ky < kH; ky++)
              {
                for(kx = 0; kx < kW; kx++)
                  sum += z * ptr_input[ky*inputWidth+kx];
              }
              ptr_input += inputHeight * inputWidth;
            }
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
