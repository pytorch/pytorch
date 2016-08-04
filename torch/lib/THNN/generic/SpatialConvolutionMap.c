#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionMap.c"
#else

void THNN_(SpatialConvolutionMap_updateOutput)(
  THNNState *state, THTensor *input, THTensor *output, THTensor *weight, THTensor *bias,
  THTensor *connTable, int nInputPlane, int nOutputPlane,
  int dW, int dH)
{
  THArgCheck(
    weight != NULL && weight->nDimension == 3
    && connTable != NULL && connTable->size[0] == weight->size[0], 4,
    "3D weight tensor expected (connTable:size(%d) x kH x kW)", TH_INDEX_BASE
  );

  real *weight_data = THTensor_(data)(weight);
  real *bias_data = THTensor_(data)(bias);
  real *connTable_data = THTensor_(data)(connTable);

  int dimw = 2;
  int dimh = 1;
  int dimc = 0;
  long nbatch = 1;

  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");

  if (input->nDimension == 4)
  {
    nbatch = input->size[0];
    dimc++;
    dimw++;
    dimh++;
  }

  const long kH       = weight->size[1];
  const long kW       = weight->size[2];

  THArgCheck(input->size[dimc] >= nInputPlane, 2, "invalid number of input planes");
  THArgCheck(input->size[dimw] >= kW && input->size[dimh] >= kH, 2, "input image smaller than kernel size");

  const long input_w  = input->size[dimw];
  const long input_h  = input->size[dimh];
  const long output_w = (input_w - kW) / dW + 1;
  const long output_h = (input_h - kH) / dH + 1;

  if (input->nDimension == 3)
    THTensor_(resize3d)(output, nOutputPlane, output_h, output_w);
  else
    THTensor_(resize4d)(output, input->size[0], nOutputPlane, output_h, output_w);

  /* contiguous */
  input = THTensor_(newContiguous)(input);
  output = THTensor_(newContiguous)(output);

  /* get raw pointers */
  real *input_data = THTensor_(data)(input);
  real *output_data = THTensor_(data)(output);

  long p;
#pragma omp parallel for private(p)
  for (p = 0; p < nOutputPlane; p++)
  {
    long m;
    for (m = 0; m < nbatch; m++)
    {
      /* add bias */
      real *ptr_output = output_data + p*output_w*output_h + m*nOutputPlane*output_w*output_h;
      long j, k;
      real z= bias_data[p];
      for (j = 0; j < output_h*output_w; j++)
        ptr_output[j] = z;

      /* convolve all maps */
      int nweight = connTable->size[0];
      for (k = 0; k < nweight; k++)
      {
        /* get offsets for input/output */
        int o = (int)connTable_data[k*2+1] - TH_INDEX_BASE;
        int i = (int)connTable_data[k*2+0] - TH_INDEX_BASE;

        if (o == p)
        {
          THTensor_(validXCorr2Dptr)(
            output_data + o*output_w*output_h + m*nOutputPlane*output_w*output_h,
            1.0,
            input_data + i*input_w*input_h + m*nInputPlane*input_w*input_h, input_h, input_w,
            weight_data + k*kW*kH,
            kH, kW,
            dH, dW
          );
        }
      }
    }
  }

  /* clean up */
  THTensor_(free)(input);
  THTensor_(free)(output);
}

void THNN_(SpatialConvolutionMap_updateGradInput)(
  THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradInput, THTensor *weight, THTensor *bias,
  THTensor *connTable, int nInputPlane, int nOutputPlane,
  int dW, int dH)
{
  THArgCheck(
    weight != NULL && weight->nDimension == 3
    && connTable != NULL && connTable->size[0] == weight->size[0], 5,
    "3D weight tensor expected (connTable:size(%d) x kH x kW)", TH_INDEX_BASE
  );

  real *weight_data = THTensor_(data)(weight);
  real *connTable_data = THTensor_(data)(connTable);

  /* and dims */
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  if (input->nDimension == 4)
  {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  const long input_h  = input->size[dimh];
  const long input_w  = input->size[dimw];
  const long output_h = gradOutput->size[dimh];
  const long output_w = gradOutput->size[dimw];
  const long kH       = weight->size[1];
  const long kW       = weight->size[2];

  /* contiguous */
  gradInput = THTensor_(newContiguous)(gradInput);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* Resize/Zero */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  /* get raw pointers */
  real *gradInput_data = THTensor_(data)(gradInput);
  real *gradOutput_data = THTensor_(data)(gradOutput);

  long p;
#pragma omp parallel for private(p)
  for (p = 0; p < nInputPlane; p++)
  {
    long m;
    for (m = 0; m < nbatch; m++)
    {
      long k;
      /* backward all */
      int nkernel = connTable->size[0];
      for (k = 0; k < nkernel; k++)
      {
        int o = (int)connTable_data[k*2+1] - TH_INDEX_BASE;
        int i = (int)connTable_data[k*2+0] - TH_INDEX_BASE;
        if (i == p)
        {
          /* gradient to input */
          THTensor_(fullConv2Dptr)(
            gradInput_data + i*input_w*input_h + m*nInputPlane*input_w*input_h, 1.0,
            gradOutput_data + o*output_w*output_h + m*nOutputPlane*output_w*output_h,  output_h,  output_w,
            weight_data + k*kW*kH, kH, kW, dH, dW
          );
        }
      }
    }
  }

  /* clean up */
  THTensor_(free)(gradInput);
  THTensor_(free)(gradOutput);
}

void THNN_(SpatialConvolutionMap_accGradParameters)(
  THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradWeight, THTensor *gradBias,
  THTensor *connTable, int nInputPlane, int nOutputPlane,
  int dW, int dH, real scale)
{
  THArgCheck(
    gradWeight != NULL && gradWeight->nDimension == 3
    && connTable != NULL && connTable->size[0] == gradWeight->size[0], 5,
    "3D gradWeight tensor expected (connTable:size(%d) x kH x kW)", TH_INDEX_BASE
  );

  real *gradWeight_data = THTensor_(data)(gradWeight);
  real *gradBias_data = THTensor_(data)(gradBias);

  /* and dims */
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  if (input->nDimension == 4)
  {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  const long input_h  = input->size[dimh];
  const long input_w  = input->size[dimw];
  const long output_h = gradOutput->size[dimh];
  const long output_w = gradOutput->size[dimw];
  const long kH       = gradWeight->size[1];
  const long kW       = gradWeight->size[2];

  /* contiguous */
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* get raw pointers */
  real *input_data = THTensor_(data)(input);
  real *gradOutput_data = THTensor_(data)(gradOutput);

  long k;
  /* gradients wrt bias */
#pragma omp parallel for private(k)
  for (k = 0; k < nOutputPlane; k++)
  {
    long m;
    for (m = 0; m < nbatch; m++)
    {
      real *ptr_gradOutput = gradOutput_data + k*output_w*output_h + m*nOutputPlane*output_w*output_h;
      long l;
      for (l = 0; l < output_h*output_w; l++)
        gradBias_data[k] += scale*ptr_gradOutput[l];
    }
  }

  /* gradients wrt weight */
  const int nkernel = connTable->size[0];
#pragma omp parallel for private(k)
  for (k = 0; k < nkernel; k++)
  {
    long m;
    for (m = 0; m < nbatch; m++)
    {
      int o = (int)THTensor_(get2d)(connTable,k,1) - TH_INDEX_BASE;
      int i = (int)THTensor_(get2d)(connTable,k,0) - TH_INDEX_BASE;

      /* gradient to kernel */
      THTensor_(validXCorr2DRevptr)(
        gradWeight_data + k*kW*kH,
        scale,
        input_data + i*input_w*input_h + m*nInputPlane*input_w*input_h, input_h, input_w,
        gradOutput_data + o*output_w*output_h + m*nOutputPlane*output_w*output_h , output_h, output_w,
        dH, dW
      );
    }
  }

  /* clean up */
  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
}

#endif
