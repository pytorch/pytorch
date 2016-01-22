#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MultiMarginCriterion.c"
#else

void THNN_(MultiMarginCriterion_updateOutput)(THNNState *state, THTensor *input, THTensor *target, THTensor* output, bool sizeAverage, int p)
{
  real *input_data, *target_data;
  long nframe, dim;
  long t, d;
  real sum;

  THArgCheck((input->nDimension == 1) || (input->nDimension == 2), 2, "vector or matrix expected");

  if (input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0]; 
  }
  else
  {
    nframe = input->size[0];
    dim = input->size[1];
    THArgCheck((target->nDimension == 1) && (target->size[0] == nframe), 3, "inconsistent target size");
  }

  for (t = 0; t < nframe; t++)
  {
    real idx = THTensor_(get1d)(target, t);
    THArgCheck((idx >= 1) && (idx <= dim), 3, "target out of range");
  }

  input = THTensor_(newContiguous)(input);
  target = THTensor_(newContiguous)(target);
  input_data = THTensor_(data)(input);
  target_data = THTensor_(data)(target);

  sum = 0;
  for (t = 0; t < nframe; t++)
  {
    long target_idx = (long)(target_data[t]-1);
    real input_target = input_data[target_idx];
    for (d = 0; d < dim; d++)
    {
      real z = 1 - input_target + input_data[d];
      if (d == target_idx)
        continue;
    
      if (z > 0)
        sum += (p == 1) ? z : z*z;
    }
    input_data += dim;
  }

  if (sizeAverage)
    sum /= dim;

  THTensor_(set1d)(output, 0, sum);

  THTensor_(free)(input);
  THTensor_(free)(target);
}

void THNN_(MultiMarginCriterion_updateGradInput)(THNNState *state, THTensor *input, THTensor *target, THTensor *gradInput, bool sizeAverage, int p)
{
  real *input_data;
  real *gradInput_data;
  real *target_data;
  long nframe, dim;
  long t, d;
  real g;

  THArgCheck((input->nDimension == 1) || (input->nDimension == 2), 2, "vector or matrix expected");

  if (input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0]; 
  }
  else
  {
    nframe = input->size[0];
    dim = input->size[1];
    THArgCheck((target->nDimension == 1) && (target->size[0] == nframe), 3, "inconsistent target size");
  }

  g = (sizeAverage ? 1./((real)dim) : 1.);

  input = THTensor_(newContiguous)(input);
  target = THTensor_(newContiguous)(target);
  input_data = THTensor_(data)(input);

  THTensor_(resizeAs)(gradInput, input);
  gradInput_data = THTensor_(data)(gradInput);

  target_data = THTensor_(data)(target);
    
  for (t = 0; t < nframe; t++)
  {
    long target_idx = (long)(target_data[t])-1;
    real input_target = input_data[target_idx];
    real gradInput_target = 0;
    for (d = 0; d < dim; d++)
    {
      real z = 1 - input_target + input_data[d];
      if (d == target_idx)
        continue;
    
      if (z > 0)
      {
        real h = (p == 1) ? g : 2*g*z;
        gradInput_target -= h;
        gradInput_data[d] = h;
      }
      else
        gradInput_data[d] = 0;
    }
    gradInput_data[target_idx] = gradInput_target;
    
    input_data += dim;
    gradInput_data += dim;
  }

  THTensor_(free)(input);  
  THTensor_(free)(target);
}

#endif
