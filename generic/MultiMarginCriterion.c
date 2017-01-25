#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MultiMarginCriterion.c"
#else

// TODO: improve error messages
void THNN_(MultiMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *output,
          bool sizeAverage,
          int p,
          THTensor *weights,
          real margin)
{
  real *input_data, *weights_data;
  THIndex_t *target_data;
  long nframe, dim;
  long t, d;
  real sum;

  THArgCheck((input->nDimension == 1) || (input->nDimension == 2), 2,
	     "vector or matrix expected");

  if (input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0];
  }
  else
  {
    nframe = input->size[0];
    dim = input->size[1];
    THArgCheck((target->nDimension == 1) && (target->size[0] == nframe), 3,
	       "inconsistent target size");
  }

  for (t = 0; t < nframe; t++)
  {
    THIndex_t idx = THIndexTensor_(get1d)(target, t);
    THArgCheck((idx >= TH_INDEX_BASE) && (idx < dim + TH_INDEX_BASE), 3,
	       "target out of range");
  }

  input = THTensor_(newContiguous)(input);
  target = THIndexTensor_(newContiguous)(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;
  input_data = THTensor_(data)(input);
  target_data = THIndexTensor_(data)(target);
  weights_data = weights ? THTensor_(data)(weights) : NULL;

  sum = 0;
  for (t = 0; t < nframe; t++)
  {
    THIndex_t target_idx = target_data[t] - TH_INDEX_BASE;
    real input_target = input_data[target_idx];
    for (d = 0; d < dim; d++)
    {
      real z = margin - input_target + input_data[d];
      if (d == target_idx)
        continue;

      if (z > 0) {
        real h = (p==1) ? z : z*z;
        if(weights_data)
          h *= weights_data[target_idx];
        sum += h;
      }
    }
    input_data += dim;
  }

  sum /= dim;
  if(sizeAverage)
    sum /= nframe;

  THTensor_(set1d)(output, 0, sum);

  THTensor_(free)(input);
  THIndexTensor_(free)(target);
  if(weights)
    THTensor_(free)(weights);
}

void THNN_(MultiMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          int p,
          THTensor *weights,
          real margin)
{
  real *input_data;
  real *gradInput_data;
  THIndex_t *target_data;
  real *weights_data;
  long nframe, dim;
  long t, d;
  real g;

  THArgCheck((input->nDimension == 1) || (input->nDimension == 2), 2,
	     "vector or matrix expected");

  if (input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0];
  }
  else
  {
    nframe = input->size[0];
    dim = input->size[1];
    THArgCheck((target->nDimension == 1) && (target->size[0] == nframe), 3,
	       "inconsistent target size");
  }

  g = (sizeAverage ? 1./((real)(nframe*dim)) : 1./((real)dim));

  input = THTensor_(newContiguous)(input);
  target = THIndexTensor_(newContiguous)(target);
  input_data = THTensor_(data)(input);

  THTensor_(resizeAs)(gradInput, input);
  gradInput_data = THTensor_(data)(gradInput);

  target_data = THIndexTensor_(data)(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;
  weights_data = weights ? THTensor_(data)(weights) : NULL;

  for (t = 0; t < nframe; t++)
  {
    THIndex_t target_idx = target_data[t] - TH_INDEX_BASE;
    real input_target = input_data[target_idx];
    real gradInput_target = 0;
    for (d = 0; d < dim; d++)
    {
      real z = margin - input_target + input_data[d];
      if (d == target_idx)
        continue;

      if (z > 0)
      {
        real h = (p == 1) ? g : 2*g*z;
        if(weights_data)
          h *= weights_data[target_idx];
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
  THIndexTensor_(free)(target);
  if(weights)
    THTensor_(free)(weights);
}

#endif
