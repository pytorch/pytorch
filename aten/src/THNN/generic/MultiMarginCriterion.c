#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MultiMarginCriterion.c"
#else

// TODO: improve error messages
void THNN_(MultiMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *output,
          int64_t reduction,
          int p,
          THTensor *weights,
          accreal margin_)
{
  scalar_t margin = TH_CONVERT_ACCREAL_TO_REAL(margin_);
  scalar_t *input_data, *weights_data;
  THIndex_t *target_data;
  int64_t nframe, dim;
  int64_t t, d;
  scalar_t sum;

  AT_CHECK(!input->is_empty() && input->dim() <= 2,
           "non-empty vector or matrix expected, got size: ", input->sizes());

  if (input->dim() <= 1)
  {
    nframe = 1;
    dim = THTensor_sizeLegacyNoScalars(input, 0);
  }
  else
  {
    nframe = input->size(0);
    dim = input->size(1);
    AT_CHECK(!target->is_empty() && (THTensor_nDimensionLegacyNoScalars(target) == 1) && (THTensor_sizeLegacyNoScalars(target, 0) == nframe),
             "inconsistent target size, got: ", target->sizes());
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
  input_data = input->data<scalar_t>();
  target_data = THIndexTensor_(data)(target);
  weights_data = weights ? weights->data<scalar_t>() : NULL;

  if (reduction == Reduction::None)
  {
    THTensor_(resize1d)(output, nframe);

    for (t = 0; t < nframe; t++)
    {
      sum = 0;
      THIndex_t target_idx = target_data[t] - TH_INDEX_BASE;
      scalar_t input_target = input_data[target_idx];
      for (d = 0; d < dim; d++)
      {
        scalar_t z = margin - input_target + input_data[d];
        if (d == target_idx)
          continue;

        if (z > 0) {
          scalar_t h = (p==1) ? z : z*z;
          if(weights_data)
            h *= weights_data[target_idx];
          sum += h;
        }
      }

      sum /= dim;
      THTensor_(fastSet1d)(output, t, sum);
      input_data += dim;
    }
  }
  else
  {
    THTensor_(resize1d)(output, 1);

    sum = 0;
    for (t = 0; t < nframe; t++)
    {
      THIndex_t target_idx = target_data[t] - TH_INDEX_BASE;
      scalar_t input_target = input_data[target_idx];
      for (d = 0; d < dim; d++)
      {
        scalar_t z = margin - input_target + input_data[d];
        if (d == target_idx)
          continue;

        if (z > 0) {
          scalar_t h = (p==1) ? z : z*z;
          if(weights_data)
            h *= weights_data[target_idx];
          sum += h;
        }
      }
      input_data += dim;
    }

    sum /= dim;
    if(reduction == Reduction::Mean)
      sum /= nframe;

    THTensor_(set1d)(output, 0, sum);
  }

  c10::raw::intrusive_ptr::decref(input);
  THIndexTensor_(free)(target);
  if(weights)
    c10::raw::intrusive_ptr::decref(weights);
}

void THNN_(MultiMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          int64_t reduction,
          int p,
          THTensor *weights,
          accreal margin_)
{
  scalar_t margin = TH_CONVERT_ACCREAL_TO_REAL(margin_);
  scalar_t *input_data;
  scalar_t *gradInput_data;
  THIndex_t *target_data;
  scalar_t *weights_data;
  int64_t nframe, dim;
  int64_t t, d;
  scalar_t g;

  AT_CHECK(!input->is_empty() && (input->dim() <= 2),
           "non-empty vector or matrix expected, got size: ", input->sizes());

  if (input->dim() <= 1)
  {
    nframe = 1;
    dim = THTensor_sizeLegacyNoScalars(input, 0);
  }
  else
  {
    nframe = input->size(0);
    dim = input->size(1);
    AT_CHECK(!target->is_empty() && (target->dim() <= 1) && (THTensor_sizeLegacyNoScalars(target, 0) == nframe),
             "inconsistent target size, got: ", target->sizes());
  }

  g = (reduction == Reduction::Mean ? 1./((scalar_t)(nframe*dim)) : 1./((scalar_t)dim));

  input = THTensor_(newContiguous)(input);
  target = THIndexTensor_(newContiguous)(target);
  input_data = input->data<scalar_t>();

  THTensor_(resizeAs)(gradInput, input);
  THArgCheck(THTensor_(isContiguous)(gradInput), 5, "gradInput must be contiguous");
  gradInput_data = gradInput->data<scalar_t>();

  target_data = THIndexTensor_(data)(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;
  weights_data = weights ? weights->data<scalar_t>() : NULL;

  for (t = 0; t < nframe; t++)
  {
    THIndex_t target_idx = target_data[t] - TH_INDEX_BASE;
    scalar_t input_target = input_data[target_idx];
    scalar_t gradInput_target = 0;
    for (d = 0; d < dim; d++)
    {
      scalar_t z = margin - input_target + input_data[d];
      if (d == target_idx)
        continue;

      if (z > 0)
      {
        scalar_t h = (p == 1) ? g : 2*g*z;
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
  gradInput_data = gradInput->data<scalar_t>();

  if (reduction != Reduction::None)
  {
    THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);
    for (t = 0; t < nframe * dim; t++) {
      gradInput_data[t] *= THTensor_(fastGetLegacy1dNoScalars)(gradOutput, 0);
    }
  }
  else
  {
    THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, nframe);
    for (t = 0; t < nframe; t++)
    {
      for (d = 0; d < dim; d++)
      {
        gradInput_data[t * dim + d] *= THTensor_(fastGetLegacy1dNoScalars)(gradOutput, t);
      }
    }
  }

  c10::raw::intrusive_ptr::decref(input);
  THIndexTensor_(free)(target);
  if(weights)
    c10::raw::intrusive_ptr::decref(weights);
}

#endif
