#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MultiLabelMarginCriterion.c"
#else

// TODO: improve error messages
void THNN_(MultiLabelMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *output,
          THTensor *isTarget,
          int64_t reduction)
{
  scalar_t *input_data, *isTarget_data;
  THIndex_t *target_data;
  int64_t nframe, dim;
  int64_t t, d, dt, ddt;
  scalar_t sum;

  AT_CHECK(!input->is_empty() && input->dim() <= 2,
           "non-empty vector or matrix expected, got size: ", input->sizes());

  if (input->dim() <= 1)
  {
    nframe = 1;
    dim = THTensor_sizeLegacyNoScalars(input, 0);
    AT_CHECK(!target->is_empty() && (target->dim() <= 1) && (THTensor_sizeLegacyNoScalars(target, 0) == dim),
             "inconsistent target size");
  }
  else
  {
    nframe = input->size(0);
    dim = input->size(1);
    AT_CHECK(!target->is_empty() && target->dim() == 2 && (target->size(0) == nframe)
             && (target->size(1) == dim), "inconsistent target size");
  }

  THArgCheck(THIndexTensor_(minall)(target) >= -1+TH_INDEX_BASE, 3, "target out of range");
  THArgCheck(THIndexTensor_(maxall)(target) < dim+TH_INDEX_BASE, 3, "target out of range");

  target = THIndexTensor_(newContiguous)(target);
  input = THTensor_(newContiguous)(input);
  input_data = input->data<scalar_t>();
  target_data = THIndexTensor_(data)(target);

  if (!isTarget->sizes().equals(target->sizes())) {
    THTensor_(resizeNd)(isTarget, target->dim(), THTensor_getSizePtr(target), nullptr);
  }
  THTensor_(zero)(isTarget);
  isTarget_data = isTarget->data<scalar_t>();

  if (reduction != Reduction::None)
  {
    THTensor_(resize1d)(output, 1);

    sum = 0;
    for (t = 0; t < nframe; t++)
    {
      for (ddt = 0; ddt < dim; ddt++)
      {
        THIndex_t target_idx = target_data[ddt] - TH_INDEX_BASE;
        if (target_idx < 0)
          break;
        isTarget_data[target_idx] = 1;
      }
      for (dt = 0; dt < dim; dt++)
      {
        THIndex_t target_idx = target_data[dt] - TH_INDEX_BASE;
        scalar_t input_target;
        if (target_idx < 0)
          break;

        input_target = input_data[target_idx];
        for (d = 0; d < dim; d++)
        {
          if (!isTarget_data[d])
          {
            scalar_t z = 1 - input_target + input_data[d];
            if (z > 0)
              sum += z;
          }
        }
      }
      input_data += dim;
      target_data += dim;
      isTarget_data += dim;
    }

    sum /= dim;
    if (reduction == Reduction::Mean)
      sum /= nframe;
    THTensor_(fastSet1d)(output, 0, sum);

    c10::raw::intrusive_ptr::decref(input);
    THIndexTensor_(free)(target);
    return;
  }

  THTensor_(resize1d)(output, nframe);

  for (t = 0; t < nframe; t++)
  {
    for (ddt = 0; ddt < dim; ddt++)
    {
      THIndex_t target_idx = target_data[ddt] - TH_INDEX_BASE;
      if (target_idx < 0)
        break;
      isTarget_data[target_idx] = 1;
    }

    sum = 0;
    for (dt = 0; dt < dim; dt++)
    {
      THIndex_t target_idx = target_data[dt] - TH_INDEX_BASE;
      scalar_t input_target;
      if (target_idx < 0)
        break;

      input_target = input_data[target_idx];
      for (d = 0; d < dim; d++)
      {
        if (!isTarget_data[d])
        {
          scalar_t z = 1 - input_target + input_data[d];
          if (z > 0)
            sum += z;
        }
      }
    }

    sum /= dim;
    THTensor_(fastSet1d)(output, t, sum);

    input_data += dim;
    target_data += dim;
    isTarget_data += dim;
  }

  c10::raw::intrusive_ptr::decref(input);
  THIndexTensor_(free)(target);
}

void THNN_(MultiLabelMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *isTarget,
          int64_t reduction)
{
  scalar_t *input_data;
  scalar_t *gradInput_data;
  THIndex_t *target_data;
  scalar_t *isTarget_data;
  int64_t nframe, dim;
  int64_t t, d, dt;
  scalar_t g;

  AT_CHECK(!input->is_empty() && input->dim() <= 2,
           "vector or matrix expected, got size: ", input->sizes());

  if (input->dim() <= 1)
  {
    nframe = 1;
    dim = THTensor_sizeLegacyNoScalars(input, 0);
    AT_CHECK((!target->is_empty() && target->dim() <= 1) && (THTensor_sizeLegacyNoScalars(target, 0) == dim),
             "inconsistent target size");
    AT_CHECK((!isTarget->is_empty() && isTarget->dim() <= 1) && (THTensor_sizeLegacyNoScalars(isTarget, 0) == dim),
             "inconsistent isTarget size");
  }
  else
  {
    nframe = input->size(0);
    dim = input->size(1);
    AT_CHECK(!target->is_empty() && (target->dim() == 2) && (target->size(0) == nframe)
             && (target->size(1) == dim), 3, "inconsistent target size");
    AT_CHECK(!isTarget->is_empty() && (isTarget->dim() == 2) && (isTarget->size(0) == nframe)
             && (isTarget->size(1) == dim), 3, "inconsistent isTarget size");
  }

  THArgCheck(THIndexTensor_(minall)(target) >= -1+TH_INDEX_BASE, 3, "target out of range");
  THArgCheck(THIndexTensor_(maxall)(target) < dim+TH_INDEX_BASE, 3, "target out of range");

  THArgCheck(THTensor_(minall)(isTarget) >= 0, 3, "isTarget out of range");
  THArgCheck(THTensor_(maxall)(isTarget) <= 1, 3, "isTarget out of range");

  target = THIndexTensor_(newContiguous)(target);
  input = THTensor_(newContiguous)(input);
  isTarget = THTensor_(newContiguous)(isTarget);
  input_data = input->data<scalar_t>();
  target_data = THIndexTensor_(data)(target);
  isTarget_data = isTarget->data<scalar_t>();

  THTensor_(resizeAs)(gradInput, input);
  gradInput = THTensor_(newContiguous)(gradInput);
  THTensor_(zero)(gradInput);
  gradInput_data = gradInput->data<scalar_t>();

  g = reduction == Reduction::Mean ? (1./((scalar_t)(nframe*dim))) : (1./((scalar_t)dim));

  for (t = 0; t < nframe; t++)
  {
    for (dt = 0; dt < dim; dt++)
    {
      THIndex_t target_idx = target_data[dt] - TH_INDEX_BASE;
      scalar_t input_target;
      if (target_idx < 0)
        break;

      input_target = input_data[target_idx];
      for (d = 0; d < dim; d++)
      {
        if (!isTarget_data[d])
        {
          scalar_t z = 1 - input_target + input_data[d];
          if (z > 0)
          {
            gradInput_data[target_idx] -= g;
            gradInput_data[d] += g;
          }
        }
      }
    }
    input_data += dim;
    target_data += dim;
    isTarget_data += dim;
    gradInput_data += dim;
  }
  gradInput_data = gradInput->data<scalar_t>();

  if (reduction != Reduction::None)
  {
    THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);
    for (t = 0; t < nframe*dim; t++)
    {
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
  c10::raw::intrusive_ptr::decref(isTarget);
  c10::raw::intrusive_ptr::decref(gradInput);
}

#endif
