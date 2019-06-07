#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/IndexLinear.c"
#else

#include <ATen/Parallel.h>

/* Threshold used to trigger multithreading */
#ifndef THNN_SPARSE_OMP_THRESHOLD
#define THNN_SPARSE_OMP_THRESHOLD 100000
#endif

/* Threshold used to trigger BLAS axpy call */
#ifndef THNN_SPARSE_OUTDIM_THRESHOLD
#define THNN_SPARSE_OUTDIM_THRESHOLD 49
#endif

/* sign MACRO */
#ifndef THNN_INDEXLINEAR_SIGN
#define THNN_INDEXLINEAR_SIGN(a) ( ( (a) < 0 )  ?  -1   : ( (a) > 0 ) )
#endif

static bool THNN_(checkKeysValues)(THLongTensor* keys, THTensor* values)
{
  return THLongTensor_size(keys, 0) == THTensor_(nElement)(values)
                && THTensor_(nDimensionLegacyAll)(values) == 1
                && THLongTensor_nDimensionLegacyAll(keys) == 1;
}

void THNN_(IndexLinear_updateOutput)(
          THNNState *state,
          THLongTensor *keys,
          int64_t keysOffset,
          THTensor *values,
          THLongTensor *sizes,
          THLongTensor *cumSumSizes,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *normalizedValues,
          int  train)
{
  /* Retrieve all the dimensions of the problem */
  int64_t batchSize = THLongTensor_size(sizes, 0);
  int64_t keysSize = THLongTensor_size(keys, 0);
  int64_t outDim = THTensor_(size)(bias, 0);
  int64_t woutDim = THTensor_(size)(weight, 1);
  int maxNormalize = woutDim - outDim;
  int64_t* sizesData = THLongTensor_data(sizes);
  int64_t* cumSumSizesData = THLongTensor_data(cumSumSizes);

  /* Define/resize the normalized values tensor if maxNormalize is  > 0 */
  scalar_t* normalizedValuesData = NULL;
  if (maxNormalize)
  {
    THTensor_(resize1d)(normalizedValues, keysSize);
    normalizedValuesData = normalizedValues->data<scalar_t>();
  }

  /* Resize the output */
  THTensor_(resize2d)(output, batchSize, outDim);

  /* Access the storage data/strides */
  scalar_t* outputData = output->data<scalar_t>();
  scalar_t* valuesData = values->data<scalar_t>();
  scalar_t* weightData = weight->data<scalar_t>();
  int64_t weightStride0 = weight->stride(0);
  scalar_t* biasData = bias->data<scalar_t>();
  int64_t* keysData = THLongTensor_data(keys);

  /* Make sure these inputs are contiguous to accelerate computations */
  THArgCheck(THLongTensor_isContiguous(keys), 1, "keys vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(values), 3, "values vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(output), 6, "output vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(weight), 7, "weight matrix must be contiguous");
  THArgCheck(THTensor_(isContiguous)(bias), 8, "bias vector must be contiguous");
  THArgCheck(THNN_(checkKeysValues)(keys, values), 1, "Keys and values should have the same number of elements");
  THArgCheck(THTensor_(isContiguous)(normalizedValues), 9, "normalizedValues vector must be contiguous");

  /* Separate cases: output dimension is == 1, or > 1
   * This allows for some optimizations. */
  if (outDim == 1)
  {
    THVector_(fill)(outputData, *biasData, batchSize);
    if (maxNormalize)
    {
      /* Parallelize on the batch itself */
      auto loop = [&](int64_t start, int64_t end) {
        for (auto j = start; j < end; j++)
        {
          scalar_t* loutputData = outputData + j;
          scalar_t val = 0;
          scalar_t absVal = 0;
          int64_t offset = j == 0 ? 0 : cumSumSizesData[j - 1];

          for (auto i = 0; i < sizesData[j]; i++)
          {
            int64_t woffset = weightStride0*(keysData[offset] + keysOffset);
            absVal = fabs(valuesData[offset]);
            if (train)
            {
              if (absVal > weightData[woffset])
              {
                weightData[woffset] = absVal;
                weightData[woffset+1] = 1/absVal;
              }

              /*
               * The following can be used to scale the size of the updates
               * depending on some rule, e.g. the frequency of a feature, ...
               * This is used at update time.
               * TODO: implement a smarter update scale.
               */
              weightData[woffset+2] = 1;
            }
            normalizedValuesData[offset] = (absVal > weightData[woffset] ? THNN_INDEXLINEAR_SIGN(valuesData[offset]):valuesData[offset]*weightData[woffset+1]) + weightData[woffset+3];
            val += normalizedValuesData[offset] * weightData[woffset+maxNormalize];
            offset++;
          }
          *loutputData += val;
        }
      };
      if (keysSize * outDim > THNN_SPARSE_OMP_THRESHOLD) {
        at::parallel_for(0, batchSize, 1, loop);
      } else {
        loop(0, batchSize);
      }
    }
    else
    {
      /* Parallelize on the batch itself */
      auto loop = [&](int64_t start, int64_t end) {
        for (auto j = start; j < end; j++)
        {
          int64_t offset = j == 0 ? 0 : cumSumSizesData[j - 1];
          scalar_t* loutputData = outputData + j;
          scalar_t val = 0;

          for (auto i = 0; i < sizesData[j]; i++)
          {
            val += weightData[weightStride0*(keysData[offset] + keysOffset)] * valuesData[offset];
            offset++;
          }
          *loutputData += val;
        }
      };
      if (keysSize * outDim > THNN_SPARSE_OMP_THRESHOLD) {
        at::parallel_for(0, batchSize, 1, loop);
      } else {
        loop(0, batchSize);
      }
    }
  }
  else {
    auto loop = [&](int64_t start, int64_t end) {
      for (auto j = start; j < end; j++)
      {
        int64_t offset = j == 0 ? 0 : cumSumSizesData[j -  1];
        scalar_t val;
        scalar_t* loutputData = outputData + j*outDim;
        scalar_t* lweightData = weightData;
        memcpy(loutputData, biasData, outDim*sizeof(scalar_t));
        for (auto i = 0; i < sizesData[j]; i++)
        {
          int64_t woffset = weightStride0*(keysData[offset] + keysOffset);
          if (maxNormalize)
          {
            val = valuesData[offset];
            scalar_t absVal = fabs(val);
            if (train)
            {
              if (absVal > weightData[woffset])
              {
                weightData[woffset] = absVal;
                weightData[woffset+1] = 1/absVal;
              }

              /*
               * The following can be used to scale the size of the updates
               * depending on some rule, e.g. the frequency of a feature, ...
               * The commented section thereafter is just an example of what can be done:
               *
               *```
               * weightData[woffset+2] = weightData[woffset+2]==0?1:(weightData[woffset+2] / (weightData[woffset+2] + 1));
               * scalar_t alpha = 1;
               * scalar_t beta = 0.01;
               * scalar_t gamma = 1 - 0.000001;
               * scalar_t l = weightData[woffset+2]==0?1/gamma:(weightData[woffset+2] - beta) / (alpha - beta);
               * l = gamma*l;
               * weightData[woffset+2] = (alpha-beta)*l + beta;
               * ```
               *
               * TODO: implement a smarter update scale.
               */
              weightData[woffset+2] = 1;
            }

            /* Normalize + Clamp */
            val = (absVal > weightData[woffset] ? THNN_INDEXLINEAR_SIGN(val):val*weightData[woffset+1]) + weightData[woffset+3];
            normalizedValuesData[offset] = val;

            lweightData = weightData + woffset + maxNormalize;
          }
          else
          {
            val = valuesData[offset];
            lweightData = weightData + woffset;
          }
          if (outDim > THNN_SPARSE_OUTDIM_THRESHOLD)
          {
            THBlas_(axpy)(outDim, val, lweightData, 1, loutputData, 1);
          }
          else
          {
            for (auto k = 0; k < outDim; k++)
            {
              loutputData[k] += lweightData[k] * val;
            }
          }
          offset++;
        }
      }
    };
    if (keysSize * outDim > THNN_SPARSE_OMP_THRESHOLD) {
      at::parallel_for(0, batchSize, 1, loop);
    } else {
      loop(0, batchSize);
    }

  }
  return;
}

void THNN_(IndexLinear_updateParameters)(
          THNNState *state,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          THLongTensor *runningKeys,
          THLongTensor *cumSumSizes,
          int64_t keysOffset,
          accreal weightDecay_,
          accreal learningRate_)
{
  scalar_t weightDecay = TH_CONVERT_ACCREAL_TO_REAL(weightDecay_);
  scalar_t learningRate = TH_CONVERT_ACCREAL_TO_REAL(learningRate_);
  /* Retrieve all the dimensions of the problem */
  int64_t outDim = THTensor_(size)(bias, 0);
  int64_t woutDim = THTensor_(size)(weight, 1);
  int maxNormalize = woutDim - outDim;
  int64_t keysSize = THLongTensor_size(runningKeys, 0);

  /* Access the storage data/strides */
  scalar_t* gradWeightData = gradWeight->data<scalar_t>();
  scalar_t* weightData = weight->data<scalar_t>();
  int64_t weightStride0 = weight->stride(0);
  scalar_t* gradBiasData = gradBias->data<scalar_t>();
  scalar_t* biasData = bias->data<scalar_t>();
  int64_t* keysData = THLongTensor_data(runningKeys);

  /* Make sure these inputs are contiguous to accelerate computations */
  THArgCheck(THTensor_(isContiguous)(gradWeight), 1, "gradWeight must be contiguous");
  THArgCheck(THTensor_(isContiguous)(gradBias), 2, "gradBias vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(weight), 3, "gradBias vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(bias), 4, "gradBias vector must be contiguous");
  THArgCheck(THLongTensor_isContiguous(runningKeys), 5, "keys vector must be contiguous");

  int j, k;

  /* Update the bias first */
  THVector_(cadd)(biasData, biasData, gradBiasData, -learningRate, outDim);

  /* Separate cases: output dimension is == 1, or > 1
   * This allows for some optimizations.
   * No multithreading here as this could
   * corrupt the results (hogwild style) */
  if (outDim == 1)
  {
    if (maxNormalize)
    {
      if (weightDecay)
      {
        for (j = 0; j < keysSize; j++)
        {
          int64_t woffset = weightStride0*(keysData[j] + keysOffset) + maxNormalize;
          scalar_t lr = learningRate*weightData[woffset-2];
          weightData[woffset-1] -= weightData[woffset]*gradWeightData[2*j]*lr;
          weightData[woffset] -= gradWeightData[2*j+1]*lr - weightDecay * weightData[woffset-2] * weightData[woffset];
        }
      }
      else
      {
        for (j = 0; j < keysSize; j++)
        {
          int64_t woffset = weightStride0*(keysData[j] + keysOffset) + maxNormalize;
          scalar_t lr = learningRate*weightData[woffset-2];
          weightData[woffset-1] -= weightData[woffset]*gradWeightData[2*j]*lr;
          weightData[woffset] -= gradWeightData[2*j+1]*lr;
        }
      }
    }
    else
    {
      if (weightDecay)
      {
        for (j = 0; j < keysSize; j++)
        {
          int64_t woffset = weightStride0*(keysData[j] + keysOffset);
          weightData[woffset] -= gradWeightData[j]*learningRate + weightDecay * weightData[woffset];
        }
      }
      else
      {
        for (j = 0; j < keysSize; j++)
        {
          weightData[weightStride0*(keysData[j] + keysOffset)] -= gradWeightData[j]*learningRate;
        }
      }
    }
  }
  else
  {
    for (j = 0; j < keysSize; j++)
    {
      scalar_t lr = learningRate;
      scalar_t wd = weightDecay;
      scalar_t* lweightData;
      int64_t woffset = weightStride0*(keysData[j] + keysOffset);
      scalar_t* lgradWeightData = gradWeightData + j*outDim;
      if (maxNormalize)
      {
        lgradWeightData += j*outDim;
        /* weightData[woffset + 2] */
        lweightData = weightData + woffset + maxNormalize - 2;
        lr = lr*lweightData[0];
        wd = weightDecay*lweightData[0];
        /* weightData[woffset + 3] */
        lweightData++;
        for (k=0; k < outDim; k++)
        {
            lweightData[0] -= lgradWeightData[k]*lweightData[k+1]*lr;
        }
        lweightData++;
        lgradWeightData += outDim;
      }
      else
      {
        lweightData = weightData + woffset;
      }

      /* We do sparse weight decay.
       * We think it makes more sense. */
      if (weightDecay)
      {
        for (k=0; k < outDim; k++)
        {
            lweightData[k] -= lweightData[k]*wd;
        }
      }

      if (outDim > THNN_SPARSE_OUTDIM_THRESHOLD)
      {
        THBlas_(axpy)(outDim, -lr, lgradWeightData, 1, lweightData, 1);
      }
      else
      {
        for (k=0; k < outDim; k++)
        {
          lweightData[k] -= lgradWeightData[k]*lr;
        }
      }
    }
  }
}


void THNN_(IndexLinear_accUpdateGradParameters)(
          THNNState *state,
          THLongTensor *keys,
          int64_t keysOffset,
          THTensor *values,
          THLongTensor *sizes,
          THLongTensor *cumSumSizes,
          THTensor *gradOutput,
          THTensor *weight,
          THTensor *bias,
          accreal weightDecay_,
          accreal scale_)
{
  scalar_t weightDecay = TH_CONVERT_ACCREAL_TO_REAL(weightDecay_);
  scalar_t scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  /* Retrieve all the dimensions of the problem */
  int64_t batchSize = THLongTensor_size(sizes, 0);
  int64_t outDim = THTensor_(size)(bias, 0);
  int64_t woutDim = THTensor_(size)(weight, 1);
  int maxNormalize = woutDim - outDim;
  THArgCheck(THNN_(checkKeysValues)(keys, values), 1, "Keys and values should have the same number of elements");

  /* Access the storage data/strides */
  scalar_t* gradOutputData = gradOutput->data<scalar_t>();
  scalar_t* valuesData =values->data<scalar_t>();
  scalar_t* weightData = weight->data<scalar_t>();
  scalar_t* biasData = bias->data<scalar_t>();
  int64_t weightStride0 = weight->stride(0);
  int64_t* keysData = THLongTensor_data(keys);
  int64_t* sizesData = THLongTensor_data(sizes);

  /* Make sure these inputs are contiguous to accelerate computations */
  THArgCheck(THLongTensor_isContiguous(keys), 1, "keys vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(values), 3, "values vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(gradOutput), 6, "gradOutput vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(weight), 7, "weight matrix must be contiguous");
  THArgCheck(THTensor_(isContiguous)(bias), 8, "bias matrix must be contiguous");

  int i,j,k;

  /* Separate cases: output dimension is == 1, or > 1
   * This allows for some optimizations.
   * No multithreading here as this could
   * corrupt the results (hogwild style) */
  if (outDim == 1)
  {
    if (maxNormalize)
    {
        int64_t offset = 0;
        for (j = 0; j < batchSize; j++)
        {
          scalar_t* lgradOutputData = gradOutputData + j;
          *biasData -= *lgradOutputData * scale;
          scalar_t val = *lgradOutputData * scale;
          for (i = 0; i < sizesData[j]; i++)
          {
            int64_t idx = weightStride0*(keysData[offset] + keysOffset) + maxNormalize;
            weightData[idx-1] -= weightData[idx]*val*weightData[idx-2];
            weightData[idx] -= (val*valuesData[offset] - weightDecay * weightData[idx])*weightData[idx-2];
            offset++;
          }
        }

        offset = 0;
        for (j = 0; j < batchSize; j++)
        {
          for (i = 0; i < sizesData[j]; i++)
          {
            int64_t idx = weightStride0*(keysData[offset] + keysOffset) + maxNormalize;
            weightData[idx-2] = 0;
            offset++;
          }
        }
    }
    else
    {
      if (weightDecay)
      {
        int64_t offset = 0;
        for (j = 0; j < batchSize; j++)
        {
          scalar_t* lgradOutputData = gradOutputData + j;
          *biasData -= *lgradOutputData * scale;
          scalar_t val = *lgradOutputData * scale;
          for (i = 0; i < sizesData[j]; i++)
          {
            int64_t idx = weightStride0*(keysData[offset] + keysOffset);
            weightData[idx] -= val * valuesData[offset] + weightData[idx] * weightDecay;
            offset++;
          }
        }
      }
      else
      {
        int64_t offset = 0;
        for (j = 0; j < batchSize; j++)
        {
          scalar_t val = gradOutputData[j] * scale;
          for (i = 0; i < sizesData[j]; i++)
          {
            weightData[(keysData[offset] + keysOffset)*weightStride0] -= val * valuesData[offset];
            offset++;
          }
          *biasData -= val;
        }
      }
    }
  }
  else {
    int64_t offset = 0;
    for (j = 0; j < batchSize; j++)
    {
      scalar_t* lgradOutputData = gradOutputData + j*outDim;
      scalar_t* lweightData = weightData;
      THVector_(cadd)(biasData, biasData, lgradOutputData, -scale, outDim);
      for (i = 0; i < sizesData[j]; i++)
      {
        scalar_t val = valuesData[offset] * scale;
        scalar_t wd = weightDecay;

        // Max normalize case
        if (maxNormalize)
        {
          lweightData = weightData + weightStride0*(keysData[offset] + keysOffset) + (maxNormalize-2);
          val *= lweightData[0];
          wd *= lweightData[0];
          for (k=0; k < outDim; k++)
          {
            lweightData[1] -= lweightData[k+2]*scale*lgradOutputData[k]*lweightData[0];
          }
          lweightData += 2;
        }
        else
        {
          lweightData = weightData + weightStride0*(keysData[offset] + keysOffset);
        }

        /* We do sparse weight decay.
         * We think it makes more sense. */
        if (weightDecay)
        {
          if (outDim > THNN_SPARSE_OUTDIM_THRESHOLD)
          {
            THBlas_(axpy)(outDim, -wd, lweightData, 1, lweightData, 1);
          }
          else
          {
            for (k=0; k < outDim; k++)
            {
              lweightData[k] -= wd * lweightData[k];
            }
          }
        }

        if (outDim > THNN_SPARSE_OUTDIM_THRESHOLD)
        {
          THBlas_(axpy)(outDim, -val, lgradOutputData, 1, lweightData, 1);
        }
        else
        {
          for (k=0; k < outDim; k++)
          {
            lweightData[k] -= val * lgradOutputData[k];
          }
        }
        offset++;
      }
    }

    /* Max Normalize case:
     * Reset the smart update scaling if
     * one does it batch-wise.
     * TODO: Decide what to do with that piece of code.
     * NB: If the code belowe is uncommented, so should the commented
     * code in IndexLinear:zeroGradParameters() */

    /*
    if (maxNormalize)
    {
      offset = 0;
      for (j = 0; j < batchSize; j++)
      {
        scalar_t* lweightData = weightData;
        for (i = 0; i < sizesData[j]; i++)
        {
          scalar_t val = valuesData[offset] * scale;
          scalar_t wd = weightDecay;

          lweightData = weightData + weightStride0*(keysData[offset] + keysOffset) + (maxNormalize-2);
          lweightData[0] = 0;
          offset++;
        }
      }
    }
    */
  }
  return;
}

void THNN_(IndexLinear_accGradParameters)(
          THNNState *state,
          THLongTensor *keys,
          int64_t keysOffset,
          THTensor *values,
          THLongTensor *sizes,
          THLongTensor *cumSumSizes,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          THTensor *valuesBuffer,
          accreal weightDecay_,
          accreal scale_)
{
  scalar_t scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  /* Retrieve all the dimensions of the problem */
  int64_t batchSize = THLongTensor_size(sizes, 0);
  int64_t keysSize = THLongTensor_size(keys, 0);
  int64_t outDim = THTensor_(size)(bias, 0);
  int64_t woutDim = THTensor_(size)(weight, 1);
  int64_t maxNormalize = (woutDim - outDim) > 0 ?1:0;
  THArgCheck(THNN_(checkKeysValues)(keys, values), 1, "Keys and values should have the same number of elements");
  int64_t* sizesData = THLongTensor_data(sizes);

  /* COmpute the cumulative sizes */
  THLongTensor* cumSizes = THLongTensor_new();
  THLongTensor_cumsum(cumSizes, sizes, 0);
  int64_t* cumSizesData = THLongTensor_data(cumSizes);

  /* Resize the gradWeight buffer to keep it dense.
   * That speeds up updates A LOT assuming random mem access. */
  THTensor_(resize2d)(gradWeight, keysSize, outDim * (maxNormalize>0?2:1));

  /* Access the storage data/strides */
  scalar_t* gradOutputData = gradOutput->data<scalar_t>();
  scalar_t* valuesData =values->data<scalar_t>();
  scalar_t* gradWeightData = gradWeight->data<scalar_t>();
  scalar_t* gradBiasData = gradBias->data<scalar_t>();

  /* Make sure these inputs are contiguous to accelerate computations */
  THArgCheck(THLongTensor_isContiguous(keys), 1, "keys vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(values), 3, "values vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(gradOutput), 6, "gradOutput vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(gradWeight), 7, "gradWeight must be contiguous");
  THArgCheck(THTensor_(isContiguous)(gradBias), 8, "gradBias vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(weight), 9, "weight must be contiguous");
  THArgCheck(THTensor_(isContiguous)(bias), 10, "bias vector must be contiguous");
  THArgCheck(THTensor_(isContiguous)(valuesBuffer), 11, "valuesBuffer must be contiguous");

  int i,j,k;

  /* Separate cases: output dimension is == 1, or > 1
   * This allows for some optimizations.
   * No multithreading here as this could
   * corrupt the results (hogwild style) */
  if (outDim == 1)
  {
    for (j = 0; j < batchSize; j++)
    {
      int64_t offset = j==0?0:cumSizesData[j-1];
      scalar_t val = gradOutputData[j] * scale;
      scalar_t* lgradWeightData = gradWeightData + offset;
      scalar_t* lvaluesData = valuesData + offset;
      int64_t end = sizesData[j];

      if (maxNormalize)
      {
        lgradWeightData += offset;
        i = 0;
        for(;i < end; i++)
        {
          lgradWeightData[2*i] = val;
          lgradWeightData[2*i+1] = val * lvaluesData[i];
        }
      }
      else
      {
        i = 0;
        for(;i < end-4; i += 4)
        {
          lgradWeightData[i] = val * lvaluesData[i];
          lgradWeightData[i+1] = val * lvaluesData[i+1];
          lgradWeightData[i+2] = val * lvaluesData[i+2];
          lgradWeightData[i+3] = val * lvaluesData[i+3];
        }

        for(; i < end; i++)
        {
          lgradWeightData[i] = val * lvaluesData[i];
        }
      }
      *gradBiasData += val;
      offset += end;
    }
  }
  else {
    for (j = 0; j < batchSize; j++)
    {
      int64_t offset = j==0?0:cumSizesData[j-1];
      scalar_t* lgradOutputData = gradOutputData + j*outDim;
      scalar_t* lgradWeightData = gradWeightData;
      THVector_(cadd)(gradBiasData, gradBiasData, lgradOutputData, scale, outDim);
      for (i = 0; i < sizesData[j]; i++)
      {
        scalar_t val = valuesData[offset] * scale;
        lgradWeightData = gradWeightData + offset*outDim;
        if (maxNormalize)
        {
          lgradWeightData += offset*outDim;
          k = 0;
          for(;k < outDim-4; k += 4)
          {
            lgradWeightData[k] = lgradOutputData[k]*scale;
            lgradWeightData[k+1] = lgradOutputData[k+1]*scale;
            lgradWeightData[k+2] = lgradOutputData[k+2]*scale;
            lgradWeightData[k+3] = lgradOutputData[k+3]*scale;
          }

          for(; k < outDim; k++)
          {
            lgradWeightData[k] = lgradOutputData[k]*scale;
          }
          lgradWeightData += outDim;
        }
        k = 0;
        for(;k < outDim-4; k += 4)
        {
          lgradWeightData[k] = val * lgradOutputData[k];
          lgradWeightData[k+1] = val * lgradOutputData[k+1];
          lgradWeightData[k+2] = val * lgradOutputData[k+2];
          lgradWeightData[k+3] = val * lgradOutputData[k+3];
        }

        for(; k < outDim; k++)
        {
          lgradWeightData[k] = val * lgradOutputData[k];
        }
        offset++;
      }
    }
  }
  THLongTensor_free(cumSizes);
  return;
}
#endif
