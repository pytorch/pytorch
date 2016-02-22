#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LookupTable.c"
#else

static void THNN_(LookupTable_resetCount)(
          THInteger_t *count_data,
          THIndexTensor *input)
{
  int i;
  THIndex_t *input_data = THIndexTensor_(data)(input);
  long numel = THIndexTensor_(nElement)(input);

  for (i = 0; i<numel; i++)
  {
    long k = input_data[i] - 1;
    count_data[k] = 0;
  }
  for (i = 0; i<numel; i++)
  {
    long k = input_data[i] - 1;
    count_data[k]++;
  }
}

void THNN_(LookupTable_accGradParameters)(
          THNNState *state,
          THIndexTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THIntegerTensor *count,
          THTensor *sorted,
          THTensor *indices,
          bool scaleGradByFreq,
          int paddingValue,
          real scale)
{
  long i;
  THInteger_t *count_data = NULL;

  if (scaleGradByFreq)
  {
    THIntegerTensor_(resize1d)(count, gradWeight->size[0]);
    count_data = THIntegerTensor_(data)(count);
  }

  if (!THTensor_(isContiguous)(gradWeight))
    THError("gradWeight must be contiguous");
  if (!THIndexTensor_(isContiguous)(input))
    THError("input must be contiguous");
  if (THIndexTensor_(nDimension)(input) != 1 && THIndexTensor_(nDimension)(input) != 2)
    THError("input must be a vector or matrix");

  THIndex_t *input_data = THIndexTensor_(data)(input);
  long numel = THIndexTensor_(nElement)(input);
  long numw = THTensor_(size)(gradWeight, 0);

  // check that inputs are all within range
  for (i=0; i<numel; i++)
    if (input_data[i] < 1 || input_data[i] > numw)
      THError("input out of range");

  gradOutput = THTensor_(newContiguous)(gradOutput);

  real *gw = THTensor_(data)(gradWeight);
  real *go = THTensor_(data)(gradOutput);
  long stride = THTensor_(stride)(gradWeight, 0);

  if (count_data)
    THNN_(LookupTable_resetCount)(count_data, input);

#ifdef _OPENMP
  if (numel > 1000)
  {
    // The strategy is to parallelize over sections of the vocabulary, so that
    // thread 1 handles updates to gradWeight[0..nVocab/nThreads]. Every thread
    // has to traverse the entire input, but the dominating factor is the axpy
    // BLAS call.
    #pragma omp parallel private(i)
    {
      int tid = omp_get_thread_num();
      int nthreads = omp_get_num_threads();

      long start = tid * (numw/nthreads + 1);
      long end = start + (numw/nthreads + 1);
      for (i=0; i<numel; i++)
      {
        if (input_data[i] != paddingValue)
        {
            long k = input_data[i] - 1;
            if (k >= start && k < end)
            {
                real scale_ = scale;
                if (count_data) scale_ /= count_data[k];
                THBlas_(axpy)(stride, scale_, go + i*stride, 1, gw + k*stride, 1);
            }
        }
      }
    }

    THTensor_(free)(gradOutput);
    return;
  }
#endif

  for (i=0; i<numel; i++)
  {
    if (input_data[i] != paddingValue)
    {
        long k = input_data[i] - 1;
        real scale_ = scale;
        if (count_data) scale_ /= count_data[k];
        THBlas_(axpy)(stride, scale_, go + i*stride, 1, gw + k*stride, 1);
     }
  }

  THTensor_(free)(gradOutput);
}

#endif
